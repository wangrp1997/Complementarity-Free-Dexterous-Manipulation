"""Play a cached fingertip grasp, hold it, then start a slow yaw exploration.

    conda activate free
    python -m ours.tactile_explore.view_explore --obj cube

Loads docs/data/grasp_init/initial_grasps.json. The hand holds the cached
pose, squeezes if the object is still sliding, then yaws the loaded
fingertip contacts together. space = pause, R = next cached grasp.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import mujoco
import mujoco.viewer
import numpy as np

from ours.grasp_init.close_grasp import OBJECTS, contact_report, load_model
from ours.grasp_init.view_grasp import (
    _apply,
    _lift_goal_ghost,
    _load_rows,
    _print_row,
    _resolve_obj,
)
from ours.tactile_explore.explore import TactileExplore

CACHE = _REPO / "docs/data/grasp_init/initial_grasps.json"


def _run_headless(model, data, ctrl0, seconds: float) -> int:
    explorer = TactileExplore(model)
    explorer.reset(data, ctrl0)
    n = int(seconds / model.opt.timestep)
    last = -1
    for _ in range(n):
        data.ctrl[:] = explorer.step(data)
        if np.isfinite(data.qpos).all():
            mujoco.mj_step(model, data)
        else:
            print("qpos became non-finite")
            return 1
        mark = int(explorer.tick * model.opt.timestep * 2)
        if mark != last:
            last = mark
            print(explorer.status(data), flush=True)
    rep = contact_report(model, data)
    print(
        f"done  phase={explorer.phase}  z={data.qpos[18]:.3f}  "
        f"ft={rep['n_fingertip']}  palm={rep['palm_fn']:.3f}  {explorer.note}"
    )
    held = explorer.phase == "explore" and float(data.qpos[18]) > 0.045 and rep["palm_fn"] < 1e-3
    return 0 if held else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", default="cube")
    parser.add_argument("--seed", type=int, default=0, help="which cached grasp")
    parser.add_argument("--cache", default=str(CACHE))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args(argv)

    obj = _resolve_obj(args.obj)
    cache = pathlib.Path(args.cache)
    try:
        rows = _load_rows(obj, cache)
    except json.JSONDecodeError:
        raise SystemExit(f"{cache} is not valid JSON yet. Wait for the grasp cache to finish writing.")
    idx = {"v": args.seed % len(rows)}
    pending = {"v": None}
    row = rows[idx["v"]]
    _print_row(obj, idx["v"], len(rows), row)

    model = load_model(obj)
    _lift_goal_ghost(model)
    data = mujoco.MjData(model)
    ctrl = _apply(model, data, row)
    explorer = TactileExplore(model)
    explorer.reset(data, ctrl)

    if args.headless:
        return _run_headless(model, data, ctrl, args.seconds)

    paused = {"v": False}

    def on_key(keycode):
        ch = chr(keycode) if 0 <= keycode < 256 else ""
        if ch == " ":
            paused["v"] = not paused["v"]
            print("paused" if paused["v"] else "running")
        elif ch in ("R", "r"):
            pending["v"] = (idx["v"] + 1) % len(rows)

    print("viewer: space=暂停  R=下一条缓存  先稳住，再绕竖直轴慢转")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = data.qpos[16:19]
        viewer.cam.distance = 0.32
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -25.0
        last_print = 0.0
        while viewer.is_running():
            step_start = time.time()
            if pending["v"] is not None:
                idx["v"] = pending["v"]
                pending["v"] = None
                _print_row(obj, idx["v"], len(rows), rows[idx["v"]])
                ctrl = _apply(model, data, rows[idx["v"]])
                explorer.reset(data, ctrl)
            if not paused["v"]:
                data.ctrl[:] = explorer.step(data)
                if np.isfinite(data.qpos).all():
                    mujoco.mj_step(model, data)
                now = time.time()
                if now - last_print > 0.5:
                    last_print = now
                    print(explorer.status(data), flush=True)
            viewer.sync()
            leftover = model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)
    return 0


if __name__ == "__main__":
    sys.exit(main())

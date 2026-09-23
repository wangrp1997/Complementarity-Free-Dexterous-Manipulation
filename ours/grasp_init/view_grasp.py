"""Watch a held Allegro fingertip grasp in the MuJoCo viewer.

    conda activate free
    python ours/grasp_init/view_grasp.py --obj cube

Loads docs/data/grasp_init/initial_grasps.json. Keys: space = pause,
R = next cached grasp of the same object.
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

from ours.grasp_init.close_grasp import (
    OBJECTS,
    contact_report,
    load_model,
    tactile_report,
)
from ours.tactile_observation import VirtualTactile

CACHE = _REPO / "docs/data/grasp_init/initial_grasps.json"


def _resolve_obj(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"foam_brick": "foambrick", "waterbottle": "water_bottle"}
    key = aliases.get(key, key)
    if key not in OBJECTS:
        known = ", ".join(OBJECTS)
        raise SystemExit(f"unknown object {name!r}. choose one of:\n  {known}")
    return key


def _load_rows(obj: str, cache_path: pathlib.Path) -> list[dict]:
    if not cache_path.is_file():
        raise SystemExit(
            f"no cache at {cache_path}. build it with:\n"
            f"  python -m ours.grasp_init.run_grasp_init"
        )
    payload = json.loads(cache_path.read_text())
    raw = payload.get("objects", {}).get(obj) or []
    rows = [row for row in raw if row.get("ok")]
    if not rows:
        notes = sorted({row.get("note") or "abandoned" for row in raw}) or ["abandoned"]
        raise SystemExit(
            f"cache has no fingertip grasp for {obj} "
            f"({'; '.join(notes)}). this object was dropped."
        )
    return rows


def _print_row(obj: str, idx: int, n: int, row: dict) -> None:
    loads = row.get("tactile_loads")
    extra = f"  tact={row.get('n_tactile')} {np.round(np.asarray(loads), 3)}" if loads is not None else ""
    print(
        f"loaded {obj} [{idx + 1}/{n}]  z={row['final_z']:.3f}  "
        f"ft={row['n_fingertip']}  palm={row['palm_fn']:.3f}{extra}",
        flush=True,
    )


def _lift_goal_ghost(model: mujoco.MjModel, z: float = 0.36) -> None:
    """Keep FREE's target ghost, parked once above the hand. Do not touch
    the model again after the viewer starts — that races the renderer."""
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not name.startswith("goal"):
            continue
        if model.geom_rgba[gid, 3] <= 0.01:
            continue
        model.geom_rgba[gid, 3] = min(float(model.geom_rgba[gid, 3]), 0.45)
        body_z = float(model.body_pos[model.geom_bodyid[gid], 2])
        model.geom_pos[gid, 2] = z - body_z


def _apply(model, data, row: dict) -> np.ndarray:
    # Never mj_resetData while launch_passive is using this MjData.
    q16 = np.asarray(row["q16"], dtype=float)
    obj_pos = np.asarray(row["obj_pos"], dtype=float)
    obj_quat = np.asarray(row["obj_quat"], dtype=float)
    stored = row.get("ctrl")
    ctrl = q16.copy() if stored is None else np.asarray(stored, dtype=float)
    data.qpos[:16] = q16
    data.qpos[16:19] = obj_pos
    data.qpos[19:23] = obj_quat
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    if hasattr(data, "qacc_warmstart"):
        data.qacc_warmstart[:] = 0.0
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    return ctrl


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True, help="one of: " + ", ".join(OBJECTS))
    parser.add_argument("--seed", type=int, default=0, help="which cached grasp")
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args(argv)

    obj = _resolve_obj(args.obj)
    rows = _load_rows(obj, pathlib.Path(args.cache))
    idx = {"v": args.seed % len(rows)}
    pending = {"v": None}
    row = rows[idx["v"]]
    _print_row(obj, idx["v"], len(rows), row)

    model = load_model(obj)
    _lift_goal_ghost(model)
    data = mujoco.MjData(model)
    sensor = VirtualTactile()
    ctrl = _apply(model, data, row)
    paused = {"v": False}

    def on_key(keycode):
        # GLFW callback: only queue. Do not touch model/data here.
        ch = chr(keycode) if 0 <= keycode < 256 else ""
        if ch == " ":
            paused["v"] = not paused["v"]
            print("paused" if paused["v"] else "running")
        elif ch in ("R", "r"):
            pending["v"] = (idx["v"] + 1) % len(rows)

    print("viewer: 左键旋转  右键平移  滚轮缩放  space=暂停  R=下一条缓存")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = data.qpos[16:19]
        viewer.cam.distance = 0.32
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -25.0
        while viewer.is_running():
            step_start = time.time()
            if pending["v"] is not None:
                idx["v"] = pending["v"]
                pending["v"] = None
                _print_row(obj, idx["v"], len(rows), rows[idx["v"]])
                ctrl = _apply(model, data, rows[idx["v"]])
            if not paused["v"]:
                data.ctrl[:] = ctrl
                if np.isfinite(data.qpos).all():
                    mujoco.mj_step(model, data)
            viewer.sync()
            leftover = model.opt.timestep - (time.time() - step_start)
            if leftover > 0:
                time.sleep(leftover)
        rep = contact_report(model, data)
        tac = tactile_report(sensor, model, data)
        print(
            f"exit  z={data.qpos[18]:.3f}  ft={rep['n_fingertip']}  "
            f"palm={rep['palm_fn']:.3f}  tact={tac['n_tactile']} "
            f"loads={np.round(tac['tactile_loads'], 3)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

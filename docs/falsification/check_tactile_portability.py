"""Can the existing virtual-tactile interface be reused on trifinger / fingertips?

`ours/tactile_observation.py::VirtualTactile` hardcodes the Allegro sensor model:
four geoms named ``fingertip0..3``, four sites named ``ftp_0..3``, and a geom
named ``palm``. This script asks, for each hand, which of those names exist in
the hand's MuJoCo model. Read-only; FREE files are untouched.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ours.tactile_observation import FINGERTIP_GEOMS, FINGERTIP_NAMES

HANDS = {
    "allegro": ("examples.mpc.allegro.cube.params", "envs.allegro_env"),
    "trifinger": ("examples.mpc.trifinger.cube.params", "envs.trifinger_env"),
    "fingertips": ("examples.mpc.fingertips.cube.params", "envs.fingertips_env"),
}


def main():
    print(f"{'hand':<12}{'fingertip0..3':>14}{'ftp_0..3 sites':>16}{'palm':>7}"
          f"{'object geoms':>14}")
    print("-" * 63)
    for hand, (params_mod, env_mod) in HANDS.items():
        params_cls = importlib.import_module(params_mod).ExplicitMPCParams
        target = "in-air" if hand == "fingertips" else "rotation"
        param = params_cls(rand_seed=0, target_type=target)
        param.headless_ = True
        env_cls = importlib.import_module(env_mod).MjSimulator
        env = env_cls(param)
        model = env.model_
        geoms = sum(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0
                    for n in FINGERTIP_GEOMS)
        sites = sum(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) >= 0
                    for n in FINGERTIP_NAMES)
        palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm") >= 0
        obj = sum(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) >= 0
                  for n in param.object_names_)
        print(f"{hand:<12}{f'{geoms}/4':>14}{f'{sites}/4':>16}{str(palm):>7}{f'{obj}':>14}")
        if getattr(env, "viewer_", None) is not None:
            env.viewer_.close()
    print("\nallegro-style VirtualTactile needs all of the first three columns;")
    print("where they are missing the module cannot be reused as-is.")


if __name__ == "__main__":
    main()

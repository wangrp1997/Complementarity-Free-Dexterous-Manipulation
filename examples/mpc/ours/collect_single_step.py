"""Single-physics-step transition collector (diagnostic; FREE is never modified).

Why this exists
---------------
The rigid contact kinematic identity ``A x + J dq = 0`` has only ever been
evaluated over a full control interval (0.1 s = frame_skip * timestep = 50
physics steps). At that scale it fails, but the failure is confounded with
rolling and contact switching inside the interval. This collector records one
transition per *physics step* (dt = model.opt.timestep = 0.002 s) so the
identity can be tested at the scale where it is supposed to hold.

Design constraints (docs/falsification/CONVENTIONS.md)
------------------------------------------------------
* FREE files are never modified. FREE's MPC is used only as an action source,
  exactly as in ``collect_tactile_dataset.py``.
* Contact quantities are evaluated on a scratch ``MjData`` copy. No
  ``mj_forward`` is inserted into the recorded rollout, so the recorded
  trajectory is bit-identical to what ``MjSimulator.step(action)`` produces.
  The collector asserts this.
* ``mj_jac`` rows are reimplemented here and cross-checked against FREE's own
  ``Contact.detect_once`` output at every recorded step.
* Contact forces / ground contacts are labels: evaluation-only, never inputs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from examples.mpc.ours.collect_tactile_dataset import (
    ACCEPTED_SOLVER_STATUS,
    clone_env,
    create_episode,
    integration_state,
)
from planning.mpc_explicit import MPCExplicit

SCHEMA_VERSION = 1
# Raw MuJoCo qpos order for this model is [hand(16), object_pos(3), object_quat_wxyz(4)].
QPOS_ORDER = "hand_qpos(16), object_pos(3), object_quat_wxyz(4)"
# Contact rows returned by Contact.reformat are reordered to [object(6), hand(16)].
CONTACT_ROW_ORDER = "object_velocity(6), hand_velocity(16)"
FORCE_THRESHOLD = 1e-8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objects", nargs="+", default=["cube", "mug", "stick"])
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--intervals", type=int, default=6,
                   help="Number of control intervals whose physics steps are recorded")
    p.add_argument("--substeps", type=int, default=50,
                   help="Physics steps recorded per control interval (frame_skip_ = 50)")
    p.add_argument("--warmup", type=int, default=0,
                   help="Control steps executed without recording substeps")
    p.add_argument("--max-control-steps", type=int, default=40)
    p.add_argument("--no-validate", action="store_true",
                   help="Skip the bit-identity check against env.step(action)")
    p.add_argument("--output", type=Path, default=None, help="Must not already exist")
    args = p.parse_args()
    if min(args.trials, args.intervals, args.substeps, args.max_control_steps) <= 0:
        p.error("trials, intervals, substeps and max-control-steps must be positive")
    if args.warmup < 0 or args.warmup + args.intervals > args.max_control_steps:
        p.error("require 0 <= warmup and warmup + intervals <= max-control-steps")
    return args


def reorder_to_object_first(rows: np.ndarray, nv: int) -> np.ndarray:
    """Match Contact.reformat: columns [object(6), hand(nv-6)]."""
    out = np.zeros((rows.shape[0], nv))
    out[:, :6] = rows[:, nv - 6:]
    out[:, 6:] = rows[:, : nv - 6]
    return out


def snapshot_contacts(env, param):
    """Object contacts at the *current* state, on the caller's scratch copy.

    Returns (jac4, kin3, phi, pairs, forces):
      jac4   (4*n, nv)  FREE's mu-mixed rows [n + mu t1, n + mu t2, n - mu t1, n - mu t2]
      kin3   (3*n, nv)  clean world-frame rows [n, t1, t2] of the object-relative point Jacobian
      phi    (n,)       MuJoCo contact distance
      pairs  (n, 2)     geom ids
      forces (n,)       normal contact force (label)
    """
    model, data = env.model_, env.data_
    object_geoms = {model.geom(n).id for n in param.object_names_}
    nv = model.nv
    jac4_rows, kin3_rows, phi, pairs, forces, indices = [], [], [], [], [], []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 not in object_geoms and g2 not in object_geoms:
            continue
        frame = contact.frame.reshape((-1, 3)).T
        frame_pmd = np.hstack((frame, -frame[:, -2:]))
        jac1 = np.zeros((3, nv))
        jac2 = np.zeros((3, nv))
        mujoco.mj_jac(model, data, jacp=jac1, jacr=None, point=contact.pos,
                      body=int(model.geom_bodyid[g1]))
        mujoco.mj_jac(model, data, jacp=jac2, jacr=None, point=contact.pos,
                      body=int(model.geom_bodyid[g2]))
        local = frame_pmd.T @ (-(jac2 - jac1) if g1 in object_geoms else (jac2 - jac1))
        mu = float(param.mu_object_)
        jac4_rows.append(reorder_to_object_first(local[0] + mu * local[1:5], nv))
        kin3_rows.append(reorder_to_object_first(local[:3], nv))
        phi.append(float(contact.dist))
        pairs.append((g1, g2))
        indices.append(i)
        forces.append(0.0)
    if pairs:
        # mj_contactForce needs efc_address, and the trailing mj_collision inside
        # Contact.detect_once invalidates it (measured: every force read as 0.0
        # while a clean forward pass reports up to 13.8 N). Recompute on a clean
        # forward pass and verify the contact set and ordering are unchanged.
        force_env = clone_env(env)
        mujoco.mj_forward(force_env.model_, force_env.data_)
        if int(force_env.data_.ncon) != int(data.ncon):
            raise AssertionError("contact set changed between collision and forward pass")
        wrench = np.zeros(6)
        for slot, index in enumerate(indices):
            g1, g2 = pairs[slot]
            contact = force_env.data_.contact[index]
            if int(contact.geom1) != g1 or int(contact.geom2) != g2:
                raise AssertionError("contact ordering changed between collision and forward pass")
            if contact.efc_address >= 0:
                mujoco.mj_contactForce(force_env.model_, force_env.data_, index, wrench)
            forces[slot] = float(wrench[0])
    return (np.asarray(jac4_rows, dtype=float).reshape(-1, nv),
            np.asarray(kin3_rows, dtype=float).reshape(-1, nv),
            np.asarray(phi, dtype=float),
            np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
            np.asarray(forces, dtype=float))


def raw_qpos(data):
    """[hand(16), object(7)] in raw MuJoCo order."""
    return np.concatenate((data.qpos[:16], data.qpos[16:23])).astype(float).copy()


def record_interval(env, contact, param, sensor, action, args):
    """Execute one control interval physics step by physics step and record each transition."""
    target = env.get_jpos() + np.asarray(action, dtype=float).reshape(-1)
    # Isolation control: a bare mj_step rollout with the same control and no
    # contact evaluation must land bit-identically. This is what proves that
    # running detect_once on a scratch copy did not disturb the dynamics.
    reference = None if args.no_validate else clone_env(env)
    if reference is not None:
        reference.data_.ctrl = target
    records = []
    for substep in range(args.substeps):
        scratch = clone_env(env)
        # detect_once performs mj_forward + mj_collision; doing it on the scratch
        # copy keeps the recorded rollout free of extra mj_forward calls.
        _, jac_mat = contact.detect_once(scratch)
        jac4, kin3, phi, pairs, forces = snapshot_contacts(scratch, param)
        n_contacts = int(phi.shape[0])
        # Contact.detect_once reserves only max_ncon_ blocks (20 for cube, 15 for
        # mug/stick) and silently truncates anything beyond that, so the row check
        # compares the first min(n_contacts, max_ncon_) blocks and the truncation
        # itself is recorded rather than treated as an error.
        n_free = int(jac_mat.shape[0] // 4)
        n_kept = min(n_contacts, n_free)
        if not np.array_equal(jac_mat[: 4 * n_kept], jac4[: 4 * n_kept]):
            max_diff = float(np.max(np.abs(jac_mat[: 4 * n_kept] - jac4[: 4 * n_kept]))) if n_kept else np.inf
            raise AssertionError(
                f"contact row reimplementation disagrees with FREE detect_once (max diff {max_diff})")
        observation = sensor.observe(scratch)
        tactile = np.asarray(observation["tactile"])
        qpos_before = raw_qpos(scratch.data_)
        qvel_before = scratch.data_.qvel.copy()
        env.data_.ctrl = target
        mujoco.mj_step(env.model_, env.data_)
        records.append(dict(
            substep=np.int32(substep),
            qpos_before=qpos_before,
            qpos_after=raw_qpos(env.data_),
            qvel_before=qvel_before,
            qvel_after=env.data_.qvel.copy(),
            ctrl=np.asarray(target, dtype=float).copy(),
            n_contacts=np.int32(n_contacts),
            free_max_ncon=np.int32(n_free),
            contacts_truncated=np.bool_(n_contacts > n_free),
            n_loaded=np.int32(int(np.count_nonzero(forces > FORCE_THRESHOLD))),
            min_distance=np.float64(phi.min()) if n_contacts else np.float64(np.nan),
            max_force=np.float64(forces.max()) if n_contacts else np.float64(0.0),
            n_active_fingers=np.int32(int(np.count_nonzero(tactile[:, 0]))),
            jac4=jac4,
            kin3=kin3,
            phi=phi,
            pairs=pairs,
            forces=forces,
        ))
    validation = None
    if reference is not None:
        for _ in range(args.substeps):
            mujoco.mj_step(reference.model_, reference.data_)
        same = bool(np.array_equal(reference.data_.qpos, env.data_.qpos)
                    and np.array_equal(reference.data_.qvel, env.data_.qvel))
        validation = {
            "bit_identical_to_bare_mj_step": same,
            "qpos_max_abs_diff": float(np.max(np.abs(reference.data_.qpos - env.data_.qpos))),
            "qvel_max_abs_diff": float(np.max(np.abs(reference.data_.qvel - env.data_.qvel))),
            "n_physics_steps": int(args.substeps),
        }
        if not same:
            raise AssertionError("recorded rollout diverged from a bare mj_step rollout")
    return records, validation


def pack(records, ragged_keys):
    """Pack a list of dicts into an npz-friendly dict with offsets for ragged fields."""
    packed = {}
    if not records:
        return packed
    for key in records[0]:
        values = [np.asarray(record[key]) for record in records]
        if key in ragged_keys:
            packed[key] = np.concatenate(values, axis=0).astype(np.float32)
            packed[f"{key}_offsets"] = np.concatenate(
                ([0], np.cumsum([len(v) for v in values]))).astype(np.int64)
        else:
            packed[key] = np.stack(values).astype(np.float32)
    return packed


RAGGED_KEYS = {"jac4", "kin3", "phi", "pairs", "forces"}


def collect_episode(obj, trial, args, output_dir):
    param, env, contact, sensor, mpc = create_episode(obj, trial, "nominal")
    episode_dir = output_dir / f"allegro_{obj}_nominal_trial{trial:03d}"
    episode_dir.mkdir(exist_ok=False)
    all_records, interval_meta, validations = [], [], []
    sol_guess = None
    stop_reason = "max_steps"
    for control_step in range(args.max_control_steps):
        state = env.get_state()
        phi_vec, jac_mat = contact.detect_once(env)
        try:
            sol = mpc.plan_once(param.target_p_, param.target_q_, state, phi_vec, jac_mat,
                                sol_guess=sol_guess)
        except RuntimeError as exc:
            interval_meta.append({"control_step": control_step, "status": "exception",
                                  "detail": str(exc)[:300]})
            stop_reason = "solver_exception"
            break
        status = str(sol["solve_status"])
        action = np.asarray(sol["action"], dtype=float).reshape(-1)
        if status not in ACCEPTED_SOLVER_STATUS or action.shape != (16,) or not np.all(np.isfinite(action)):
            interval_meta.append({"control_step": control_step, "status": status, "recorded": False})
            stop_reason = "rejected_solver_result"
            break
        sol_guess = sol["sol_guess"]
        recording = args.warmup <= control_step < args.warmup + args.intervals
        if recording:
            records, validation = record_interval(env, contact, param, sensor, action, args)
            for record in records:
                record["control_step"] = np.int32(control_step)
                record["action"] = action.copy()
            all_records.extend(records)
            validations.append({"control_step": control_step, **(validation or {})})
            interval_meta.append({"control_step": control_step, "status": status, "recorded": True,
                                  "n_contacts_mean": float(np.mean([r["n_contacts"] for r in records])),
                                  "n_active_fingers_mean": float(np.mean([r["n_active_fingers"] for r in records]))})
        else:
            env.step(action)
            interval_meta.append({"control_step": control_step, "status": status, "recorded": False})
        if not np.all(np.isfinite(integration_state(env))):
            stop_reason = "nonfinite_state"
            break
    np.savez_compressed(episode_dir / "steps.npz",
                        schema_version=np.int32(SCHEMA_VERSION),
                        target_position=np.asarray(param.target_p_, dtype=np.float32),
                        target_quaternion=np.asarray(param.target_q_, dtype=np.float32),
                        **pack(all_records, RAGGED_KEYS))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "object": obj, "mode": "nominal", "trial": trial,
        "n_records": len(all_records),
        "n_intervals_recorded": sum(1 for m in interval_meta if m.get("recorded")),
        "substeps_per_interval": args.substeps,
        "stop_reason": stop_reason,
        "mujoco_version": mujoco.__version__,
        "model_path": str(param.model_path_),
        "physics_timestep": float(env.model_.opt.timestep),
        "frame_skip": int(param.frame_skip_),
        "control_interval": float(env.model_.opt.timestep * param.frame_skip_),
        "qpos_order": QPOS_ORDER,
        "contact_row_order": CONTACT_ROW_ORDER,
        "mu_object": float(param.mu_object_),
        "action": "16 joint target increments in rad; target = current hand qpos + action",
        "label_fields": ["forces", "pairs", "phi"],
        "validation_vs_env_step": validations,
        "intervals": interval_meta,
        "limitations": [
            "FREE's MPC is an action source, not a certified controller.",
            "A recorded substep is a one-step replay of the same physical state; it is not an online method.",
            "Only nominal plant is collected; unknown-dyn is a separate question.",
        ],
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    summary = {"object": obj, "records": len(all_records),
               "intervals": metadata["n_intervals_recorded"], "path": str(episode_dir)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    args = parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = Path("docs/data") / f"single_step_{stamp}"
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for obj in args.objects:
        for trial in range(args.trials):
            summaries.append(collect_episode(obj, trial, args, args.output))
            (args.output / "manifest.json").write_text(
                json.dumps({"schema_version": SCHEMA_VERSION, "episodes": summaries},
                           ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"episodes": len(summaries),
                      "records": sum(s["records"] for s in summaries),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

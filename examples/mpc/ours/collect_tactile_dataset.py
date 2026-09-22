"""Collect diagnostic tactile transitions without modifying the FREE algorithm.

Schema v2 separates ideal sensor/visual/proprioceptive observations from
privileged simulation labels and physical parameters. Paired probes replay the
same complete MuJoCo snapshot under FREE's action and small symmetric changes.
This is offline experimental intervention, not a trained or safe controller.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np

from planning.mpc_explicit import MPCExplicit
from ours.tactile_observation import VirtualTactile, flatten_contact_records, CONTACT_RECORD_COLUMNS
from utils import metrics
from utils.plant_mismatch import UNKNOWN_COM, UNKNOWN_FRICTION, UNKNOWN_INERTIA_SCALE

SCHEMA_VERSION = 2
ACCEPTED_SOLVER_STATUS = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
INTEGRATION_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
TRUTH_CONTACT_COLUMNS = ("geom1", "geom2", "distance", "normal_force", "friction_0", "friction_1", "friction_2", "friction_3", "friction_4")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objects", nargs="+", default=["cube"])
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--modes", nargs="+", choices=["nominal", "unknown-dyn"], default=["nominal", "unknown-dyn"])
    p.add_argument("--protocol", choices=["trajectory", "paired"], default="paired")
    p.add_argument("--probe-count", type=int, default=2, help="Maximum accepted snapshots per episode")
    p.add_argument("--probe-joints", nargs="+", type=int, default=[1, 13])
    p.add_argument("--epsilons", nargs="+", type=float, default=[0.002, 0.004])
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--probe-interval", type=int, default=1)
    p.add_argument("--min-fingers", type=int, default=2)
    p.add_argument("--output", type=Path, default=None, help="Must not already exist")
    args = p.parse_args()
    if min(args.trials, args.max_steps, args.probe_count, args.probe_interval) <= 0:
        p.error("trials, max-steps, probe-count and probe-interval must be positive")
    if args.warmup_steps < 0 or not 1 <= args.min_fingers <= 4:
        p.error("warmup-steps must be nonnegative; min-fingers must be 1..4")
    if not args.probe_joints or any(not 0 <= j < 16 for j in args.probe_joints):
        p.error("probe-joints must be 0..15")
    if len(set(args.probe_joints)) != len(args.probe_joints):
        p.error("probe-joints must be unique")
    if any(not np.isfinite(e) or e <= 0 for e in args.epsilons) or len(set(args.epsilons)) != len(args.epsilons):
        p.error("epsilons must be finite, positive and unique")
    return args


def _load_object(obj):
    params_cls = importlib.import_module(f"examples.mpc.allegro.{obj}.params").ExplicitMPCParams
    env_cls = importlib.import_module("envs.allegro_env").MjSimulator
    contact_cls = importlib.import_module("contact.allegro_collision_detection").Contact
    return params_cls, env_cls, contact_cls


def integration_state(env):
    state = np.empty(mujoco.mj_stateSize(env.model_, INTEGRATION_SPEC), dtype=np.float64)
    mujoco.mj_getState(env.model_, env.data_, state, INTEGRATION_SPEC)
    return state


def integration_layout(model):
    fields, offset = [], 0
    for name, flag in mujoco.mjtState.__members__.items():
        bit = int(flag)
        if not name.startswith("mjSTATE_") or bit <= 0 or bit & (bit - 1) or not bit & int(INTEGRATION_SPEC):
            continue
        size = int(mujoco.mj_stateSize(model, flag))
        fields.append({"name": name, "offset": offset, "size": size})
        offset += size
    if offset != mujoco.mj_stateSize(model, INTEGRATION_SPEC):
        raise AssertionError("Unexpected MuJoCo integration state layout")
    return fields


def clone_env(env):
    result = copy.copy(env)
    result.data_ = mujoco.MjData(env.model_)
    mujoco.mj_copyData(result.data_, env.model_, env.data_)
    result.viewer_ = None
    return result


def create_episode(obj, trial, mode):
    params_cls, env_cls, contact_cls = _load_object(obj)
    param = params_cls(rand_seed=trial, target_type="rotation")
    param.headless_ = True
    env = env_cls(param)
    initial = integration_state(env)
    initial_pose = env.get_state().copy()
    if mode == "unknown-dyn":
        # mj_setConst in the existing helper resets qpos in MuJoCo 3.13.
        # Restore the same initial physical/control state, then recompute
        # derived fields under the changed model. Do not modify the helper.
        env.apply_object_plant_mismatch(UNKNOWN_COM, UNKNOWN_INERTIA_SCALE, UNKNOWN_FRICTION)
        mujoco.mj_setState(env.model_, env.data_, initial, INTEGRATION_SPEC)
        mujoco.mj_forward(env.model_, env.data_)
    if not np.array_equal(initial_pose, env.get_state()):
        raise AssertionError("Plant mismatch must not change initial object/hand position")
    return param, env, contact_cls(param), VirtualTactile(param.object_names_), MPCExplicit(param)


def physical_metadata(env, param):
    model = env.model_
    object_geoms = [model.geom(n).id for n in param.object_names_]
    bodies = sorted({int(model.geom_bodyid[g]) for g in object_geoms})
    return {
        "access": "evaluation-only; never required controller inputs",
        "object_bodies": [{"id": b, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                           "mass": float(model.body_mass[b]), "com_body": model.body_ipos[b].tolist(),
                           "inertia_principal": model.body_inertia[b].tolist(),
                           "inertia_frame_quaternion": model.body_iquat[b].tolist()} for b in bodies],
        "geom_friction": [{"id": g, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                           "friction": model.geom_friction[g].tolist(), "priority": int(model.geom_priority[g])}
                          for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]],
        "friction_note": "Object geom friction alone is not the effective pair friction. Actual contact.friction is saved in labels.",
        "mismatch_note": "unknown-dyn uses one fixed COM assignment/inertia scaling/object geom friction change; mass is unchanged",
    }


def observe_sample(env, sensor, param):
    obs = sensor.observe(env)
    sample = {key: np.asarray(obs[key]).copy() for key in ("tactile", "fingertip_positions", "fingertip_positions_object")}
    sample.update(state=env.get_state().copy(), joint_velocity=env.data_.qvel[:16].copy(),
                  time=np.asarray(env.data_.time), contact_records=flatten_contact_records(obs["contact_records"]))
    truth = {key: np.asarray(value, dtype=np.int32) for key, value in obs["diagnostics"].items()}
    truth["integration_state"] = integration_state(env)
    truth["object_velocity_mujoco"] = env.data_.qvel[-6:].copy()
    # Truth-only contact details include geometry separation and effective
    # friction. Compute on an independent copy, exactly as virtual sensing.
    scratch = clone_env(env)
    mujoco.mj_forward(env.model_, scratch.data_)
    object_geoms = {env.model_.geom(n).id for n in param.object_names_}
    records = []
    ground_count = 0
    for i in range(scratch.data_.ncon):
        c = scratch.data_.contact[i]
        if int(c.geom1) not in object_geoms and int(c.geom2) not in object_geoms:
            continue
        wrench = np.zeros(6)
        if c.efc_address >= 0:
            mujoco.mj_contactForce(env.model_, scratch.data_, i, wrench)
        records.append([c.geom1, c.geom2, c.dist, wrench[0], *np.asarray(c.friction)])
        other = int(c.geom2 if int(c.geom1) in object_geoms else c.geom1)
        if (env.model_.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE
                and wrench[0] > sensor.force_threshold):
            ground_count += 1
    truth["n_ground_contacts"] = np.asarray(ground_count, dtype=np.int32)
    truth["contact_truth_records"] = np.asarray(records, dtype=float).reshape(-1, len(TRUTH_CONTACT_COLUMNS))
    return sample, truth


def action_bounds(env, param):
    lower = np.broadcast_to(np.asarray(param.mpc_u_lb_, dtype=float), (16,)).copy()
    upper = np.broadcast_to(np.asarray(param.mpc_u_ub_, dtype=float), (16,)).copy()
    joint_lower, joint_upper = np.full(16, -np.inf), np.full(16, np.inf)
    for j in range(env.model_.njnt):
        qadr = int(env.model_.jnt_qposadr[j])
        if qadr < 16 and env.model_.jnt_limited[j]:
            joint_lower[qadr], joint_upper[qadr] = env.model_.jnt_range[j]
    for actuator in range(env.model_.nu):
        if env.model_.actuator_ctrllimited[actuator]:
            joint = int(env.model_.actuator_trnid[actuator, 0])
            qadr = int(env.model_.jnt_qposadr[joint])
            if qadr < 16:
                ctrl_lower, ctrl_upper = env.model_.actuator_ctrlrange[actuator]
                joint_lower[qadr] = max(joint_lower[qadr], ctrl_lower)
                joint_upper[qadr] = min(joint_upper[qadr], ctrl_upper)
    lower = np.maximum(lower, joint_lower - env.get_jpos())
    upper = np.minimum(upper, joint_upper - env.get_jpos())
    return lower, upper, joint_lower, joint_upper


def qualify_snapshot(env, param, before, truth, action, args):
    if not np.all(np.isfinite(integration_state(env))) or not np.all(np.isfinite(before["tactile"])):
        return [], "nonfinite_snapshot"
    if abs(np.linalg.norm(before["state"][3:7]) - 1) > 1e-6:
        return [], "invalid_quaternion"
    if np.count_nonzero(before["tactile"][:, 0]) < args.min_fingers:
        return [], "insufficient_fingertip_contacts"
    if int(truth["n_ground_contacts"]):
        return [], "object_on_ground"
    lower, upper, joint_lower, joint_upper = action_bounds(env, param)
    if np.any(env.get_jpos() < joint_lower - 1e-6) or np.any(env.get_jpos() > joint_upper + 1e-6):
        return [], "joint_state_out_of_bounds"
    if np.any(action < lower - 1e-7) or np.any(action > upper + 1e-7):
        return [], "baseline_target_out_of_bounds"
    # Reject a perturbation axis unless BOTH signs at ALL amplitudes fit;
    # do not clip probes and thereby destroy symmetric experimental design.
    radius = max(args.epsilons)
    joints = [j for j in args.probe_joints if action[j] - radius >= lower[j]
              and action[j] + radius <= upper[j]]
    return joints, "accepted" if joints else "no_symmetric_probe_within_bounds"


def rotation_angle(q0, q1):
    q0, q1 = np.asarray(q0) / np.linalg.norm(q0), np.asarray(q1) / np.linalg.norm(q1)
    return 2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))


def transition(before, before_truth, after, after_truth, action, base, status, source_step, pair_id=-1, joint=-1, sign=0, epsilon=0.0):
    row = dict(before)
    row.update({f"next_{k}": v for k, v in after.items()})
    row.update(action=np.asarray(action).copy(), base_action=np.asarray(base).copy(),
               action_perturbation=np.asarray(action) - np.asarray(base), solver_status=np.asarray(status, dtype="U64"),
               source_step=np.asarray(source_step, dtype=np.int32), pair_id=np.asarray(pair_id, dtype=np.int32),
               probe_joint=np.asarray(joint, dtype=np.int32), probe_sign=np.asarray(sign, dtype=np.int32),
               probe_epsilon=np.asarray(epsilon))
    labels = dict(before_truth)
    labels.update({f"next_{k}": v for k, v in after_truth.items()})
    labels["rotation_delta_rad"] = np.asarray(rotation_angle(before["state"][3:7], after["state"][3:7]))
    labels["translation_delta"] = after["state"][:3] - before["state"][:3]
    return row, labels


def pack_rows(rows, ragged_fields):
    packed = {"schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32)}
    if not rows:
        return packed
    for key in rows[0]:
        arrays = [np.asarray(row[key]) for row in rows]
        if key in ragged_fields:
            packed[key] = np.concatenate(arrays, axis=0)
            packed[f"{key}_offsets"] = np.concatenate(([0], np.cumsum([len(a) for a in arrays]))).astype(np.int64)
        else:
            packed[key] = np.stack(arrays)
        if packed[key].dtype == object:
            raise TypeError(f"Object dtype forbidden: {key}")
    return packed


def collect_episode(obj, trial, mode, args, output_dir):
    param, env, contact, sensor, mpc = create_episode(obj, trial, mode)
    episode_dir = output_dir / f"allegro_{obj}_{mode}_trial{trial:03d}"
    episode_dir.mkdir(exist_ok=False)
    rows, labels, rejections, solver_attempts = [], [], [], []
    sol_guess = None
    pair_count = 0
    consecutive_success = 0
    success_step = -1
    stop_reason = "max_steps"
    initial_state = env.get_state().copy()
    for step in range(args.max_steps):
        state = env.get_state()
        phi_vec, jac_mat = contact.detect_once(env)
        try:
            sol = mpc.plan_once(param.target_p_, param.target_q_, state, phi_vec, jac_mat, sol_guess=sol_guess)
        except RuntimeError as exc:
            solver_attempts.append({"source_step": step, "status": "exception", "detail": str(exc)[:500], "executed": False})
            stop_reason = "solver_exception"
            break
        status = str(sol["solve_status"])
        action = np.asarray(sol["action"], dtype=float).reshape(-1)
        accepted = status in ACCEPTED_SOLVER_STATUS and action.shape == (16,) and np.all(np.isfinite(action))
        solver_attempts.append({"source_step": step, "status": status, "executed": bool(accepted)})
        if not accepted:
            stop_reason = "rejected_solver_result"
            break
        sol_guess = sol["sol_guess"]
        before, before_truth = observe_sample(env, sensor, param)
        is_probe_step = args.protocol == "paired" and step >= args.warmup_steps and (step - args.warmup_steps) % args.probe_interval == 0
        joints = []
        if is_probe_step:
            joints, reason = qualify_snapshot(env, param, before, before_truth, action, args)
            if not joints:
                rejections.append({"source_step": step, "reason": reason})
        if joints:
            # Clone each complete MjData from exactly the same source. Neither
            # observation nor sibling branches modifies that source snapshot.
            variants = [(-1, 0, 0.0, action.copy())]
            for joint in joints:
                for epsilon in args.epsilons:
                    for sign in (-1, 1):
                        probe_action = action.copy()
                        probe_action[joint] += sign * epsilon
                        variants.append((joint, sign, epsilon, probe_action))
            baseline_end = None
            for joint, sign, epsilon, branch_action in variants:
                branch = clone_env(env)
                if not np.array_equal(integration_state(branch), before_truth["integration_state"]):
                    raise AssertionError("Incomplete probe snapshot copy")
                branch.step(branch_action)
                after, after_truth = observe_sample(branch, sensor, param)
                row, truth_row = transition(before, before_truth, after, after_truth, branch_action, action, status, step, pair_count, joint, sign, epsilon)
                rows.append(row)
                labels.append(truth_row)
                if sign == 0:
                    baseline_end = branch
            mujoco.mj_copyData(env.data_, env.model_, baseline_end.data_)
            pair_count += 1
        else:
            env.step(action)
            # Keep baseline history even when a snapshot is rejected for probing.
            after, after_truth = observe_sample(env, sensor, param)
            row, truth_row = transition(before, before_truth, after, after_truth, action, action, status, step)
            rows.append(row)
            labels.append(truth_row)
        quat_error = metrics.comp_quat_error(env.get_state()[3:7], param.target_q_)
        consecutive_success = consecutive_success + 1 if quat_error < 0.04 else 0
        if consecutive_success >= 21 and success_step < 0:
            success_step = step + 1
        if args.protocol == "paired" and pair_count >= args.probe_count:
            stop_reason = "probe_count_reached"
            break
        if not np.all(np.isfinite(integration_state(env))):
            stop_reason = "nonfinite_baseline_state"
            break
    inputs_path = episode_dir / "inputs.npz"
    labels_path = episode_dir / "labels.npz"
    np.savez_compressed(inputs_path, **pack_rows(rows, {"contact_records", "next_contact_records"}),
                        target_position=np.asarray(param.target_p_), target_quaternion=np.asarray(param.target_q_))
    np.savez_compressed(labels_path, **pack_rows(labels, {"contact_truth_records", "next_contact_truth_records"}))
    metadata = {
        "schema_version": SCHEMA_VERSION, "object": obj, "mode": mode, "trial": trial,
        "protocol": args.protocol, "rows": len(rows), "paired_snapshots": pair_count,
        "stop_reason": stop_reason, "initial_state": initial_state.tolist(),
        "initial_state_order": "object_position(3), object_quaternion_wxyz(4), hand_joint_positions(16)",
        "mujoco_version": mujoco.__version__, "model_path": str(param.model_path_),
        "physics_timestep": float(env.model_.opt.timestep), "frame_skip": int(param.frame_skip_),
        "control_interval": float(env.model_.opt.timestep * param.frame_skip_),
        "action": "16 joint target increments in rad; target = current hand qpos + action",
        "state_observation": "ideal noiseless visual object pose and hand joint encoders; pose taken from simulation",
        "tactile_schema": sensor.schema_metadata(), "contact_record_columns": list(CONTACT_RECORD_COLUMNS),
        "diagnostic_input_fields": ["solver_status", "source_step", "pair_id", "probe_joint", "probe_sign", "probe_epsilon", "base_action", "action_perturbation"],
        "labels": {"access": "evaluation-only; not controller input",
                   "integration_state": "mj_getState(mjSTATE_INTEGRATION), MuJoCo-defined state layout, includes controls and warmstart",
                   "integration_state_spec": int(INTEGRATION_SPEC),
                   "integration_state_size": int(mujoco.mj_stateSize(env.model_, INTEGRATION_SPEC)),
                   "integration_state_fields": integration_layout(env.model_),
                   "contact_truth_columns": list(TRUTH_CONTACT_COLUMNS),
                   "n_ground_contacts": "loaded object contacts with a plane geom; separate from palm/other counts",
                   "rotation_delta_rad": "SO(3) geodesic angle between before and after object orientations"},
        "physical_parameters": physical_metadata(env, param),
        "success": {"sustained_success_step": success_step, "required_consecutive_steps": 21,
                    "criterion": "FREE Allegro quaternion error 1-dot(q,target)^2 < 0.04; orientation only",
                    "scope": "baseline branch diagnostic only; short probe collection is not success-rate evaluation"},
        "probe_settings": {"joints_requested": args.probe_joints, "epsilons_rad": args.epsilons,
                           "min_active_fingers": args.min_fingers, "snapshot_copy": "complete mj_copyData, including controls/time/warmstart",
                           "bound_policy": "both signs at all radii must fit action, joint-target and actuator control limits; no probe clipping",
                           "qualification": "finite snapshot, unit object quaternion, min active fingertips, no plane contact, joint and target limits"},
        "solver_attempts": solver_attempts, "rejected_snapshots": rejections,
        "limitations": ["FREE is an action source, not a guaranteed-safe controller.",
                        "Warmup FREE actions are not probe-qualified; their transitions are retained with pair_id=-1, not paired probe rows.",
                        "No offline model training or learned controller is performed.",
                        "One-step replay from identical physical state diagnoses local response; it is not an online method.",
                        "One fixed mismatch is not broad unknown-dynamics coverage.",
                        "Inputs and labels are separate files, but applications must explicitly restrict allowed features."],
    }
    (episode_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    summary = {"object": obj, "mode": mode, "trial": trial, "rows": len(rows), "paired_snapshots": pair_count,
               "stop_reason": stop_reason, "path": str(episode_dir)}
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    args = parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.output = Path("docs/data") / f"tactile_probe_v2_{stamp}"
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for obj in args.objects:
        for mode in args.modes:
            for trial in range(args.trials):
                summaries.append(collect_episode(obj, trial, mode, args, args.output))
                (args.output / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "episodes": summaries}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"episodes": len(summaries), "rows": sum(s["rows"] for s in summaries), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Read-only task observer: does the object actually move, and what carries it?

Runs FREE's own MPC loop unchanged (same hand config and success thresholds as
``examples/mpc/eval_unknown_dyn.py``) and logs, at every control step, only
quantities FREE already computes:

  * object pose from ``env.get_state()``
  * contact composition against the object, classified by the *other* geom and
    body name (table / fingertip / palm / other)
  * per-contact normal force from ``mj_contactForce`` after a fresh
    ``mj_forward``

FREE's ``envs/``, ``models/``, ``planning/``, ``contact/`` and ``utils/`` are
never modified. This is an observer, not a controller, and it is not part of the
FREE baseline: it only records what the baseline already does.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np

from examples.mpc.eval_unknown_dyn import HAND, _is_success
from planning.mpc_explicit import MPCExplicit
from utils import metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", required=True, choices=sorted(HAND))
    p.add_argument("--objects", nargs="+", default=["cube"])
    p.add_argument("--target-type", default=None)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--all-targets", action="store_true",
                   help="Fingertips only: sweep all three target types")
    p.add_argument("--output", type=Path, default=None,
                   help="Defaults to docs/data/motion_probe_<hand>_<stamp>")
    return p.parse_args()


def _name(model, obj_type, idx):
    if idx < 0:
        return None
    return mujoco.mj_id2name(model, obj_type, idx)


def classify(geom_name, body_name):
    """Bucket the non-object geom in a contact by what it belongs to."""
    for name in (geom_name or "", body_name or ""):
        low = name.lower()
        if "fingertip" in low or low.startswith("ftp_"):
            return "fingertip"
        if low == "table" or "groundplane" in low:
            return "table"
        if low.startswith("palm"):
            return "palm"
    return "other"


def observe_contacts(model, data, object_geoms):
    """Return per-contact rows for object contacts. Recomputes derived fields.

    ``mj_forward`` is idempotent for a fixed state and is already what
    ``Contact.detect_once`` calls in the untouched baseline loop, so the extra
    call does not perturb the rollout. It is required because
    ``detect_once`` ends in ``mj_collision``, which invalidates ``efc_address``
    and makes every ``mj_contactForce`` read return 0.
    """
    mujoco.mj_forward(model, data)
    rows = []
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        in1, in2 = g1 in object_geoms, g2 in object_geoms
        if in1 == in2:
            continue
        other = g2 if in1 else g1
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other]))
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, wrench)
        rows.append({
            "geom": geom_name,
            "body": body_name,
            "class": classify(geom_name, body_name),
            "normal_force": float(wrench[0]),
            "dist": float(contact.dist),
            "pos": np.asarray(contact.pos, dtype=np.float64).tolist(),
        })
    return rows


def quat_angle(q1, q2):
    """Smallest rotation angle (rad) between two unit quaternions."""
    dot = float(np.clip(abs(np.dot(np.asarray(q1), np.asarray(q2))), 0.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def run_episode(hand, obj, target_type, trial):
    cfg = HAND[hand]
    params_cls = importlib.import_module(f"examples.mpc.{hand}.{obj}.params").ExplicitMPCParams
    env_cls = importlib.import_module(cfg["env"]).MjSimulator
    contact_cls = importlib.import_module(cfg["contact"]).Contact

    param = params_cls(rand_seed=trial, target_type=target_type)
    param.headless_ = True
    init_quat = np.asarray(param.init_obj_qpos_[3:7], dtype=np.float64)
    contact = contact_cls(param)
    env = env_cls(param)
    mpc = MPCExplicit(param)
    object_geoms = {
        mujoco.mj_name2id(env.model_, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in param.object_names_
    }

    trace = []
    rollout_step = 0
    consecutive_success_time = 0
    while rollout_step < cfg["max_steps"]:
        if env.dyn_paused_:
            continue
        curr_q = env.get_state()
        rows = observe_contacts(env.model_, env.data_, object_geoms)
        if hasattr(env, "get_finger_jpos"):
            robot_qpos = np.asarray(env.get_finger_jpos(), dtype=np.float64)
        else:
            robot_qpos = np.asarray(curr_q[7:], dtype=np.float64)
        trace.append({
            "step": rollout_step,
            "qpos": np.asarray(curr_q, dtype=np.float64).tolist(),
            "robot_qpos": robot_qpos.tolist(),
            "contacts": rows,
        })

        phi_vec, jac_mat = contact.detect_once(env)
        sol = mpc.plan_once(
            param.target_p_, param.target_q_, curr_q, phi_vec, jac_mat,
            sol_guess=param.sol_guess_)
        param.sol_guess_ = sol["sol_guess"]
        env.step(sol["action"])
        rollout_step += 1
        curr_q = env.get_state()
        if _is_success(cfg, curr_q, param):
            consecutive_success_time += 1
        else:
            consecutive_success_time = 0
        if consecutive_success_time > 20:
            break

    success = rollout_step < cfg["max_steps"]
    if getattr(env, "viewer_", None) is not None:
        env.viewer_.close()

    qpos = np.asarray([t["qpos"] for t in trace], dtype=np.float64)
    quats = qpos[:, 3:7]
    final_q = env.get_state()
    return {
        "hand": hand,
        "object": obj,
        "target_type": target_type,
        "trial": trial,
        "steps": rollout_step,
        "success": bool(success),
        "quat_err": float(metrics.comp_quat_error(final_q[3:7], param.target_q_)),
        "pos_err": float(metrics.comp_pos_error(final_q[0:3], param.target_p_)),
        "rot_travel_deg": float(np.degrees(sum(
            quat_angle(quats[i], quats[i + 1]) for i in range(len(quats) - 1)))),
        "rot_net_deg": float(np.degrees(quat_angle(quats[0], quats[-1]))),
        "rot_from_init_deg": float(np.degrees(quat_angle(init_quat, quats[-1]))),
        "z_init": float(qpos[0, 2]),
        "z_min": float(qpos[:, 2].min()),
        "z_max": float(qpos[:, 2].max()),
        "z_final": float(qpos[-1, 2]),
        "xy_travel": float(np.sum(np.linalg.norm(
            np.diff(qpos[:, 0:2], axis=0), axis=1))),
        "trace": trace,
    }


def compact_trace(ep):
    """Per-step numeric record (T, 15): step, qpos(7), class counts(4), loads(3)."""
    rows = []
    for frame in ep["trace"]:
        contacts = frame["contacts"]
        counts = {"table": 0, "fingertip": 0, "palm": 0, "other": 0}
        for c in contacts:
            counts[c["class"]] += 1
        rows.append([
            frame["step"], *frame["qpos"][:7],
            counts["table"], counts["fingertip"], counts["palm"], counts["other"],
            sum(c["normal_force"] for c in contacts if c["class"] != "table"),
            sum(c["normal_force"] for c in contacts if c["class"] == "table"),
            max((c["normal_force"] for c in contacts), default=0.0),
        ])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 15)


def robot_trace(ep):
    """Per-step robot joint positions (T, k); k differs per hand."""
    return np.asarray([f["robot_qpos"] for f in ep["trace"]], dtype=np.float64)


def summarize(episodes, object_half_extent):
    """Aggregate per-episode traces into the numbers that answer 'did it move'."""
    out = []
    for ep in episodes:
        counts = {"table": 0, "fingertip": 0, "palm": 0, "other": 0}
        steps_with_fingertip = 0
        steps_with_table = 0
        steps_with_any_object_contact = 0
        non_table_load = []
        table_load = []
        for frame in ep["trace"]:
            rows = frame["contacts"]
            if rows:
                steps_with_any_object_contact += 1
            classes = {r["class"] for r in rows}
            for r in rows:
                counts[r["class"]] += 1
            if "fingertip" in classes:
                steps_with_fingertip += 1
            if "table" in classes:
                steps_with_table += 1
            non_table_load.append(sum(
                r["normal_force"] for r in rows if r["class"] != "table"))
            table_load.append(sum(
                r["normal_force"] for r in rows if r["class"] == "table"))
        n = max(len(ep["trace"]), 1)
        clearance = ep["z_max"] - object_half_extent
        out.append({
            "hand": ep["hand"], "object": ep["object"],
            "target_type": ep["target_type"], "trial": ep["trial"],
            "success": ep["success"], "steps": ep["steps"],
            "rot_travel_deg": round(ep["rot_travel_deg"], 3),
            "rot_net_deg": round(ep["rot_net_deg"], 3),
            "quat_err": round(ep["quat_err"], 5),
            "pos_err": round(ep["pos_err"], 5),
            "z_init": round(ep["z_init"], 5),
            "z_max": round(ep["z_max"], 5),
            "z_final": round(ep["z_final"], 5),
            "max_clearance_above_table": round(clearance, 5),
            "xy_travel": round(ep["xy_travel"], 5),
            "frac_steps_with_fingertip_contact": round(steps_with_fingertip / n, 4),
            "frac_steps_with_table_contact": round(steps_with_table / n, 4),
            "frac_steps_with_any_object_contact": round(
                steps_with_any_object_contact / n, 4),
            "frac_steps_airborne_no_table": round(
                (n - steps_with_table) / n, 4),
            "mean_non_table_normal_force": round(float(np.mean(non_table_load)), 5),
            "mean_table_normal_force": round(float(np.mean(table_load)), 5),
            "contact_class_counts": counts,
        })
    return out


def main():
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output or Path("docs/data") / f"motion_probe_{args.hand}_{stamp}"
    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output dir: {out_dir}")
    out_dir.mkdir(parents=True)

    episodes = []
    for obj in args.objects:
        base = args.target_type or HAND[args.hand]["target_type"]
        if args.hand == "fingertips" and args.all_targets:
            target_types = ["in-air", "ground-rotation", "ground-flip"]
        else:
            target_types = [base]
        for tt in target_types:
            for trial in range(args.trials):
                ep = run_episode(args.hand, obj, tt, trial)
                print(f"  {args.hand}/{obj}/{tt} trial {trial:02d}: "
                      f"{'SUCCESS' if ep['success'] else 'FAIL':7s} "
                      f"steps={ep['steps']:4d} rot_net={ep['rot_net_deg']:7.2f}deg "
                      f"rot_travel={ep['rot_travel_deg']:8.2f}deg "
                      f"z={ep['z_init']:.3f}->{ep['z_max']:.3f}->{ep['z_final']:.3f} "
                      f"quat_err={ep['quat_err']:.4f}", flush=True)
                episodes.append(ep)

    half_extent = 0.03
    for obj in args.objects:
        line = next((ln for ln in
                     Path(f"envs/xmls/env_{args.hand}_{obj}.xml").read_text().splitlines()
                     if 'geom name="obj"' in ln and "size=" in ln), None)
        if line:
            half_extent = float(line.split("size=")[1].split('"')[1].split()[0])
            break

    summary = summarize(episodes, half_extent)
    (out_dir / "summary.json").write_text(json.dumps({
        "schema": "motion_probe_v1",
        "hand": args.hand,
        "objects": args.objects,
        "trials": args.trials,
        "object_half_extent": half_extent,
        "note": "Observer only. FREE files untouched; no controller changes.",
        "episodes": summary,
    }, indent=2))

    np.savez_compressed(out_dir / "traces.npz", **{k: v for i, ep in enumerate(episodes)
        for k, v in ((f"ep{i:03d}", compact_trace(ep)),
                     (f"robot{i:03d}", robot_trace(ep)))})
    (out_dir / "traces_index.json").write_text(json.dumps([
        {"key": f"ep{i:03d}", "hand": ep["hand"], "object": ep["object"],
         "target_type": ep["target_type"], "trial": ep["trial"],
         "success": ep["success"], "steps": ep["steps"]}
        for i, ep in enumerate(episodes)
    ], indent=2))
    print(f"\nwrote {out_dir}/summary.json ({len(summary)} episodes)")


if __name__ == "__main__":
    main()

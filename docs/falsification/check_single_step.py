"""Single-physics-step test of the contact-consistency criterion.

Rationale: over a whole control interval (baseline action ~0.2 rad) the first-order
contact relation does not hold (measured normalised residual 0.42-1.05). The honest
time scale for a linearised contact test is one physics step. This script restores
each recorded baseline state, applies the recorded base action for exactly ONE
mj_step, and scores hypotheses with the two-sided criterion.

Labels are PRIVILEGED: used only to select the true contact set for scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def ragged(data, field, row):
    off = data[field + "_offsets"]
    return data[field][off[row]:off[row + 1]]


def jac3d(model, data, point, body):
    j = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp=j, jacr=None, point=point, body=body)
    return j


def contact_rows(model, data, object_geoms, mu):
    """(geom1, geom2, rows(4, nv), dist) for contacts involving the object."""
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in object_geoms and g2 not in object_geoms:
            continue
        frame = c.frame.reshape((-1, 3)).T
        fpmd = np.hstack((frame, -frame[:, -2:]))
        k1 = fpmd.T @ jac3d(model, data, c.pos, int(model.geom_bodyid[g1]))
        k2 = fpmd.T @ jac3d(model, data, c.pos, int(model.geom_bodyid[g2]))
        cj = -(k2 - k1) if g1 in object_geoms else (k2 - k1)
        out.append((g1, g2, cj[0][None, :] + mu * cj[1:], float(c.dist)))
    return out


def score(rows_list, dq_hand, x_obs):
    if not rows_list:
        return dict(nrows=0, rank=0, null_dim=6, consistency=float("inf"), consistency_rel=float("inf"), adequacy=float("inf"))
    J = np.vstack([r[:, :16] for r in rows_list])
    A = np.vstack([r[:, 16:22] for r in rows_list])
    rank = int(np.linalg.matrix_rank(A))
    cons = float(np.linalg.norm(A @ x_obs + J @ dq_hand))
    jn = float(np.linalg.norm(J @ dq_hand))
    pred = -np.linalg.pinv(A) @ (J @ dq_hand)
    if rank >= 1:
        U, _, _ = np.linalg.svd(A.T @ A)
        P = U[:, :rank] @ U[:, :rank].T
        diff = P @ (pred - x_obs)
        adeq = float(np.linalg.norm(diff) / max(float(np.linalg.norm(P @ x_obs)), 1e-12))
    else:
        adeq = float("nan")
    return dict(nrows=int(A.shape[0]), rank=rank, null_dim=6 - rank,
                consistency=cons, consistency_rel=cons / max(jn, 1e-12), adequacy=adeq)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("docs/data/tactile_probe_v2_review_20260920"))
    ap.add_argument("--episode", default="allegro_cube_nominal_trial000")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--output", type=Path, default=Path("docs/falsification/out/single_step.json"))
    args = ap.parse_args()

    folder = args.dataset / args.episode
    meta = json.loads((folder / "metadata.json").read_text())
    model = mujoco.MjModel.from_xml_path(meta["model_path"])
    data = mujoco.MjData(model)
    nv, dt = model.nv, model.opt.timestep
    object_geoms = {model.geom(n).id for n in meta["tactile_schema"]["object_geoms"]}
    inp = np.load(folder / "inputs.npz", allow_pickle=False)
    lab = np.load(folder / "labels.npz", allow_pickle=False)
    base = np.flatnonzero(inp["probe_sign"] == 0)
    base = base[np.argsort(inp["source_step"][base])]

    print(f"timestep={dt:g} s   nv={nv}   frames={min(args.frames, len(base))}")
    records = []
    for idx in base[: args.frames]:
        state = inp["state"][idx]
        action = inp["action"][idx]
        qpos0 = np.concatenate((state[7:23], state[0:7]))
        data.qpos[:] = qpos0
        data.qvel[:] = 0.0
        data.qvel[:16] = inp["joint_velocity"][idx]
        mujoco.mj_forward(model, data)

        rows = contact_rows(model, data, object_geoms, mu=0.5)
        truth = ragged(lab, "contact_truth_records", idx)
        loaded = truth[truth[:, 3] > 1e-8]
        pairs = {(int(c[0]), int(c[1])) for c in loaded} | {(int(c[1]), int(c[0])) for c in loaded}
        true_rows = [r for g1, g2, r, _ in rows if (g1, g2) in pairs]
        other_rows = [r for g1, g2, r, _ in rows if (g1, g2) not in pairs]

        ctrl = data.qpos[:16] + action
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        qpos1 = data.qpos.copy()
        dv = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dv, dt, qpos0, qpos1)
        dq_hand, x_obs = dv[:16], dv[16:22]

        entry = {"source_step": int(inp["source_step"][idx]),
                 "n_true": len(true_rows), "n_other": len(other_rows),
                 "dq_norm": float(np.linalg.norm(dq_hand)),
                 "x_norm": float(np.linalg.norm(x_obs)),
                 "H_true": score(true_rows, dq_hand, x_obs),
                 "H_add": score(true_rows + other_rows, dq_hand, x_obs) if other_rows else None,
                 "drop_one": [dict(k=k, **score([r for j, r in enumerate(true_rows) if j != k], dq_hand, x_obs))
                              for k in range(len(true_rows))],
                 "normal_only": score([r[0:1] for r in true_rows], dq_hand, x_obs)}
        records.append(entry)
        ht = entry["H_true"]
        print(f"step {entry['source_step']:>3} n_true={len(true_rows)} n_other={len(other_rows)} "
              f"|dq|={entry['dq_norm']:.2e} |x|={entry['x_norm']:.2e} | "
              f"H_true rank={ht['rank']} null={ht['null_dim']} cons={ht['consistency']:.3e} "
              f"cons_rel={ht['consistency_rel']:.3f} adeq={ht['adequacy']:.3f}")
        d1 = [d for d in entry["drop_one"] if d["rank"] == ht["rank"] and d["nrows"] > 0]
        if d1:
            print(f"        drop-one: min cons={min(d['consistency'] for d in d1):.3e} "
                  f"min adeq={min(d['adequacy'] for d in d1):.3f}")
        if entry["H_add"]:
            print(f"        add-other: cons={entry['H_add']['consistency']:.3e} adeq={entry['H_add']['adequacy']:.3f}")
        print(f"        normal-only: rank={entry['normal_only']['rank']} "
              f"cons={entry['normal_only']['consistency']:.3e} adeq={entry['normal_only']['adequacy']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

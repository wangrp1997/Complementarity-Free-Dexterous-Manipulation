"""Discrimination test for contact hypotheses on recorded frames.

Motivation: the naive residual ||A_h x + J_h dq|| improves monotonically as rows
are removed, so a hypothesis that simply DROPS contacts always wins. The
well-posed criterion is two-sided:

  consistency  r_h = ||A_h x_obs + J_h dq|| / ||J_h dq||
  adequacy     e_h = ||x_pred - x_obs|| / ||x_obs||,  x_pred = -pinv(A_h) J_h dq

with rank(A_h) = 6 required. This script tests whether that criterion separates
the true contact set from leave-one-out and spurious-contact variants.

Read-only diagnostic; labels are PRIVILEGED (used only to select the true set).
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


def contact_jacobians(model, data, object_geoms, nv, mu=0.5):
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in object_geoms and g2 not in object_geoms:
            continue
        frame = c.frame.reshape((-1, 3)).T
        frame_pmd = np.hstack((frame, -frame[:, -2:]))
        j1 = np.zeros((3, nv)); j2 = np.zeros((3, nv))
        mujoco.mj_jac(model, data, jacp=j1, jacr=None, point=c.pos, body=int(model.geom_bodyid[g1]))
        mujoco.mj_jac(model, data, jacp=j2, jacr=None, point=c.pos, body=int(model.geom_bodyid[g2]))
        k1 = frame_pmd.T @ j1
        k2 = frame_pmd.T @ j2
        con_jac = -(k2 - k1) if g1 in object_geoms else (k2 - k1)
        rows = con_jac[0][None, :] + mu * con_jac[1:]
        out.append((g1, g2, rows, float(c.dist)))
    return out


def score(rows_list, dq_hand, x_obs):
    if not rows_list:
        return dict(nrows=0, rank=0, consistency=float("inf"), adequacy=float("inf"), mu=None)
    J = np.vstack([r[:, :16] for r in rows_list])
    A = np.vstack([r[:, 16:22] for r in rows_list])
    denom_j = max(float(np.linalg.norm(J @ dq_hand)), 1e-12)
    denom_x = max(float(np.linalg.norm(x_obs)), 1e-12)
    r = float(np.linalg.norm(A @ x_obs + J @ dq_hand)) / denom_j
    pred = -np.linalg.pinv(A) @ (J @ dq_hand)
    e = float(np.linalg.norm(pred - x_obs)) / denom_x
    return dict(nrows=int(A.shape[0]), rank=int(np.linalg.matrix_rank(A)),
                consistency=r, adequacy=e, mu=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("docs/data/tactile_probe_v2_review_20260920"))
    ap.add_argument("--episode", default="allegro_cube_nominal_trial000")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--output", type=Path, default=Path("docs/falsification/out/discrimination.json"))
    args = ap.parse_args()

    folder = args.dataset / args.episode
    meta = json.loads((folder / "metadata.json").read_text())
    model = mujoco.MjModel.from_xml_path(meta["model_path"])
    data = mujoco.MjData(model)
    nv = model.nv
    object_geoms = {model.geom(n).id for n in meta["tactile_schema"]["object_geoms"]}
    inp = np.load(folder / "inputs.npz", allow_pickle=False)
    lab = np.load(folder / "labels.npz", allow_pickle=False)
    base = np.flatnonzero(inp["probe_sign"] == 0)
    base = base[np.argsort(inp["source_step"][base])]

    records = []
    for idx in base[: args.frames]:
        state = inp["state"][idx]
        data.qpos[:16] = state[7:23]
        data.qpos[16:23] = state[0:7]
        data.qvel[:] = 0.0
        data.qvel[:16] = inp["joint_velocity"][idx]
        mujoco.mj_forward(model, data)
        jacs = contact_jacobians(model, data, object_geoms, nv)

        qn = np.concatenate((inp["next_state"][idx][7:23], inp["next_state"][idx][0:7]))
        qp = np.concatenate((state[7:23], state[0:7]))
        dv = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dv, 1.0, qp, qn)
        dq_hand, x_obs = dv[:16], dv[16:22]

        truth = ragged(lab, "contact_truth_records", idx)
        loaded = truth[truth[:, 3] > 1e-8]
        pairs = {(int(c[0]), int(c[1])) for c in loaded} | {(int(c[1]), int(c[0])) for c in loaded}
        true_rows = [r for g1, g2, r, _ in jacs if (g1, g2) in pairs]
        other_rows = [r for g1, g2, r, _ in jacs if (g1, g2) not in pairs]

        entry = {
            "source_step": int(inp["source_step"][idx]),
            "n_true_contacts": len(true_rows),
            "n_other_detected": len(other_rows),
            "H_true": score(true_rows, dq_hand, x_obs),
            "H_true_plus_other": score(true_rows + other_rows, dq_hand, x_obs) if other_rows else None,
            "drop_one_contact": [dict(dropped=k, **score([r for j, r in enumerate(true_rows) if j != k],
                                                          dq_hand, x_obs))
                                 for k in range(len(true_rows))],
            "free_one_contact_tangents": [dict(freed=k, **score([r[0:1] if j == k else r
                                                                 for j, r in enumerate(true_rows)],
                                                                  dq_hand, x_obs))
                                          for k in range(len(true_rows))],
        }
        records.append(entry)
        ht = entry["H_true"]
        print(f"step {entry['source_step']:>3} n_true={entry['n_true_contacts']} n_other={entry['n_other_detected']} "
              f"H_true rank={ht['rank']:>2} cons={ht['consistency']:.4f} adeq={ht['adequacy']:.4f}")
        d1 = [d for d in entry["drop_one_contact"] if d["rank"] >= 6]
        if d1:
            print(f"        drop-one (rank>=6): min cons={min(d['consistency'] for d in d1):.4f} "
                  f"min adeq={min(d['adequacy'] for d in d1):.4f}  n={len(d1)}")
        else:
            print("        drop-one: no rank>=6 variant (true set is minimal)")
        f1 = [d for d in entry["free_one_contact_tangents"] if d["rank"] >= 6]
        if f1:
            print(f"        free-tangents  : min cons={min(d['consistency'] for d in f1):.4f} "
                  f"min adeq={min(d['adequacy'] for d in f1):.4f}  n={len(f1)}")
        if entry["H_true_plus_other"]:
            print(f"        add-other      : cons={entry['H_true_plus_other']['consistency']:.4f} "
                  f"adeq={entry['H_true_plus_other']['adequacy']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

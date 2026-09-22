"""Independent check of the core relation A_h x = -J_h dq on real recorded frames.

Read-only diagnostic. Does NOT modify FREE files. Uses labels ONLY to select the
true contact set for this sanity check (marked PRIVILEGED).

Purpose: if the linearised contact relation does not hold on recorded data, the
whole L1-L4 ladder is void, so this is checked before anything else.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def ragged(data, field, row):
    offsets = data[field + "_offsets"]
    return data[field][offsets[row]:offsets[row + 1]]


def build_model(model_path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)


def set_state_from_record(model, data, state, joint_velocity):
    """state = [obj_pos(3), obj_quat_wxyz(4), hand(16)] -> qpos = [hand(16), obj(7)]."""
    data.qpos[:16] = state[7:23]
    data.qpos[16:23] = state[0:7]
    data.qvel[:] = 0.0
    data.qvel[:16] = joint_velocity
    mujoco.mj_forward(model, data)


def contact_jacobians(model, data, object_geoms, nv):
    """Reimplement FREE's row structure (4 rows per contact) without importing FREE.

    Returns list of (geom1, geom2, jac_rows(4, nv)) for contacts involving the object.
    """
    mujoco.mj_forward(model, data)
    mujoco.mj_collision(model, data)
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in object_geoms and g2 not in object_geoms:
            continue
        con_frame = c.frame.reshape((-1, 3)).T
        con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))
        jacp1 = np.zeros((3, nv))
        jacp2 = np.zeros((3, nv))
        mujoco.mj_jac(model, data, jacp=jacp1, jacr=None, point=c.pos, body=int(model.geom_bodyid[g1]))
        mujoco.mj_jac(model, data, jacp=jacp2, jacr=None, point=c.pos, body=int(model.geom_bodyid[g2]))
        j1 = con_frame_pmd.T @ jacp1
        j2 = con_frame_pmd.T @ jacp2
        con_jac = -(j2 - j1) if g1 in object_geoms else (j2 - j1)
        n_row = con_jac[0]
        f_rows = con_jac[1:]
        mu = 0.5
        rows = n_row[None, :] + mu * f_rows
        out.append((g1, g2, rows))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("docs/data/tactile_probe_v2_review_20260920"))
    ap.add_argument("--episode", default="allegro_cube_nominal_trial000")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--output", type=Path, default=Path("docs/falsification/out/core_relation_check.json"))
    args = ap.parse_args()

    folder = args.dataset / args.episode
    meta = json.loads((folder / "metadata.json").read_text())
    model = build_model(meta["model_path"])
    data = mujoco.MjData(model)
    nv = model.nv
    object_geoms = {model.geom(n).id for n in meta["tactile_schema"]["object_geoms"]}

    inp = np.load(folder / "inputs.npz", allow_pickle=False)
    lab = np.load(folder / "labels.npz", allow_pickle=False)
    baseline = np.flatnonzero(inp["probe_sign"] == 0)
    order = np.argsort(inp["source_step"][baseline])
    baseline = baseline[order]

    rows_out = []
    for idx in baseline[: args.frames]:
        state = inp["state"][idx]
        jv = inp["joint_velocity"][idx]
        set_state_from_record(model, data, state, jv)
        jacs = contact_jacobians(model, data, object_geoms, nv)

        # Actual per-step displacement in MuJoCo velocity coordinates (hand & object).
        qpos_next = np.concatenate((inp["next_state"][idx][7:23], inp["next_state"][idx][0:7]))
        qpos_now = np.concatenate((state[7:23], state[0:7]))
        dqvel = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dqvel, 1.0, qpos_now, qpos_next)

        # PRIVILEGED: truth contact set, used only to select which rows to test here.
        truth = ragged(lab, "contact_truth_records", idx)
        loaded = truth[truth[:, 3] > 1e-8]
        true_pairs = {(int(c[0]), int(c[1])) for c in loaded}
        true_pairs |= {(int(c[1]), int(c[0])) for c in loaded}

        n_true = 0
        res_true = []
        res_all = []
        for g1, g2, rows in jacs:
            r = rows @ dqvel
            res_all.append(float(np.linalg.norm(r)))
            if (g1, g2) in true_pairs:
                n_true += 1
                res_true.append(float(np.linalg.norm(r)))

        rows_out.append(dict(
            source_step=int(inp["source_step"][idx]),
            n_contacts_in_free_rows=len(jacs),
            n_true_loaded_contacts=int(len(loaded)),
            n_matched=n_true,
            dqvel_norm_hand=float(np.linalg.norm(dqvel[:16])),
            dqvel_norm_object=float(np.linalg.norm(dqvel[16:])),
            residual_true_contacts=res_true,
            residual_all_contacts=res_all,
            median_residual_true=float(np.median(res_true)) if res_true else None,
            max_residual_true=float(np.max(res_true)) if res_true else None,
            rows_per_contact=int(jacs[0][2].shape[0]) if jacs else None,
            nv=int(nv),
        ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(episode=args.episode, frames=rows_out), indent=2) + "\n")
    for r in rows_out:
        print(f"step {r['source_step']:>3}  free_rows={r['n_contacts_in_free_rows']:>2} "
              f"true={r['n_true_loaded_contacts']:>2} matched={r['n_matched']:>2} "
              f"|dq|={r['dqvel_norm_hand']:.3e} |x|={r['dqvel_norm_object']:.3e} "
              f"median_res={r['median_residual_true']} max_res={r['max_residual_true']}")


if __name__ == "__main__":
    main()

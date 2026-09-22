"""区分三种约束行结构，判断控制区间尺度上"黏着假设"是否成立。

背景：check_discrimination.py 用的是 FREE 的软摩擦锥行 (n + mu*t)，
那是"滑移/摩擦饱和"的近似，不是"黏着"。拿它评价黏着假设，
残差大是模型的预期后果，不能直接当成"运动学关系失效"。

本脚本对同一批帧同时报告三种口径下的相对速度违反量：
  stick3  : 接触系三行 (n, t1, t2) —— 黏着的正确判据
  normal  : 仅法向一行           —— 只要求不穿透
  pyra    : FREE 的 n + mu*t     —— 对照，复现 check_discrimination 口径

Read-only。labels 仅用于选出真值接触集（PRIVILEGED）。
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


def true_contact_jacobians(model, data, object_geoms, nv, dqvel, pairs, mu):
    """逐接触用各自的 c.pos 重算接触系相对速度 Jacobian，返回各口径违反量。"""
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in object_geoms and g2 not in object_geoms:
            continue
        if (g1, g2) not in pairs:
            continue
        frame = c.frame.reshape((-1, 3)).T
        frame_pmd = np.hstack((frame, -frame[:, -2:]))
        j1 = np.zeros((3, nv))
        j2 = np.zeros((3, nv))
        mujoco.mj_jac(model, data, jacp=j1, jacr=None, point=c.pos,
                      body=int(model.geom_bodyid[g1]))
        mujoco.mj_jac(model, data, jacp=j2, jacr=None, point=c.pos,
                      body=int(model.geom_bodyid[g2]))
        k1 = frame_pmd.T @ j1
        k2 = frame_pmd.T @ j2
        con_jac = -(k2 - k1) if g1 in object_geoms else (k2 - k1)
        v = con_jac @ dqvel
        rows.append(dict(
            g1=g1, g2=g2,
            stick3=float(np.linalg.norm(v)),
            normal=float(abs(v[0])),
            pyra=float(np.linalg.norm(v[0] + mu * v[1:])),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path("docs/data/tactile_probe_v2_review_20260920"))
    ap.add_argument("--episode", default="allegro_cube_nominal_trial000")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--output", type=Path,
                    default=Path("docs/falsification/out/row_structure.json"))
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
        mujoco.mj_collision(model, data)

        qp = np.concatenate((state[7:23], state[0:7]))
        qn = np.concatenate((inp["next_state"][idx][7:23],
                             inp["next_state"][idx][0:7]))
        dv = np.zeros(nv)
        mujoco.mj_differentiatePos(model, dv, 1.0, qp, qn)

        truth = ragged(lab, "contact_truth_records", idx)
        loaded = truth[truth[:, 3] > 1e-8]
        pairs = {(int(c[0]), int(c[1])) for c in loaded}
        pairs |= {(int(c[1]), int(c[0])) for c in loaded}

        rows = true_contact_jacobians(model, data, object_geoms, nv, dv, pairs, args.mu)
        hand_speed = float(np.linalg.norm(dv[:16]))
        rec = dict(
            source_step=int(inp["source_step"][idx]),
            n_true=len(rows),
            hand_speed=hand_speed,
            obj_speed=float(np.linalg.norm(dv[16:])),
            max_stick3=max((r["stick3"] for r in rows), default=None),
            max_normal=max((r["normal"] for r in rows), default=None),
            max_pyra=max((r["pyra"] for r in rows), default=None),
            stick3_over_hand=[round(r["stick3"] / max(hand_speed, 1e-12), 3)
                              for r in rows],
        )
        records.append(rec)
        print(f"step {rec['source_step']:>3} n={rec['n_true']:>2} "
              f"|dq|={hand_speed:.3f} stick3={rec['max_stick3']:.4f} "
              f"normal={rec['max_normal']:.4f} pyra={rec['max_pyra']:.4f} "
              f"ratio={rec['stick3_over_hand']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

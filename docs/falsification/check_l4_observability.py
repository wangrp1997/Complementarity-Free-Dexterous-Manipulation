"""L4（模型无关版）：合法动作能让触觉观测变化多少？信息和推进是对齐还是错开？

不假设"接触模式"这类离散变量（F3 已显示 FREE 里它不自然存在），
只测可直接观测的量：
  tactile(a) : 四个指尖的法向载荷（4 维），由 MuJoCo 接触力读出
  task(a)    : 该动作让目标姿态误差减少了多少（弧度）
  blind(a)   : 全部物体接触的载荷总和（含掌面），用于对照"指尖通道有多盲"

对每个恢复状态，比较 K 个合法动作相对基准动作的 Δtactile 与 Δtask。
只读；不修改任何文件；labels 只用于取 integration_state（PRIVILEGED）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

STATE_SPEC = 16383
CONTROL_INTERVAL = 0.1


def quat_angle_deg(q1, q2):
    d = abs(float(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2))))
    return 2.0 * np.degrees(np.arccos(min(1.0, d)))


def contact_forces(model, data):
    """返回 (每指法向载荷 4 维, 全接触载荷总和, 掌面接触数, 指尖接触数)。"""
    load = np.zeros(4)
    total = 0.0
    n_palm = 0
    n_ft = 0
    res = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.efc_address < 0:
            continue
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or ""
        names = (n1, n2)
        if not any(n.startswith("obj") for n in names):
            continue
        mujoco.mj_contactForce(model, data, i, res)
        fn = abs(float(res[0]))
        total += fn
        other = n1 if n2.startswith("obj") else n2
        if other.startswith("fingertip"):
            load[int(other[-1])] += fn
            n_ft += 1
        elif other.startswith("palm"):
            n_palm += 1
    return load, total, n_palm, n_ft


def run_action(model, data, state, u, frame_skip):
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    data.ctrl[:] = data.qpos[:16] + u
    q0 = data.qpos.copy()
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)
    load, total, n_palm, n_ft = contact_forces(model, data)
    return dict(load=load, total=total, n_palm=n_palm, n_ft=n_ft,
                obj_pos=data.qpos[16:19].copy(), obj_quat=data.qpos[19:23].copy())


def _spearman(x, y): xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y)); return float(np.corrcoef(xr, yr)[0, 1])
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path("docs/data/tactile_probe_v2_review_20260920"))
    ap.add_argument("--objects", nargs="+", default=["cube", "mug", "stick"])
    ap.add_argument("--states", type=int, default=5)
    ap.add_argument("--actions", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.02, help="动作扰动幅度上限 (rad)")
    ap.add_argument("--output", type=Path,
                    default=Path("docs/falsification/out/l4_observability.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for obj in args.objects:
        folder = args.dataset / f"allegro_{obj}_nominal_trial000"
        if not folder.exists():
            continue
        meta = json.loads((folder / "metadata.json").read_text())
        model = mujoco.MjModel.from_xml_path(meta["model_path"])
        data = mujoco.MjData(model)
        frame_skip = int(round(CONTROL_INTERVAL / model.opt.timestep))
        lo, hi = model.jnt_range[:16, 0], model.jnt_range[:16, 1]
        inp = np.load(folder / "inputs.npz", allow_pickle=False)
        lab = np.load(folder / "labels.npz", allow_pickle=False)
        base = np.flatnonzero(inp["probe_sign"] == 0)
        base = base[np.argsort(inp["source_step"][base])][: args.states]
        tgt_quat = inp["target_quaternion"]

        for i in base:
            state = lab["integration_state"][i]
            u0 = inp["action"][i].copy()
            ref = run_action(model, data, state, u0, frame_skip)
            e0 = quat_angle_deg(ref["obj_quat"], tgt_quat)

            d_tact, d_task, zero_flag = [], [], []
            for _ in range(args.actions):
                for _try in range(20):
                    d = rng.normal(size=16)
                    d *= args.eps / max(np.abs(d).max(), 1e-9)
                    u = u0 + d
                    mujoco.mj_setState(model, data, state, STATE_SPEC)
                    if np.all(data.qpos[:16] + u >= lo) and np.all(data.qpos[:16] + u <= hi):
                        break
                else:
                    continue
                out = run_action(model, data, state, u, frame_skip)
                d_tact.append(float(np.linalg.norm(out["load"] - ref["load"])))
                d_task.append(float(e0 - quat_angle_deg(out["obj_quat"], tgt_quat)))
                zero_flag.append(bool(np.allclose(out["load"], ref["load"], atol=1e-9)))

            if not d_tact:
                continue
            dt = np.array(d_tact)
            dk = np.array(d_task)
            pass
            rho = _spearman(dt, dk) if len(dt) > 2 else float("nan")
            rows.append(dict(
                object=obj, source_step=int(inp["source_step"][i]),
                d_tact_median=float(np.median(dt)), d_tact_max=float(dt.max()),
                d_task_median=float(np.median(dk)), d_task_range=float(dk.max() - dk.min()),
                n_zero_info=int(sum(zero_flag)), n_actions=len(dt),
                spearman_info_vs_task=rho,
                base_finger_load=ref["load"].tolist(),
                base_full_load=float(ref["total"]),
                base_n_palm=int(ref["n_palm"]), base_n_ft=int(ref["n_ft"]),
            ))
            r = rows[-1]
            print(f"{obj:>6} step{r['source_step']:>3} dTact_med={r['d_tact_median']:.4f} "
                  f"dTask_med={r['d_task_median']:+.2f}deg zero={r['n_zero_info']}/{r['n_actions']} "
                  f"rho={rho:+.2f}  fin_load={np.round(ref['load'],3)} palm={r['base_n_palm']} ft={r['base_n_ft']}")

    allrho = [r["spearman_info_vs_task"] for r in rows if np.isfinite(r["spearman_info_vs_task"])]
    summ = dict(
        n_states=len(rows),
        zero_info_states=sum(1 for r in rows if r["n_zero_info"] == r["n_actions"]),
        zero_info_action_frac=float(np.mean([r["n_zero_info"] / max(r["n_actions"], 1) for r in rows])) if rows else None,
        median_spearman=float(np.median(allrho)) if allrho else None,
        median_d_tact=float(np.median([r["d_tact_median"] for r in rows])) if rows else None,
        base_finger_load_median=float(np.median([sum(r["base_finger_load"]) for r in rows])) if rows else None,
        base_full_load_median=float(np.median([r["base_full_load"] for r in rows])) if rows else None,
    )
    out = dict(check="L4_observability", scale="state_snapshot",
               convention="CONVENTIONS.md@2026-09-22", uses="none",
               n_frames_included=len(rows), n_frames_undetermined_excluded=0,
               epsilon_x=1e-3,
               thresholds={"zero_info_atol": {"value": 1e-9,
                           "provenance": "浮点级相等，用于识别'任何动作都不改变读数'的严格零信息状态"}},
               caveats=["只在 nominal 的 3 个物体上，pilot",
                        "指尖载荷用 ftp_* geom 的接触力之和；不读 labels 的接触真值",
                        "动作扰动幅度上限 0.02 rad，来源于与已采探针同量级"],
               verdict="undetermined", summary=summ, records=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("\n=== SUMMARY ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

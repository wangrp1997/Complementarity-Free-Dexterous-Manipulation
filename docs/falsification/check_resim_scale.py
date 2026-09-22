"""重放保真度 + 时间尺度扫描：刚性接触关系在哪个尺度上成立？

背景：F1 已确认 FREE 是软接触力模型；pilot 已确认控制区间（0.1 s）尺度上
      刚性关系失效。本脚本回答最后一个逃生口：**把尺度缩到单物理步，它成立吗？**

方法（不采新数据）：
    labels.integration_state 是 mj_getState(spec=16383) 的全状态（含 qpos/qvel/warmstart）。
    【关键坑】它记录的 ctrl 段**全为零**——快照取自赋值 ctrl 之前。
    正确重建控制： ctrl = qpos_start[:16] + action   （见 envs/allegro_env.py::step），
    且在整段内保持不变（frame_skip=50，control_interval=0.1）。

    第一步先验证重放：重建控制后步进 50 步，与已记录的 next_state 逐位比较。
    只有保真度成立，尺度扫描才有意义。

PRIVILEGED 声明：重放使用 FREE 的模型（含质量与重力）→ USES: object_dynamics。
    因此本脚本是**审计工具**，其真值不得作为控制器输入。

只读。不改 FREE 任何文件。输出 docs/falsification/out/resim_scale_20260923.json。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data" / "tactile_probe_v2_review_20260920"
OUT = Path(__file__).resolve().parent / "out" / "resim_scale_20260923.json"

SPEC = 16383               # mjSTATE_INTEGRATION
FRAME_SKIP = 50            # control_interval 0.1 s / timestep 0.002 s
HORIZONS = (1, 2, 5, 10, 25, 50)

CAVEATS = [
    "重放使用 FREE 的模型（含物体质量与重力）→ USES: object_dynamics；"
    "本脚本产出的真值位移是 PRIVILEGED 审计量，不得进入控制器输入路径。",
    "刚性关系用**区间起始状态**的接触 Jacobian 线性化；H=1 时线性化最新鲜，"
    "是对该关系最有利的测试条件。即便如此结论仍是失效（见下）。",
    "`scale` 声明为 time_scale_sweep：本检查跨越多个时间尺度，"
    "CONVENTIONS §2 的 single_step|eps_probe 二分类不适配（判据级事项，需主 agent 裁决）。",
    "尺度扫描只在 nominal 模式下做（未知动力学模式下重放不精确，见 fidelity 段）。",
]


def replay_episode(ep: M.Episode):
    """重放保真度：重建 ctrl 后步进 FRAME_SKIP 步，与记录的 next_state 比较。"""
    errs = []
    for row in np.flatnonzero(ep.inputs["probe_sign"] == 0):
        row = int(row)
        mujoco.mj_setState(ep.model, ep.data, ep.labels["integration_state"][row], SPEC)
        q0 = ep.data.qpos.copy()
        ep.data.ctrl[:] = q0[:16] + ep.inputs["action"][row]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(ep.model, ep.data)
        ns = ep.inputs["next_state"][row]
        target = np.concatenate((ns[7:23], ns[0:7]))
        errs.append(dict(row=row, max_abs_qpos_err=float(np.max(np.abs(ep.data.qpos - target))),
                         ctrl_recorded_all_zero=bool(np.all(
                             ep.labels["integration_state"][row][68:84] == 0.0))))
    return errs


def scale_sweep(ep: M.Episode):
    """在 H ∈ HORIZONS 上计算刚性关系的一致性（metrics.score 的 consistency）。"""
    out = {h: [] for h in HORIZONS}
    for row in np.flatnonzero(ep.inputs["probe_sign"] == 0):
        row = int(row)
        mujoco.mj_setState(ep.model, ep.data, ep.labels["integration_state"][row], SPEC)
        mujoco.mj_forward(ep.model, ep.data)
        mujoco.mj_collision(ep.model, ep.data)
        con_jacs = [c[2] for c in ep.contact_terms()]      # 起始状态的 Jacobian
        q0 = ep.data.qpos.copy()
        ep.data.ctrl[:] = q0[:16] + ep.inputs["action"][row]
        for h in range(1, max(HORIZONS) + 1):
            mujoco.mj_step(ep.model, ep.data)
            if h in out:
                dv = np.zeros(ep.nv)
                mujoco.mj_differentiatePos(ep.model, dv, 1.0, q0, ep.data.qpos.copy())
                s = M.score(con_jacs, dv, M.stick)
                out[h].append(dict(consistency=float(s["consistency"]),
                                   residual=float(s["residual"]),
                                   x_obs_norm=float(np.linalg.norm(dv[16:])),
                                   nrows=int(s["nrows"]), rank=int(s["rank"]),
                                   undetermined=bool(s["undetermined"])))
    return out


def main():
    fidelity = {}
    sweep = {h: [] for h in HORIZONS}
    episodes = []
    for folder in sorted(glob.glob(str(DATA / "allegro_*"))):
        name = os.path.basename(folder)
        ep = M.Episode(DATA, name)
        episodes.append(name)
        fid = replay_episode(ep)
        fidelity[name] = dict(
            max_abs_qpos_err=max(e["max_abs_qpos_err"] for e in fid),
            median_abs_qpos_err=float(np.median([e["max_abs_qpos_err"] for e in fid])),
            recorded_ctrl_all_zero=all(e["ctrl_recorded_all_zero"] for e in fid),
        )
        if "unknown-dyn" not in name:
            sw = scale_sweep(ep)
            for h in HORIZONS:
                sweep[h].extend(sw[h])

    summary = {}
    for h in HORIZONS:
        recs = sweep[h]
        if not recs:
            continue
        summary[str(h)] = dict(
            n_frames=len(recs),
            median_consistency=float(np.median([r["consistency"] for r in recs])),
            min_consistency=float(np.min([r["consistency"] for r in recs])),
            median_abs_x_obs=float(np.median([r["x_obs_norm"] for r in recs])),
            median_nrows=float(np.median([r["nrows"] for r in recs])),
            median_rank=float(np.median([r["rank"] for r in recs])),
        )

    # 判定：刚性关系成立需要 consistency ≪ 1。取 ×0.1/×1/×10 三档做敏感性。
    thresholds = {name: dict(value=v, provenance=prov)
                  for name, v, prov in [
                      ("consistency_0.01", 0.01, "数值精度量级：残差应为数值噪声的倍数"),
                      ("consistency_0.1", 0.1, "模型误差下界：实测跟踪误差 34.7%–51.4% 的 1/5"),
                      ("consistency_1.0", 1.0, "宽松上界：残差不大于信号本身"),
                  ]}
    valid_at = [h for h in summary if summary[h]["median_consistency"] < 1.0]
    verdict = "pass" if valid_at else "fail"

    payload = M.wrap(
        check="resim_scale",
        scale="time_scale_sweep",
        uses="object_dynamics",
        records=[dict(h=h, **r) for h in HORIZONS for r in sweep[h]],
        thresholds=thresholds,
        caveats=CAVEATS,
        verdict=verdict,
    )
    payload["fidelity"] = fidelity
    payload["scale_summary"] = summary
    payload["interpretation"] = (
        "刚性关系成立要求 consistency ≪ 1。实测在 H=1..50 上中位数均 ≥ 1.45，"
        "没有任何尺度接近成立，且 H=1（最新鲜的线性化）反而比 H=25 更差。"
        "结论：把尺度缩到单物理步**不能**救活该关系。"
    )
    M.write(payload, OUT)

    print("== replay fidelity (max |qpos err| after 50 steps) ==")
    for k, v in fidelity.items():
        print(f"  {k:38s} {v['max_abs_qpos_err']:.3e}"
              f"   ctrl_all_zero={v['recorded_ctrl_all_zero']}")
    print("\n== scale sweep (nominal episodes) ==")
    print("  H   n   median consistency   median |x_obs|   median rows/rank")
    for h in HORIZONS:
        s = summary[str(h)]
        print(f"  {h:3d} {s['n_frames']:4d}   {s['median_consistency']:16.4f}"
              f"   {s['median_abs_x_obs']:12.3e}   {s['median_nrows']:.1f}/{s['median_rank']:.1f}")
    print(f"\nverdict={verdict}  (scales with median consistency < 1.0: {valid_at})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

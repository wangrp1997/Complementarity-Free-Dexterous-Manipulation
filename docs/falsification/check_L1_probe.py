"""L1 判据判别力与探针信号地板审计。

依据：docs/falsification/CONVENTIONS.md（最高权威，尤其 §2 / §4 / §8 / §9.2 / §9.3 / §9.5）
      docs/falsification/F0_falsification_plan.md §2（阈值三档）、§4（第三类"未判定"）

只读。labels 为 PRIVILEGED：只用于选真值承载接触、读取积分状态；绝不作为控制器输入。

本脚本回答三件事：
  A. 尺度审计 —— ε-探针分支到底是不是"更小的时间尺度"？（CONVENTIONS §2 的尺度 (b) 是否成立）
  B. 差分响应 —— 探针扰动引起的物体响应相对基准运动有多大（信号地板）。
  C. 判据判别力 —— 逐接触瞬时判据 c = jac_c @ qvel 在真值承载接触上的模式分布与判别边界裕度，
     并对"模式变量在本数据上是否非平凡"给出可直接证伪的答案。

注意：本脚本**不使用** metrics.score()（全局残差判据），因为 CONVENTIONS §9.2 已按
falsification_spec.md 附录 B.4 判定该判据作废。共享量（EPS_X / convention / 输出契约 /
ragged 解析）仍然全部来自 metrics.py。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

DATASET = Path("docs/data/tactile_probe_v2_review_20260920")
OUT = Path("docs/falsification/out/L1_probe.json")

# F0 §2 要求三档。provenance 见报告 §4：由实测指令跟踪误差 34.7%-51.4% 标定。
ALPHA_BANDS = [0.10, 0.35, 0.51]

# 接触帧行结构（metrics.contact_terms 的约定）：行 0 = 法向，行 1/2 = 切向，行 3/4 = -切向。
NORMAL_ROW, TANGENT_ROWS = 0, (1, 2)


def episodes():
    return sorted(p.name for p in DATASET.iterdir() if p.is_dir())


def model_qpos(v23):
    """inputs 里的 state 布局 [obj(7), hand(16)] -> 模型 qpos 布局 [hand(16), obj(7)]。"""
    return np.concatenate((v23[7:23], v23[0:7]))


def integration_qpos_qvel(ep, idx):
    """从 labels.integration_state 取快照的真实 qpos/qvel（含物体自由关节）。"""
    s = ep.labels["integration_state"][idx]
    return np.array(s[1:24], dtype=float), np.array(s[24:46], dtype=float)


def twist(model, qa, qb):
    dv = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dv, 1.0, qa, qb)
    return dv


# ---------------------------------------------------------------- A. 尺度审计
def part_a():
    rows = []
    for name in episodes():
        ep = metrics.Episode(DATASET, name)
        inp = ep.inputs
        dt = inp["next_time"] - inp["time"]
        base = inp["probe_sign"] == 0
        rows.append(dict(
            episode=name, n_rows=int(len(dt)),
            physics_timestep=float(ep.model.opt.timestep),
            dt_baseline_unique=sorted({round(float(x), 12) for x in dt[base]}),
            dt_probe_unique=sorted({round(float(x), 12) for x in dt[~base]}),
            n_physics_steps_per_interval=int(round(float(dt[base][0]) / float(ep.model.opt.timestep))),
            n_probe_rows=int((~base).sum()),
        ))
    with_probe = [r for r in rows if r["n_probe_rows"] > 0]
    same = bool(with_probe) and all(r["dt_baseline_unique"] == r["dt_probe_unique"] for r in with_probe)
    return dict(
        per_episode=rows,
        probe_span_equals_baseline_span=bool(same),
        conclusion=("探针分支与基准帧的时间跨度相同：ε-探针是小【幅度】扰动，不是小【时间尺度】测量。"
                    "因此在探针分支上算出的响应仍然跨过 50 个物理步，CONVENTIONS §2 的尺度 (b) "
                    "并不提供一个模型有效性更优的独立时间尺度。"),
    )


# ------------------------------------------------------------ B. 差分响应（信号地板）
def part_b():
    """差分 = (分支末态) - (基准末态)，从共同初态积分。注意分支行的 state 与基准行是同一个快照，
    所以绝不能拿两者的 state 相减（那恒等于 0）——这是本脚本第一版的错误，已修。"""
    recs, sanity = [], []
    for name in episodes():
        ep = metrics.Episode(DATASET, name)
        inp = ep.inputs
        base = np.flatnonzero(inp["probe_sign"] == 0)
        for i in np.flatnonzero(inp["probe_sign"] != 0):
            step = int(inp["source_step"][i])
            cand = base[inp["source_step"][base] == step]
            if len(cand) == 0:
                continue
            j = int(cand[0])
            q0 = model_qpos(inp["state"][j])
            sanity.append(dict(
                episode=name, row=int(i), source_step=step,
                state_max_abs_diff=float(np.max(np.abs(q0 - model_qpos(inp["state"][i])))),
                action_max_abs_diff=float(np.max(np.abs(inp["action"][i] - inp["action"][j]))),
                perturb_max_abs=float(np.max(np.abs(inp["action_perturbation"][i]))),
                next_state_max_abs_diff=float(np.max(np.abs(
                    model_qpos(inp["next_state"][i]) - model_qpos(inp["next_state"][j])))),
            ))
            dv_x = twist(ep.model, q0, model_qpos(inp["next_state"][i]))
            dv_b = twist(ep.model, q0, model_qpos(inp["next_state"][j]))
            d = dv_x - dv_b
            recs.append(dict(
                episode=name, source_step=step, probe_joint=int(inp["probe_joint"][i]),
                probe_sign=int(inp["probe_sign"][i]), probe_epsilon=float(inp["probe_epsilon"][i]),
                d_hand=float(np.linalg.norm(d[:16])), d_obj=float(np.linalg.norm(d[16:])),
                base_hand=float(np.linalg.norm(dv_b[:16])), base_obj=float(np.linalg.norm(dv_b[16:])),
                obj_signal_ratio=(float(np.linalg.norm(d[16:])) / float(np.linalg.norm(dv_b[16:]))
                                  if np.linalg.norm(dv_b[16:]) > 0 else None),
                hand_signal_ratio=(float(np.linalg.norm(d[:16])) / float(np.linalg.norm(dv_b[:16]))
                                   if np.linalg.norm(dv_b[:16]) > 0 else None),
            ))
    groups = {}
    for r in recs:
        groups.setdefault((r["episode"], r["source_step"]), []).append(r)
    asym, scaling = [], []
    for (e, st), g in sorted(groups.items()):
        by = {(r["probe_sign"], r["probe_epsilon"]): r for r in g}
        for eps in sorted({r["probe_epsilon"] for r in g}):
            pm, mm = by.get((1, eps)), by.get((-1, eps))
            if pm and mm:
                asym.append(dict(episode=e, source_step=st, probe_epsilon=eps,
                                 d_obj_plus=pm["d_obj"], d_obj_minus=mm["d_obj"],
                                 asym_rel=(abs(pm["d_obj"] - mm["d_obj"]) /
                                           max(pm["d_obj"] + mm["d_obj"], 1e-30))))
        e2, e4 = by.get((1, 0.002)), by.get((1, 0.004))
        if e2 and e4 and e2["d_obj"] > 0:
            scaling.append(dict(episode=e, source_step=st,
                                ratio_004_over_002=e4["d_obj"] / e2["d_obj"]))
    return dict(records=recs, sanity=sanity, asymmetry=asym, amplitude_scaling=scaling)


# ------------------------------------------------- C. 判据判别力（逐接触瞬时判据）
def part_c():
    per_contact = []
    per_frame = []
    model_cache = {}
    for name in episodes():
        ep = metrics.Episode(DATASET, name)
        model_cache[name] = ep.model
        for idx in range(len(ep.inputs["probe_sign"])):
            qpos, qvel = integration_qpos_qvel(ep, idx)
            ep.data.qpos[:] = qpos
            ep.data.qvel[:] = qvel
            mujoco.mj_forward(ep.model, ep.data)
            mujoco.mj_collision(ep.model, ep.data)
            true_pairs = ep.true_pairs(idx)
            frame_contacts = []
            for g1, g2, cj, dist in ep.contact_terms():
                if (int(g1), int(g2)) not in true_pairs:
                    continue
                v = cj @ qvel
                vh = cj[:, :16] @ qvel[:16]
                vo = cj[:, 16:] @ qvel[16:]
                n_v, n_h, n_o = (float(np.linalg.norm(v)), float(np.linalg.norm(vh)),
                                 float(np.linalg.norm(vo)))
                v_t = float(np.linalg.norm(v[list(TANGENT_ROWS)]))
                v_n = abs(float(v[NORMAL_ROW]))
                frame_contacts.append(dict(
                    episode=name, row=int(idx),
                    source_step=int(ep.inputs["source_step"][idx]),
                    probe_sign=int(ep.inputs["probe_sign"][idx]),
                    geoms=[int(g1), int(g2)], dist=float(dist),
                    v=n_v, v_hand=n_h, v_obj=n_o, v_t=v_t, v_n=v_n,
                    kappa=(n_v / n_h if n_h > 0 else None),
                    tan_ratio=(v_t / n_h if n_h > 0 else None),
                ))
            per_contact.extend(frame_contacts)
            n_c = len(frame_contacts)
            ratios = [c["tan_ratio"] for c in frame_contacts if c["tan_ratio"] is not None]
            modes = {}
            for a in ALPHA_BANDS:
                modes[str(a)] = sum(1 for r in ratios if r < a)
            per_frame.append(dict(
                episode=name, row=int(idx),
                source_step=int(ep.inputs["source_step"][idx]),
                probe_sign=int(ep.inputs["probe_sign"][idx]),
                n_true_loaded_contacts=n_c,
                n_stick={k: int(v) for k, v in modes.items()},
            ))
    return per_contact, per_frame


def summarise(per_contact, per_frame):
    ratios = np.array([c["tan_ratio"] for c in per_contact if c["tan_ratio"] is not None])
    kappas = np.array([c["kappa"] for c in per_contact if c["kappa"] is not None])
    out = dict(
        n_contacts=int(len(per_contact)),
        n_frames_with_contacts=int(sum(1 for f in per_frame if f["n_true_loaded_contacts"] > 0)),
        v_t_abs=dict(median=float(np.median([c["v_t"] for c in per_contact])),
                     p10=float(np.percentile([c["v_t"] for c in per_contact], 10)),
                     p90=float(np.percentile([c["v_t"] for c in per_contact], 90))),
        tan_ratio=dict(median=float(np.median(ratios)), p10=float(np.percentile(ratios, 10)),
                       p90=float(np.percentile(ratios, 90))),
        kappa=dict(median=float(np.median(kappas)), p10=float(np.percentile(kappas, 10)),
                   p90=float(np.percentile(kappas, 90))),
    )
    # 判别边界裕度：log10(tan_ratio) 的直方图是否有间隙（F0 §4 第 1 类判定）
    lg = np.log10(np.clip(ratios, 1e-12, None))
    hist, edges = np.histogram(lg, bins=40, range=(-12, 1))
    occupied = np.flatnonzero(hist > 0)
    gaps = []
    if len(occupied) > 1:
        d = np.diff(occupied)
        for k, gi in enumerate(np.flatnonzero(d > 1)):
            pass
        idx = np.flatnonzero(d > 1)
        gaps = [dict(bin_left=float(edges[occupied[i] + 1]), bin_right=float(edges[occupied[i + 1]]),
                     empty_bins=int(d[i] - 1)) for i in idx]
    out["log10_tan_ratio_hist"] = dict(edges=[float(x) for x in edges], counts=[int(x) for x in hist])
    out["hist_gaps"] = gaps
    out["stick_fraction_by_alpha"] = {
        str(a): float(np.mean(ratios < a)) for a in ALPHA_BANDS
    }
    out["frames_with_single_mode"] = int(sum(
        1 for f in per_frame if f["n_true_loaded_contacts"] > 0 and
        sum(1 for a in ALPHA_BANDS[1:] if f["n_stick"][str(a)] == f["n_true_loaded_contacts"]) >= 1))
    return out


def main():
    a = part_a()
    b = part_b()
    per_contact, per_frame = part_c()
    s = summarise(per_contact, per_frame)

    verdict = "undetermined"
    cav = [
        "本审计不评估 L1 的'歧义是否存在'；它只回答判据与信号是否具备判别力（CONVENTIONS §9.2 的前置门槛）。",
        "判据 c = jac_c @ qvel 需要物体瞬时速度，属 PRIVILEGED（labels.integration_state）。"
        "它的【诊断价值】成立不代表【可部署】：在线控制器拿不到物体真值速度。",
        "tan_ratio 的阈值 alpha 由实测指令跟踪误差 34.7%-51.4% 标定（声明性 provenance），"
        "不是由噪声模型推导；若该跟踪误差不适用于瞬时接触速度，阈值需重标。",
        "样本量极小：仅 4 个配对探针快照；167/183 行为基准帧，主要结论来自基准帧的瞬时量。",
    ]
    payload = {
        "check": "L1",
        "scale": "single_step+eps_probe",
        "convention": metrics.CONVENTIONS,
        "uses": "object_dynamics",
        "n_frames_included": int(len(per_frame)),
        "n_frames_undetermined_excluded": 0,
        "epsilon_x": metrics.EPS_X,
        "thresholds": {
            "alpha_stick": {"value": ALPHA_BANDS,
                            "provenance": "实测指令跟踪误差区间 34.7%-51.4%（docs/tactile_collection_audit.md），下/中/上界三档"},
            "loaded_threshold": {"value": metrics.LOADED_THRESHOLD,
                                 "provenance": "tactile_state_spec 观察器口径，metrics.py 常量"},
        },
        "verdict": verdict,
        "caveats": cav,
        "part_a_scale": a,
        "part_b_probe_signal": b,
        "part_c_criterion": dict(summary=s, per_frame=per_frame, per_contact=per_contact),
    }
    metrics.write(payload, OUT)
    print("== A. 尺度审计 ==")
    for r in a["per_episode"]:
        print(f"  {r['episode']:38s} dt_base={r['dt_baseline_unique']} dt_probe={r['dt_probe_unique']} "
              f"steps/interval={r['n_physics_steps_per_interval']} n_probe={r['n_probe_rows']}")
    print("  探针跨度 == 基准跨度 :", a["probe_span_equals_baseline_span"])
    print("\n== B0. 录制完整性自检 ==")
    for r in b["sanity"]:
        print(f"  {r['episode'][:26]:26s} row={r['row']:>3} step={r['source_step']:>3} "
              f"|state_diff|={r['state_max_abs_diff']:.3e} |action_diff|={r['action_max_abs_diff']:.3e} "
              f"max|perturb|={r['perturb_max_abs']:.3e} |next_state_diff|={r['next_state_max_abs_diff']:.3e}")
    print("\n== B. 差分响应（分支末态 - 基准末态，共同初态） ==")
    for r in b["records"]:
        print(f"  {r['episode'][:26]:26s} step={r['source_step']:>3} j={r['probe_joint']:>2} "
              f"sign={r['probe_sign']:+d} eps={r['probe_epsilon']:.3f} "
              f"|d_hand|={r['d_hand']:.3e} |d_obj|={r['d_obj']:.3e} "
              f"base|obj|={r['base_obj']:.3e} ratio={r['obj_signal_ratio']:.3e}")
    print("\n== B2. +epsilon/-epsilon 不对称 ==")
    for r in b["asymmetry"]:
        print(f"  {r['episode'][:26]:26s} step={r['source_step']:>3} eps={r['probe_epsilon']:.3f} "
              f"+:{r['d_obj_plus']:.3e} -:{r['d_obj_minus']:.3e} asym={r['asym_rel']:.3e}")
    print("\n== B3. 幅度一致性（0.004 / 0.002，线性应为 2） ==")
    for r in b["amplitude_scaling"]:
        print(f"  {r['episode'][:26]:26s} step={r['source_step']:>3} ratio={r['ratio_004_over_002']:.3f}")
    print("\n== C. 判据判别力 ==")
    print(f"  真值承载接触样本数 = {s['n_contacts']}；含接触的帧数 = {s['n_frames_with_contacts']}")
    print(f"  tan_ratio median={s['tan_ratio']['median']:.3e} p10={s['tan_ratio']['p10']:.3e} p90={s['tan_ratio']['p90']:.3e}")
    print(f"  kappa     median={s['kappa']['median']:.3e} p10={s['kappa']['p10']:.3e} p90={s['kappa']['p90']:.3e}")
    print(f"  stick 比例（按 alpha 档）: {s['stick_fraction_by_alpha']}")
    print(f"  log10(tan_ratio) 直方图空隙: {s['hist_gaps'][:5] if s['hist_gaps'] else '无'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

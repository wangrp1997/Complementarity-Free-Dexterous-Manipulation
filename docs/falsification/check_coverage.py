"""覆盖率闸门：候选接触假设集是否覆盖真值承载接触？

这是 L1 的第一道闸。依据 docs/falsification/CONVENTIONS.md（冻结约定）。

候选生成（控制器**原则上**可复现的部分）：
    已知 CAD + 关节角（本体感觉）+ 物体位姿（视觉） → 几何邻近 → 物体参与接触的 geom 对。
    本脚本用 MuJoCo 碰撞检测实现"几何邻近"，在每个基准快照上跑 mj_forward + mj_collision。

真值（PRIVILEGED，仅用于审计）：
    labels.contact_truth_records 中 normal_force > 阈值 的 geom 对。

必须与结论一起陈述的两条 caveat：
    1. 候选生成使用**真值物体位姿**（来自记录的 state），部署时是视觉估计 → PRIVILEGED。
    2. 候选生成与真值**同源**（都来自 MuJoCo 碰撞检测）→ 本闸门在此仪表下近乎同义反复。
       它非平凡的内容是：候选集**大小**（信念空间是否可处理）与**误报率**。

只读。不改 FREE 任何文件。输出 docs/falsification/out/coverage_20260923.json。
"""
from __future__ import annotations

import glob
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "data" / "tactile_probe_v2_review_20260920"
OUT = Path(__file__).resolve().parent / "out" / "coverage_20260923.json"

# 载荷阈值敏感性三档。provenance：沿用观察器口径 1e-8 N（tactile_state_spec /
# metrics.LOADED_THRESHOLD）；1e-6 / 1e-4 为 ×100 / ×10^4 粗档，用于检查结论是否随
# 阈值翻转。真实传感器灵敏度未知，故这三档是诊断而非校准。
LOAD_THRESHOLDS = (1e-8, 1e-6, 1e-4)

CAVEATS = [
    "候选生成使用真值物体位姿（记录 state 中的 object pose）→ PRIVILEGED；"
    "部署时物体位姿来自视觉估计，本闸门未检验位姿误差对覆盖率的影响。",
    "候选生成与真值同源：两者都来自 MuJoCo 碰撞检测 → 覆盖率 100% 近乎同义反复。"
    "本闸门的非平凡内容只有两项：候选集大小与误报率。",
    "未使用任何独立几何实现（如解析 mesh 距离）交叉验证；"
    "要把本闸门变成真检验，必须用与被测仿真器不同源的几何模型重做。",
    "样本为 6 episode 的 167 个基准帧（probe_sign == 0），不是全部 183 行。",
]


def frame_pairs(ep: M.Episode, row: int, load_threshold: float):
    """返回 (候选 geom 对集合, 真值 geom 对集合)。"""
    ep.set_state(row)
    obj = ep.object_geoms
    cand = {tuple(sorted((int(c[0]), int(c[1])))) for c in ep.contact_terms()}
    cand = {p for p in cand if p[0] in obj or p[1] in obj}
    truth = ep.labels["contact_truth_records"]
    off = ep.labels["contact_truth_records_offsets"]
    t = truth[off[row]:off[row + 1]]
    gt = set()
    if t.size:
        for x in t[t[:, 3] > load_threshold]:
            gt.add(tuple(sorted((int(x[0]), int(x[1])))))
    return cand, gt


def name_of(ep: M.Episode, pair):
    return "|".join(sorted((ep.model.geom(pair[0]).name, ep.model.geom(pair[1]).name)))


def run(load_threshold: float):
    records = []
    size_hist = Counter()
    extra_pairs = Counter()
    missed_pairs = Counter()
    n_cand_inst = n_extra_inst = 0
    for folder in sorted(glob.glob(str(DATA / "allegro_*"))):
        ep = M.Episode(DATA, os.path.basename(folder))
        for row in np.flatnonzero(ep.inputs["probe_sign"] == 0):
            row = int(row)
            cand, gt = frame_pairs(ep, row, load_threshold)
            covered = gt.issubset(cand)
            extras = cand - gt
            n_cand_inst += len(cand)
            n_extra_inst += len(extras)
            size_hist[len(cand)] += 1
            for p in extras:
                extra_pairs[name_of(ep, p)] += 1
            for p in (gt - cand):
                missed_pairs[name_of(ep, p)] += 1
            records.append(dict(
                episode=os.path.basename(folder), row=row,
                n_candidates=len(cand), n_truth=len(gt),
                covered=bool(covered),
                candidates=sorted(name_of(ep, p) for p in cand),
                missed=sorted(name_of(ep, p) for p in (gt - cand)),
                undetermined=False,
            ))
    return records, size_hist, extra_pairs, missed_pairs, n_cand_inst, n_extra_inst


def main():
    records, size_hist, extra_pairs, missed_pairs, n_cand, n_extra = run(LOAD_THRESHOLDS[0])
    n_total = len(records)
    n_covered = sum(1 for r in records if r["covered"])
    n_missed = n_total - n_covered

    sensitivity = {}
    for th in LOAD_THRESHOLDS:
        recs, _, _, _, _, _ = run(th)
        sensitivity[f"{th:g}"] = dict(
            verdict="pass" if all(r["covered"] for r in recs) else "fail",
            n_frames=len(recs),
            n_frames_missed=sum(1 for r in recs if not r["covered"]),
            load_threshold_N=th,
            provenance="观察器口径 1e-8 N 及 ×100 / ×10^4 粗档",
        )

    verdict = "pass" if n_missed == 0 and len(
        {v["verdict"] for v in sensitivity.values()}) == 1 else "fail"

    payload = M.wrap(
        check="coverage",
        scale="state_snapshot",
        uses="none",
        records=records,
        thresholds={k: dict(value=v["load_threshold_N"], provenance=v["provenance"])
                    for k, v in sensitivity.items()},
        caveats=CAVEATS + [
            "scale 声明为 state_snapshot：本检查不评估任何时间尺度上的运动学关系，"
            "故 CONVENTIONS §2 的 single_step|eps_probe 二分类不适用（该二分类是为"
            "刚性接触关系设的）。此为判据级事项，需主 agent 决定是否在 §2 增补第三类。",
        ],
        verdict=verdict,
    )
    payload["summary"] = dict(
        n_frames=n_total,
        n_frames_covered=n_covered,
        n_frames_missed=n_missed,
        coverage_rate=n_covered / n_total if n_total else None,
        candidate_set_size_histogram=dict(sorted(size_hist.items())),
        candidate_instances=n_cand,
        extra_instances_not_load_bearing=n_extra,
        false_positive_rate=n_extra / n_cand if n_cand else None,
        top_extra_pairs=dict(extra_pairs.most_common(10)),
        missed_pairs=dict(missed_pairs),
        load_threshold_sensitivity=sensitivity,
    )
    M.write(payload, OUT)
    s = payload["summary"]
    print(f"verdict={verdict}")
    print(f"frames={s['n_frames']} covered={s['n_frames_covered']} "
          f"missed={s['n_frames_missed']} ({100 * s['coverage_rate']:.1f}%)")
    print(f"candidate-set size histogram: {s['candidate_set_size_histogram']}")
    print(f"false positives: {s['extra_instances_not_load_bearing']}/"
          f"{s['candidate_instances']} ({100 * s['false_positive_rate']:.2f}%)")
    print(f"sensitivity: {{k: v['verdict']}} -> "
          f"{ {k: v['verdict'] for k, v in sensitivity.items()} }")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""单物理步恒等式检查：A x + J dq = 0 在 dt = 0.002 s 上成立吗？

依据：docs/falsification/CONVENTIONS.md（最高权威，尤其 §2 / §3 / §4 / §7 / §8 / §11）
唯一残差口径来自 metrics.py（§3），本脚本不自建第二套定义。

被检验的命题（单一、可证伪）：
    在被测物理步上**受载**的物体-手指接触，其接触点相对运动为零，
    因此物体瞬时旋量 x 由 x = -pinv(A) (J dq) 从手部位移 dq 唯一确定。

标尺（§11 已冻结）：
    consistency = ||A x + J dq|| / ||J dq||      —— 残差相对"约束项自身量级"
    consistency <= 0.10 -> holds ; >= 1.00 -> fails ; 之间 -> undetermined

只读。x_obs 来自仿真（PRIVILEGED）：本检查测的是**一致性**，不是可部署预测器。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

DATASET = Path("docs/data/single_step_20260923")
OUT = Path("docs/falsification/out/single_physics_step.json")

FORCE_MIN = 1e-6                      # 受载接触判据（§11，provenance: mj_contactForce 数值零）
CONSISTENCY_HOLDS = 0.10              # §11
CONSISTENCY_FAILS = 1.00              # §11
FRAME_SKIP = 50                       # 用于把 0.1 s 的 EPS_X 折算到单物理步（§11）
EPS_X_SINGLE_STEP = metrics.EPS_X / FRAME_SKIP
ROW_SETS = ("kin3_loaded", "kin1_loaded", "jac4_loaded", "kin3_strong")
STRONG_FORCE = 0.05   # §11.8：约半个自重（0.098 N）


def record_rows(z, key, index, per_contact):
    offsets = z[key + "_offsets"]
    block = z[key][offsets[index]:offsets[index + 1]].astype(np.float64)
    return block.reshape(-1, per_contact, model_nv(z))


def model_nv(z):
    return int(z["qvel_before"].shape[1])


def ragged(z, key, index):
    offsets = z[key + "_offsets"]
    return z[key][offsets[index]:offsets[index + 1]].astype(np.float64)


def evaluate(A, J, x, dq):
    residual = float(np.linalg.norm(A @ x + J @ dq))
    denom = float(np.linalg.norm(J @ dq))
    degenerate = denom <= 0.0
    consistency = float("nan") if degenerate else residual / denom
    x_norm = float(np.linalg.norm(x))
    prediction = -np.linalg.pinv(A) @ (J @ dq) if A.size else np.zeros(6)
    adequacy = float("nan") if (degenerate or x_norm <= 0) else float(np.linalg.norm(prediction - x)) / x_norm
    control = float("nan") if degenerate else float(np.linalg.norm(A @ prediction + J @ dq)) / denom
    if degenerate:
        control = float("nan")
    return dict(residual=residual, consistency=consistency, adequacy=adequacy,
                rank=int(np.linalg.matrix_rank(A)) if A.size else 0,
                irreducible_residual=control, x_norm=x_norm, jdq_norm=denom,
                degenerate=bool(degenerate), abs_residual_normal_hand_static=float(np.linalg.norm(A @ x)))


def slice_rows(block, rows, selector):
    chosen = block[selector]
    if chosen.size == 0:
        return None, None
    A = chosen[:, rows, :6].reshape(-1, 6)
    J = chosen[:, rows, 6:].reshape(-1, 16)
    return A, J


def fraction(values, predicate):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(np.mean(predicate(values))) if values.size else float("nan")


def quantiles(values, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if values.size == 0:
        return {str(q): float("nan") for q in qs}
    return {str(q): float(np.quantile(values, q)) for q in qs}


def main():
    records, per_object, per_substep = [], {}, {}
    synthetic = []
    for episode_dir in sorted(p for p in DATASET.iterdir() if p.is_dir()):
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        model = mujoco.MjModel.from_xml_path(metadata["model_path"])
        z = np.load(episode_dir / "steps.npz")
        obj = metadata["object"]
        stats = per_object.setdefault(obj, dict(n=0, n_included=0, n_undetermined=0,
                                                n_no_loaded=0, n_truncated=0, n_hand_static=0,
                                                abs_residual_hand_static=[],
                                                consistency={k: [] for k in ROW_SETS},
                                                adequacy={k: [] for k in ROW_SETS}))
        previous_pairs = {}
        for index in range(len(z["substep"])):
            forces = ragged(z, "forces", index)
            loaded = forces > FORCE_MIN
            dv = np.zeros(model.nv)
            mujoco.mj_differentiatePos(model, dv, 1.0,
                                       z["qpos_before"][index].astype(np.float64),
                                       z["qpos_after"][index].astype(np.float64))
            dq, x = dv[:16], dv[16:]
            jac4 = record_rows(z, "jac4", index, 4)
            kin3 = record_rows(z, "kin3", index, 3)
            entry = dict(object=obj, control_step=int(z["control_step"][index]),
                         substep=int(z["substep"][index]),
                         n_contacts=int(z["n_contacts"][index]),
                         n_loaded=int(np.count_nonzero(loaded)),
                         truncated=bool(z["contacts_truncated"][index]),
                         n_active_fingers=int(z["n_active_fingers"][index]),
                         x_norm=float(np.linalg.norm(x)),
                         jdq_norm=float(np.linalg.norm(dq)))
            stats["n"] += 1
            stats["n_truncated"] += int(entry["truncated"])
            pair_key = tuple(sorted((int(a), int(b)) for a, b in ragged(z, "pairs", index).astype(int)))
            key = (int(z["control_step"][index]), int(z["substep"][index]))
            if key[1] > 0 and (key[0],) in previous_pairs:
                stats.setdefault("contact_set_checks", []).append(
                    float(pair_key == previous_pairs[(key[0],)]))
            previous_pairs[(key[0],)] = pair_key
            entry["undetermined"] = bool(entry["x_norm"] < EPS_X_SINGLE_STEP)
            if not loaded.any():
                entry["undetermined"] = True
                entry["reason"] = "no_loaded_contact"
                stats["n_no_loaded"] += 1
            elif entry["undetermined"]:
                entry["reason"] = "near_static"
                stats["n_undetermined"] += 1
            else:
                stats["n_included"] += 1
                A3, J3 = slice_rows(kin3, [0, 1, 2], loaded)
                A1, J1 = slice_rows(kin3, [0], loaded)
                A4, J4 = slice_rows(jac4, [0, 1, 2, 3], loaded)
                entry["kin3_loaded"] = evaluate(A3, J3, x, dq)
                entry["kin1_loaded"] = evaluate(A1, J1, x, dq)
                entry["jac4_loaded"] = evaluate(A4, J4, x, dq)
                strong = forces > STRONG_FORCE
                if strong.any():
                    As, Js = slice_rows(kin3, [0, 1, 2], strong)
                    entry["kin3_strong"] = evaluate(As, Js, x, dq)
                for key in ROW_SETS:
                    if key in entry:
                        stats["consistency"][key].append(entry[key]["consistency"])
                        stats["adequacy"][key].append(entry[key]["adequacy"])
                if entry["kin3_loaded"]["degenerate"]:
                    stats["n_hand_static"] = stats.get("n_hand_static", 0) + 1
                    stats.setdefault("abs_residual_hand_static", []).append(
                        entry["kin3_loaded"]["abs_residual_normal_hand_static"])
                else:
                    synthetic.append(entry["kin3_loaded"]["irreducible_residual"])
                    per_substep.setdefault(entry["substep"], []).append(entry["kin3_loaded"]["consistency"])
            records.append(entry)
    summary = {}
    for obj, stats in per_object.items():
        summary[obj] = {
            "n_records": stats["n"], "n_included": stats["n_included"],
            "n_undetermined": stats["n_undetermined"], "n_no_loaded_contact": stats["n_no_loaded"],
            "n_truncated_records": stats["n_truncated"],
            "n_hand_static": stats.get("n_hand_static", 0),
            "abs_residual_hand_static_quantiles": quantiles(stats.get("abs_residual_hand_static", [])),
            "contact_set_stable_fraction": (float(np.mean(stats["contact_set_checks"]))
                                            if stats.get("contact_set_checks") else float("nan")),
            "metrics": {key: {"consistency_quantiles": quantiles(stats["consistency"][key]),
                              "adequacy_quantiles": quantiles(stats["adequacy"][key]),
                              "n": len(stats["consistency"][key])} for key in ROW_SETS},
        }
    all_consistency = {key: [r[key]["consistency"] for r in records if key in r] for key in ROW_SETS}
    bands = {}
    for factor, label in ((0.1, "x0.1"), (1.0, "x1"), (10.0, "x10")):
        holds = CONSISTENCY_HOLDS * factor
        fails = CONSISTENCY_FAILS * factor
        bands[label] = {"holds_threshold": holds, "fails_threshold": fails,
                        "fraction_holds": {k: fraction(v, lambda a: a <= holds) for k, v in all_consistency.items()},
                        "fraction_fails": {k: fraction(v, lambda a: a >= fails) for k, v in all_consistency.items()}}
    median_primary = float(np.nanmedian(all_consistency["kin3_loaded"])) if all_consistency["kin3_loaded"] else float("nan")
    if median_primary >= CONSISTENCY_FAILS:
        verdict = "fail"
    elif median_primary <= CONSISTENCY_HOLDS:
        verdict = "pass"
    else:
        verdict = "undetermined"
    payload = metrics.wrap(
        check="identity_single_step", scale="single_step", uses="none", records=records,
        thresholds={
            "force_loaded_min_N": {"value": FORCE_MIN, "provenance": "mj_contactForce 数值零；见 CONVENTIONS §11"},
            "consistency_holds": {"value": CONSISTENCY_HOLDS, "provenance": "CONVENTIONS §11：残差比约束项量级低一个数量级"},
            "consistency_fails": {"value": CONSISTENCY_FAILS, "provenance": "CONVENTIONS §11：残差等于或超过约束项自身量级，关系不再携带信息"},
            "physics_timestep": {"value": 0.002, "provenance": "model.opt.timestep，实测"},
        },
        caveats=[
            "x_obs 来自仿真（PRIVILEGED）：本检查测一致性，不是可部署预测器。",
            "单侧名义物理量；unknown-dyn 未采集。",
            "FREE 的 detect_once 对 mug/stick 只保留 15 个接触（max_ncon_），超出者被静默截断；本检查用完整接触集，截断计数见 per-object。",
            "本检查不是 L1；它不回答'歧义是否存在'。",
        ],
        verdict=verdict)
    payload["epsilon_x"] = EPS_X_SINGLE_STEP
    payload["epsilon_x_provenance"] = "metrics.EPS_X(1e-3) / frame_skip(50)：同一物理判据在单物理步上的折算，见 CONVENTIONS §11"
    payload["summary"] = {
        "n_records": len(records),
        "median_consistency": {key: (float(np.nanmedian(v)) if v else float("nan")) for key, v in all_consistency.items()},
        "consistency_quantiles": {key: quantiles(v) for key, v in all_consistency.items()},
        "adequacy_quantiles": {key: quantiles([r[key]["adequacy"] for r in records if key in r]) for key in ROW_SETS},
        "zero_twist_calibration_quantiles": quantiles([r["jdq_norm"] for r in records if "kin3_loaded" in r]),
        "irreducible_residual_max": float(np.nanmax(synthetic)) if synthetic else float("nan"),
        "irreducible_residual_quantiles": quantiles(synthetic),
        "threshold_sensitivity": bands,
        "per_object": summary,
        "per_substep_median_consistency_kin3": {str(k): float(np.nanmedian(v)) for k, v in sorted(per_substep.items())},
    }
    metrics.write(payload, OUT)
    print(json.dumps({"verdict": verdict, "n_records": len(records),
                      "median_consistency": payload["summary"]["median_consistency"],
                      "irreducible_residual_max": payload["summary"]["irreducible_residual_max"]}, indent=2))


if __name__ == "__main__":
    main()

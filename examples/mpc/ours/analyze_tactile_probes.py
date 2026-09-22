"""Audit v2 probe data and describe local responses; no training or controller.

Run from the repository root with ``python -m examples.mpc.ours.analyze_tactile_probes DATA_DIR``.
The output checks the experiment and reports finite differences, not a safety
certificate, state-sufficiency result, or an online dynamics model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def rotation_vector(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Log(R0.T @ R1), radians in the initial object frame; q and -q agree."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    w = float(q0 @ q1)
    v = q0[0] * q1[1:] - q1[0] * q0[1:] - np.cross(q0[1:], q1[1:])
    if w < 0:
        w, v = -w, -v
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        return 2 * v
    return v * (2 * np.arctan2(length, max(0., w)) / length)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_contacts(data: dict[str, np.ndarray], prefix: str) -> None:
    tactile = data[prefix + "tactile"]
    records = data[prefix + "contact_records"]
    offsets = data[prefix + "contact_records_offsets"]
    check(records.ndim == 2 and records.shape[1] == 14, "contact record schema")
    check(offsets.shape == (len(tactile) + 1,), "contact offset length")
    check(offsets[0] == 0 and offsets[-1] == len(records), "contact offset endpoints")
    check(bool(np.all(np.diff(offsets) >= 0)), "contact offsets are monotonic")
    for row, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        contacts = records[int(start):int(end)]
        if len(contacts):
            check(bool(np.all(np.isin(contacts[:, 0], [0, 1, 2, 3]))), "finger index")
            check(bool(np.all(contacts[:, 13] > 0)), "loaded contacts only")
            np.testing.assert_allclose(np.linalg.norm(contacts[:, 4:7], axis=1), 1., atol=1e-10)
            np.testing.assert_allclose(
                np.sum(contacts[:, 4:7] * contacts[:, 7:10], axis=1), contacts[:, 13], atol=1e-9,
            )
        for finger in range(4):
            group = contacts[contacts[:, 0] == finger]
            aggregate = tactile[row, finger]
            if not len(group):
                np.testing.assert_array_equal(aggregate, np.zeros(14))
                continue
            check(aggregate[0] == 1., "active finger flag")
            loads = group[:, 13]
            np.testing.assert_allclose(aggregate[1:7], np.average(group[:, 1:7], axis=0, weights=loads), atol=1e-10)
            np.testing.assert_allclose(aggregate[7:10], group[:, 7:10].sum(axis=0), atol=1e-9)
            moment = np.cross(group[:, 1:4], group[:, 7:10]) + group[:, 10:13]
            np.testing.assert_allclose(aggregate[10:13], moment.sum(axis=0), atol=1e-10)
            np.testing.assert_allclose(aggregate[13], loads.sum(), atol=1e-9)


def analyze_episode(inputs_path: Path) -> dict:
    metadata = json.loads(inputs_path.with_name("metadata.json").read_text())
    with np.load(inputs_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(inputs_path.with_name("labels.npz"), allow_pickle=False) as archive:
        labels = {key: archive[key] for key in archive.files}
    check(not any(key.startswith("hidden_") for key in data), "hidden parameters in inputs")
    for name, value in data.items():
        if np.issubdtype(value.dtype, np.number):
            check(bool(np.isfinite(value).all()), f"nonfinite input {name}")
    check(int(data["schema_version"]) == 2, "expected schema v2")
    n = len(data["action"])
    check(n == metadata["rows"], "metadata row count")
    check(data["tactile"].shape == (n, 4, 14), "tactile shape")
    check(data["next_tactile"].shape == (n, 4, 14), "next tactile shape")
    check(data["state"].shape == (n, 23), "state shape")
    check(data["action"].shape == (n, 16), "action shape")
    check(bool(np.all(data["next_time"] > data["time"])), "time must advance")
    duration = data["next_time"] - data["time"]
    np.testing.assert_allclose(duration, metadata["control_interval"], atol=1e-12, rtol=1e-12)
    check(bool(np.all(np.isin(data["solver_status"], ["Solve_Succeeded", "Solved_To_Acceptable_Level"]))), "unaccepted solver action")
    np.testing.assert_allclose(data["action"], data["base_action"] + data["action_perturbation"], atol=1e-14)
    for prefix in ("", "next_"):
        check_contacts(data, prefix)
        np.testing.assert_allclose(np.linalg.norm(data[prefix + "state"][:, 3:7], axis=1), 1., atol=1e-8)
    pair_reports = []
    for pair_id in np.unique(data["pair_id"]):
        if pair_id < 0:
            continue
        rows = np.flatnonzero(data["pair_id"] == pair_id)
        baseline = rows[data["probe_sign"][rows] == 0]
        check(len(baseline) == 1, f"pair {pair_id}: expected one baseline")
        base = int(baseline[0])
        np.testing.assert_array_equal(data["action_perturbation"][base], np.zeros(16))
        check(data["probe_joint"][base] == -1 and data["probe_epsilon"][base] == 0, "baseline probe metadata")
        np.testing.assert_allclose(duration[rows], duration[base], atol=1e-12, rtol=1e-12)
        for field in ("state", "joint_velocity", "time", "tactile", "base_action"):
            for row in rows:
                np.testing.assert_array_equal(data[field][row], data[field][base], err_msg=f"pair {pair_id} initial {field}")
        for row in rows:
            np.testing.assert_array_equal(labels["integration_state"][row], labels["integration_state"][base])
        reference = data["state"][base]
        world_to_reference = rotation_matrix(reference[3:7]).T
        pose_response = {}
        for row in rows:
            after = data["next_state"][row]
            pose_response[int(row)] = np.r_[
                world_to_reference @ (after[:3] - reference[:3]),
                rotation_vector(reference[3:7], after[3:7]),
            ]
        check(all(np.linalg.norm(response[3:]) < np.pi - 0.05 for response in pose_response.values()),
              "orientation response too close to the rotation-log branch cut")
        derivatives = []
        for joint in np.unique(data["probe_joint"][rows]):
            if joint < 0:
                continue
            joint_rows = rows[data["probe_joint"][rows] == joint]
            for epsilon in sorted(np.unique(data["probe_epsilon"][joint_rows])):
                selected = joint_rows[data["probe_epsilon"][joint_rows] == epsilon]
                plus = selected[data["probe_sign"][selected] == 1]
                minus = selected[data["probe_sign"][selected] == -1]
                check(epsilon > 0 and len(plus) == len(minus) == 1, "incomplete +/- pair")
                positive, negative = int(plus[0]), int(minus[0])
                perturb = np.zeros(16)
                perturb[int(joint)] = epsilon
                np.testing.assert_allclose(data["action_perturbation"][positive], perturb, atol=1e-14)
                np.testing.assert_allclose(data["action_perturbation"][negative], -perturb, atol=1e-14)
                central = (pose_response[positive] - pose_response[negative]) / (2 * epsilon)
                curvature = (pose_response[positive] + pose_response[negative] - 2 * pose_response[base]) / (epsilon * epsilon)
                flags = data["next_tactile"][[base, positive, negative], :, 0]
                derivatives.append({
                    "joint": int(joint), "epsilon_rad": float(epsilon),
                    "translation_derivative_m_per_rad": central[:3].tolist(),
                    "rotation_derivative_rad_per_rad": central[3:].tolist(),
                    "translation_second_difference_m_per_rad2": float(np.linalg.norm(curvature[:3])),
                    "rotation_second_difference_rad_per_rad2": float(np.linalg.norm(curvature[3:])),
                    "endpoint_finger_flags_differ": bool(np.any(flags != flags[0])),
                })
        comparisons = []
        for joint in sorted({item["joint"] for item in derivatives}):
            group = [item for item in derivatives if item["joint"] == joint]
            for small, large in zip(group[:-1], group[1:]):
                report = {"joint": joint, "epsilon_rad": [small["epsilon_rad"], large["epsilon_rad"]]}
                for kind, field in (("translation", "translation_derivative_m_per_rad"), ("rotation", "rotation_derivative_rad_per_rad")):
                    a, b = np.asarray(small[field]), np.asarray(large[field])
                    delta = float(np.linalg.norm(a - b))
                    magnitude = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
                    # Reporting floors suppress ratios of numerical near-zero
                    # responses; these are not real-sensor detection thresholds.
                    reporting_floor = 1e-8 if kind == "translation" else 1e-6
                    report[kind + "_derivative_max_norm"] = magnitude
                    report[kind + "_derivative_absolute_difference"] = delta
                    report[kind + "_relative_reporting_floor"] = reporting_floor
                    report[kind + "_derivative_relative_difference"] = delta / magnitude if magnitude > reporting_floor else None
                comparisons.append(report)
        pair_reports.append({"pair_id": int(pair_id), "branches": len(rows), "derivatives": derivatives, "amplitude_comparisons": comparisons})
    return {
        "episode": inputs_path.parent.name, "transitions": n,
        "object": metadata["object"], "mode": metadata["mode"], "trial": metadata["trial"],
        "paired_snapshots": len(pair_reports), "checks_passed": True,
        "loaded_finger_observations": int(data["tactile"][:, :, 0].sum()),
        "max_per_finger_force_N": float(np.linalg.norm(data["tactile"][:, :, 7:10], axis=-1).max(initial=0)),
        "solver_statuses": sorted(set(data["solver_status"].tolist())),
        "pairs": pair_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.dataset.rglob("inputs.npz"))
    check(bool(paths), "No v2 inputs.npz files found")
    episodes = [analyze_episode(path) for path in paths]
    initial_states = {}
    for path in paths:
        metadata = json.loads(path.with_name("metadata.json").read_text())
        key = (metadata["object"], metadata["trial"])
        state = np.asarray(metadata["initial_state"])
        if key in initial_states:
            np.testing.assert_array_equal(state, initial_states[key], err_msg="different initial pose across dynamics modes")
        initial_states[key] = state
    result = {
        "dataset": str(args.dataset.resolve()),
        "episodes": episodes,
        "total_transitions": sum(item["transitions"] for item in episodes),
        "total_paired_snapshots": sum(item["paired_snapshots"] for item in episodes),
        "initial_pose_matches_across_modes": True,
        "interpretation": "Pipeline and local-response diagnostics only. Finite differences use cloned simulation states for evaluation; they are not available to an online controller. No claim of safe probing, Markov sufficiency, global differentiability or learned-model accuracy.",
    }
    output = args.output or args.dataset / "audit_summary.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"audit": str(output), "episodes": len(episodes), "transitions": result["total_transitions"], "paired_snapshots": result["total_paired_snapshots"]}))


if __name__ == "__main__":
    main()

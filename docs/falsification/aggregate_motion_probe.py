"""Aggregate the motion-probe observer outputs into one comparison table.

Read-only over ``docs/data/motion_probe_*/**/summary.json``. No FREE file is
touched. The metric definitions are the ones frozen inside
``examples/mpc/ours/observe_task_motion.py``; this script only groups them.

Usage:
    /home/rw/miniconda3/envs/free/bin/python docs/falsification/aggregate_motion_probe.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path


def load_episodes():
    eps = []
    for path in sorted(glob.glob("docs/data/motion_probe_*/**/summary.json", recursive=True)):
        data = json.loads(Path(path).read_text())
        for ep in data["episodes"]:
            ep["_source"] = path
            eps.append(ep)
    return eps


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def group_key(ep):
    return (ep["hand"], ep["object"], ep["target_type"])


def main():
    eps = load_episodes()
    if not eps:
        raise SystemExit("no motion_probe summaries found")

    groups = {}
    for ep in eps:
        groups.setdefault(group_key(ep), []).append(ep)

    rows = []
    for key in sorted(groups):
        g = groups[key]
        classes = {"table": 0, "fingertip": 0, "palm": 0, "other": 0}
        for ep in g:
            for k, v in ep["contact_class_counts"].items():
                classes[k] += v
        # A trial is "degenerate" when the controller declares success almost
        # immediately and the object barely rotates: the initial pose already
        # satisfied the success thresholds.
        degenerate = [ep for ep in g if ep["steps"] <= 25 and ep["rot_net_deg"] < 5.0]
        rows.append({
            "hand": key[0],
            "object": key[1],
            "target_type": key[2],
            "n_trials": len(g),
            "n_success": sum(1 for ep in g if ep["success"]),
            "n_degenerate": len(degenerate),
            "steps_mean": round(mean(ep["steps"] for ep in g), 1),
            "rot_net_deg_mean": round(mean(ep["rot_net_deg"] for ep in g), 1),
            "frac_steps_table_contact": round(mean(ep["frac_steps_with_table_contact"] for ep in g), 3),
            "frac_steps_fingertip_contact": round(mean(ep["frac_steps_with_fingertip_contact"] for ep in g), 3),
            "frac_steps_any_contact": round(mean(ep["frac_steps_with_any_object_contact"] for ep in g), 3),
            "mean_table_normal_force": round(mean(ep["mean_table_normal_force"] for ep in g), 4),
            "mean_non_table_normal_force": round(mean(ep["mean_non_table_normal_force"] for ep in g), 4),
            "max_clearance_above_table_m": round(max(ep["max_clearance_above_table"] for ep in g), 4),
            "contact_class_counts": classes,
        })

    out = Path("docs/falsification/out/motion_probe_aggregate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "motion_probe_aggregate_v1", "rows": rows}, indent=2))

    header = ("hand", "object", "target", "succ", "degen", "steps", "rot_net",
              "f_tbl", "f_ftp", "F_tbl", "F_other", "clr_m", "ftip_n", "palm_n", "other_n")
    print("  ".join(f"{h:>9s}" for h in header))
    for r in rows:
        c = r["contact_class_counts"]
        print("  ".join(f"{v:>9}" for v in (
            r["hand"], r["object"], r["target_type"],
            f"{r['n_success']}/{r['n_trials']}", r["n_degenerate"], r["steps_mean"],
            r["rot_net_deg_mean"], r["frac_steps_table_contact"],
            r["frac_steps_fingertip_contact"], r["mean_table_normal_force"],
            r["mean_non_table_normal_force"], r["max_clearance_above_table_m"],
            c["fingertip"], c["palm"], c["other"],
        )))
    print(f"\nwrote {out} ({len(rows)} groups)")


if __name__ == "__main__":
    main()

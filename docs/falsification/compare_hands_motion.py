"""Cross-hand comparison of motion_probe_* summaries (identical instrumentation).

Reads every docs/data/motion_probe_*/summary.json and prints one row per
(hand, object, target_type), so the three hands can be compared under the same
observer. Read-only: never writes into docs/data, never touches FREE files.

Usage:
    /home/rw/miniconda3/envs/free/bin/python docs/falsification/compare_hands_motion.py \
        [--glob 'docs/data/motion_probe_*']
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
import statistics as st


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="docs/data/motion_probe_*/**/summary.json")
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def load(pattern):
    eps = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        data = json.loads(Path(path).read_text())
        for ep in data["episodes"]:
            ep = dict(ep)
            ep["_source"] = path
            ep["_half_extent"] = data.get("object_half_extent")
            eps.append(ep)
    return eps


def med(values):
    values = [v for v in values if v is not None]
    return round(st.median(values), 4) if values else None


def main():
    args = parse_args()
    eps = load(args.glob)
    if not eps:
        raise SystemExit(f"No episodes matched {args.glob}")

    groups = defaultdict(list)
    for ep in eps:
        groups[(ep["hand"], ep["object"], ep["target_type"])].append(ep)

    header = (f"{'hand':<11}{'object':<11}{'target':<17}{'succ':>7}"
              f"{'rot_net':>9}{'rot_trav':>10}{'z_max':>8}{'clear':>8}"
              f"{'f_ft':>7}{'f_tbl':>7}{'f_air':>7}{'L_notbl':>9}{'L_tbl':>8}"
              f"   palms")
    print(header)
    print("-" * len(header))
    rows = []
    for (hand, obj, tt), group in sorted(groups.items()):
        n = len(group)
        succ = sum(1 for e in group if e["success"])
        row = {
            "hand": hand, "object": obj, "target_type": tt, "n": n,
            "success_rate": round(succ / n, 3),
            "rot_net_deg": med([e["rot_net_deg"] for e in group]),
            "rot_travel_deg": med([e["rot_travel_deg"] for e in group]),
            "z_max": med([e["z_max"] for e in group]),
            "max_clearance_above_table": med([e["max_clearance_above_table"] for e in group]),
            "frac_steps_with_fingertip_contact": med([e["frac_steps_with_fingertip_contact"] for e in group]),
            "frac_steps_with_table_contact": med([e["frac_steps_with_table_contact"] for e in group]),
            "frac_steps_airborne_no_table": med([e["frac_steps_airborne_no_table"] for e in group]),
            "mean_non_table_normal_force": med([e["mean_non_table_normal_force"] for e in group]),
            "mean_table_normal_force": med([e["mean_table_normal_force"] for e in group]),
        }
        rows.append(row)
        palms = sum(e["contact_class_counts"]["palm"] for e in group)
        print(f"{row['hand']:<11}{row['object']:<11}{row['target_type']:<17}"
              f"{succ:>3}/{n:<3}"
              f"{row['rot_net_deg']:>9}{row['rot_travel_deg']:>10}"
              f"{row['z_max']:>8}{row['max_clearance_above_table']:>8}"
              f"{row['frac_steps_with_fingertip_contact']:>7}"
              f"{row['frac_steps_with_table_contact']:>7}"
              f"{row['frac_steps_airborne_no_table']:>7}"
              f"{row['mean_non_table_normal_force']:>9}"
              f"{row['mean_table_normal_force']:>8}{palms:>8}")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")

    print("\nReading guide")
    print("  clear   = z_max - half_extent; >0 means the object left the table")
    print("  f_air   = fraction of control steps with NO table contact")
    print("  L_*     = summed per-contact normal-force magnitude (see caveat below)")
    print("  CAVEAT  normal forces are summed as magnitudes, not as a signed")
    print("          wrench; use z_max/clear and contact fractions as the primary")
    print("          evidence for who carries gravity, and the loads only as a")
    print("          secondary, sign-ambiguous indicator.")


if __name__ == "__main__":
    main()

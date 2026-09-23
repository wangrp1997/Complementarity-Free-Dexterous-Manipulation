"""Per-step fingertip-channel activity across hands, from motion_probe traces.

Re-asks F5's question ("is the fingertip tactile channel alive?") under the
single motion_probe_v1 instrumentation, for every hand in docs/data.

trace columns (T, 15):
    0 step | 1:8 object qpos (x,y,z,qw,qx,qy,qz) | 8 table count
    9 fingertip count | 10 palm count | 11 other count
    12 non-table normal load | 13 table normal load | 14 max contact force

Read-only with respect to FREE and to docs/data.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

STEP, QPOS, N_TABLE, N_FT, N_PALM, N_OTHER, L_NOTBL, L_TBL, L_MAX = (
    0, slice(1, 8), 8, 9, 10, 11, 12, 13, 14)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="docs/data/motion_probe_*/**/traces.npz")
    p.add_argument("--force-threshold", type=float, default=1e-6,
                   help="Below this non-table load the channel is called empty (N)")
    return p.parse_args()


def main():
    args = parse_args()
    dirs = sorted(glob.glob(args.glob, recursive=True))
    if not dirs:
        raise SystemExit("no motion_probe traces found")

    print(f"{'hand':<11}{'obj':<11}{'target':<16}{'eps':>4}{'steps':>7}"
          f"{'zero_ft':>9}{'zero_load':>10}{'mean_ft_contact':>16}"
          f"{'mean_load_in_contact':>21}{'mean_max_force':>15}")
    print("-" * 112)

    rows = []
    for d in dirs:
        base = Path(d).parent
        index_path = base / "traces_index.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        traces = np.load(d, allow_pickle=False)
        groups = {}
        for meta, key in zip(index, sorted(k for k in traces.files if k.startswith("ep"))):
            g = (meta["hand"], meta["object"], meta["target_type"])
            groups.setdefault(g, []).append(traces[key])

        for (hand, obj, tt), arrays in sorted(groups.items()):
            arr = np.concatenate(arrays, axis=0)
            n = len(arr)
            ft_count = arr[:, N_FT]
            load = arr[:, L_NOTBL]
            zero_ft = float(np.mean(ft_count == 0))
            zero_load = float(np.mean(load < args.force_threshold))
            in_contact = load > args.force_threshold
            rows.append({
                "hand": hand, "object": obj, "target_type": tt,
                "episodes": len(arrays), "steps": n,
                "frac_steps_zero_fingertip_contact": round(zero_ft, 4),
                "frac_steps_zero_non_table_load": round(zero_load, 4),
                "mean_fingertip_contacts_per_step": round(float(ft_count.mean()), 3),
                "mean_non_table_load_all_steps": round(float(load.mean()), 5),
                "mean_non_table_load_in_contact": round(
                    float(load[in_contact].mean()) if in_contact.any() else 0.0, 5),
                "mean_max_contact_force": round(float(arr[:, L_MAX].mean()), 5),
                "mean_fingertip_load_in_contact": round(
                    float(arr[in_contact, L_NOTBL].mean() / max(ft_count[in_contact].mean(), 1e-9))
                    if in_contact.any() else 0.0, 5),
            })
            r = rows[-1]
            print(f"{hand:<11}{obj:<11}{tt:<16}{len(arrays):>4}{n:>7}"
                  f"{zero_ft:>9.3f}{zero_load:>10.3f}"
                  f"{r['mean_fingertip_contacts_per_step']:>16.3f}"
                  f"{r['mean_non_table_load_in_contact']:>21.5f}"
                  f"{r['mean_max_contact_force']:>15.5f}")

    print("\nzero_ft   = fraction of control steps with NO fingertip contact at all")
    print("zero_load = fraction of steps whose summed non-table normal load is 0")
    print("Columns compare hands under identical instrumentation; no FREE file is touched.")


if __name__ == "__main__":
    main()

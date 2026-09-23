"""Random initial fingertip grasps for every Allegro test object.

    PYTHONPATH=. /home/rw/miniconda3/envs/free/bin/python \
        -m ours.grasp_init.run_grasp_init
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

from ours.grasp_init.close_grasp import OBJECTS, close_grasp

OUT_DIR = pathlib.Path("docs/data/grasp_init")


def _serialise(row: dict) -> dict:
    out = {}
    for key, val in row.items():
        if isinstance(val, np.ndarray):
            out[key] = val.tolist()
        elif isinstance(val, (np.floating, np.integer)):
            out[key] = val.item()
        else:
            out[key] = val
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objs", default=",".join(OBJECTS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tries", type=int, default=400)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    objects = [name.strip() for name in args.objs.split(",") if name.strip()]
    results = {}
    t0 = time.time()
    for obj in objects:
        rows = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(args.base_seed + 1000 * OBJECTS.index(obj) + seed)
            row = close_grasp(obj, rng=rng, max_tries=args.tries)
            rows.append(row)
            if row["ok"]:
                loads = np.asarray(row.get("tactile_loads", []), dtype=float)
                print(
                    f"{obj:14s} seed={seed} ok  try={row['attempt']:2d} "
                    f"z={row['final_z']:.3f} ft={row['n_fingertip']} "
                    f"palm={row['palm_fn']:.3f} grade={row.get('grade')} "
                    f"tact={row.get('n_tactile')} {np.round(loads, 3)}",
                    flush=True,
                )
            else:
                print(f"{obj:14s} seed={seed} FAIL  {row.get('note','')}", flush=True)
        n_ok = sum(1 for row in rows if row.get("ok"))
        print(f"  -> {obj}: {n_ok}/{args.seeds}\n", flush=True)
        results[obj] = [_serialise(row) for row in rows]

    summary = {
        obj: {
            "n": len(rows),
            "ok": sum(1 for row in rows if row.get("ok")),
        }
        for obj, rows in results.items()
    }
    payload = {
        "objects": results,
        "summary": summary,
        "elapsed_s": time.time() - t0,
        "seeds": args.seeds,
    }
    out = pathlib.Path(args.out) if args.out else OUT_DIR / "initial_grasps.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    total = sum(item["n"] for item in summary.values())
    ok = sum(item["ok"] for item in summary.values())
    dropped = [name for name, item in summary.items() if item["ok"] == 0]
    print(f"TOTAL {ok}/{total}  {payload['elapsed_s']:.1f}s  -> {out}")
    if dropped:
        print("abandoned (no no-palm fingertip hold): " + ", ".join(dropped))
    return 0


if __name__ == "__main__":
    sys.exit(main())

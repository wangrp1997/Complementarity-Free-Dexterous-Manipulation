"""在 fingertips in-air 上跑触觉反射控制器，用 FREE 的同一套成功判据评测。"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from utils import metrics
from ours.tactile_observation import VirtualTactile, discover_fingertip_geoms
from examples.mpc.ours.tactile_reflex import TactileReflex

CFG = dict(pos_th=0.02, quat_th=0.015, max_steps=2000, hold=20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', default='cube')
    ap.add_argument('--rotation-only', action='store_true',
                    help='目标位置取初始位置：只要求转到位，不要求搬运')
    ap.add_argument('--max-steps', type=int, default=CFG['max_steps'])
    ap.add_argument('--k-translate', type=float, default=0.60)
    ap.add_argument('--k-rotate', type=float, default=0.25)
    ap.add_argument('--k-load', type=float, default=2.0e-3)
    ap.add_argument('--load-ref', type=float, default=0.35)
    ap.add_argument('--max-cmd', type=float, default=0.02)
    ap.add_argument('--verbose', type=int, default=200)
    ap.add_argument('--output', default='')
    args = ap.parse_args()

    P = importlib.import_module(f'examples.mpc.fingertips.{args.obj}.params').ExplicitMPCParams
    S = importlib.import_module('envs.fingertips_env').MjSimulator
    p = P(target_type='in-air')
    p.headless_ = True
    env = S(p)
    if args.rotation_only:
        p.target_p_ = env.data_.qpos[0:3].copy()
    vt = VirtualTactile(object_names=tuple(p.object_names_))
    vt._bind_model(env.model_)
    names = discover_fingertip_geoms(env.model_)
    reflex = TactileReflex(p.target_p_, p.target_q_, names,
                           k_translate=args.k_translate, k_rotate=args.k_rotate,
                           k_load=args.k_load, load_ref=args.load_ref, max_cmd=args.max_cmd)

    consec = 0
    success = False
    hist = []
    for k in range(args.max_steps):
        out = vt.observe(env)
        loads = out['tactile'][:, 13]
        cmd = reflex.act(env, loads)
        env.step(cmd)
        q = env.get_state()
        qe = float(metrics.comp_quat_error(q[3:7], p.target_q_))
        pe = float(metrics.comp_pos_error(q[0:3], p.target_p_))
        ok = (qe < CFG['quat_th']) and (pe < CFG['pos_th'])
        consec = consec + 1 if ok else 0
        hist.append(dict(k=k, qe=qe, pe=pe, z=float(q[2]),
                         loads=np.round(loads, 4).tolist(),
                         n_loads=int((loads > 1e-9).sum())))
        if k % args.verbose == 0 or ok:
            print(f'k={k:>4} qe={qe:.5f} pe={pe:.4f} z={q[2]:+.4f} '
                  f'loads={np.round(loads,3)} consec={consec}')
        if consec > CFG['hold']:
            success = True
            break

    q = env.get_state()
    qe = float(metrics.comp_quat_error(q[3:7], p.target_q_))
    pe = float(metrics.comp_pos_error(q[0:3], p.target_p_))
    print(f'\nRESULT: {"SUCCESS" if success else "FAIL"} steps={k+1} '
          f'quat_err={qe:.5f} (th {CFG["quat_th"]}) pos_err={pe:.4f} (th {CFG["pos_th"]}) '
          f'z={q[2]:+.4f}')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(
            obj=args.obj, success=bool(success), steps=k + 1, quat_err=qe, pos_err=pe,
            params=vars(args), history=hist), indent=2) + '\n')
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()

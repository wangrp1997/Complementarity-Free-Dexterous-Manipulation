"""FREE 预热 W 步（把物体举离桌面）后，把控制权交给触觉反射，看姿态误差是否下降。

只回答一个问题：物体已经在空中之后，纯触觉驱动的切向推动能不能把它转向目标。
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from utils import metrics
from planning.mpc_explicit import MPCExplicit
from ours.tactile_observation import VirtualTactile, discover_fingertip_geoms
from examples.mpc.ours.tactile_reflex import TactileReflex

CFG = dict(pos_th=0.02, quat_th=0.015, hold=20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', default='cube')
    ap.add_argument('--warmup', type=int, default=150)
    ap.add_argument('--max-steps', type=int, default=600)
    ap.add_argument('--k-rotate', type=float, default=3.0)
    ap.add_argument('--load-ref', type=float, default=0.35)
    ap.add_argument('--verbose', type=int, default=25)
    ap.add_argument('--output', default='')
    args = ap.parse_args()

    P = importlib.import_module(f'examples.mpc.fingertips.{args.obj}.params').ExplicitMPCParams
    S = importlib.import_module('envs.fingertips_env').MjSimulator
    C = importlib.import_module('contact.fingertips_collision_detection').Contact
    p = P(target_type='in-air'); p.headless_ = True
    contact = C(p); env = S(p); mpc = MPCExplicit(p)
    vt = VirtualTactile(object_names=tuple(p.object_names_)); vt._bind_model(env.model_)
    reflex = TactileReflex(p.target_p_, p.target_q_, discover_fingertip_geoms(env.model_),
                           k_rotate=args.k_rotate, load_ref=args.load_ref)

    hist = []
    for k in range(args.max_steps):
        curr_q = env.get_state()
        qe = float(metrics.comp_quat_error(curr_q[3:7], p.target_q_))
        pe = float(metrics.comp_pos_error(curr_q[0:3], p.target_p_))
        phase = 'FREE' if k < args.warmup else 'reflex'
        hist.append(dict(k=k, phase=phase, qe=qe, pe=pe, z=float(curr_q[2])))
        if k % args.verbose == 0:
            print(f'k={k:>4} {phase:>6} qe={qe:.5f} pe={pe:.4f} z={curr_q[2]:+.4f}')
        if k < args.warmup:
            phi_vec, jac_mat = contact.detect_once(env)
            sol = mpc.plan_once(p.target_p_, p.target_q_, curr_q, phi_vec, jac_mat,
                                sol_guess=p.sol_guess_)
            p.sol_guess_ = sol['sol_guess']
            env.step(sol['action'])
        else:
            out = vt.observe(env)
            env.step(reflex.act(env, out['tactile'][:, 13]))

    q = env.get_state()
    qe = float(metrics.comp_quat_error(q[3:7], p.target_q_))
    pe = float(metrics.comp_pos_error(q[0:3], p.target_p_))
    w = [h for h in hist if h['k'] == args.warmup - 1]
    w = w[0] if w else hist[0]
    print(f'\n接管时刻 (k={args.warmup-1}): qe={w["qe"]:.5f} pe={w["pe"]:.4f} z={w["z"]:+.4f}')
    print(f'结束     (k={args.max_steps-1}): qe={qe:.5f} pe={pe:.4f} z={q[2]:+.4f}')
    print(f'姿态误差变化: {qe - w["qe"]:+.5f}  ({"改善" if qe < w["qe"] else "变差"})')
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(dict(
            obj=args.obj, warmup=args.warmup, params=vars(args),
            qe_at_handover=w['qe'], qe_final=qe, pe_final=pe, history=hist), indent=2) + '\n')


if __name__ == '__main__':
    main()

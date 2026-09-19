import argparse
import importlib
import time

from planning.mpc_explicit import MPCExplicit
from utils import metrics
from utils.plant_mismatch import UNKNOWN_COM, UNKNOWN_FRICTION, UNKNOWN_INERTIA_SCALE

HAND = {
    'allegro': dict(
        env='envs.allegro_env',
        contact='contact.allegro_collision_detection',
        success='quat',
        pos_th=0.02,
        quat_th=0.04,
        max_steps=500,
        target_type='rotation',
    ),
    'trifinger': dict(
        env='envs.trifinger_env',
        contact='contact.trifinger_collision_detection',
        success='pose',
        pos_th=0.02,
        quat_th=0.04,
        max_steps=500,
        target_type='rotation',
    ),
    'fingertips': dict(
        env='envs.fingertips_env',
        contact='contact.fingertips_collision_detection',
        success='pose',
        pos_th=0.02,
        quat_th=0.015,
        max_steps=2000,
        target_type='in-air',
    ),
}


def parse_args(default_hand, default_obj, default_target):
    parser = argparse.ArgumentParser()
    parser.add_argument('--unknown-dyn', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--hand', default=default_hand)
    parser.add_argument('--obj', default=default_obj)
    parser.add_argument('--target-type', default=default_target)
    return parser.parse_args()


def _is_success(hand_cfg, curr_q, param):
    quat_ok = metrics.comp_quat_error(curr_q[3:7], param.target_q_) < hand_cfg['quat_th']
    if hand_cfg['success'] == 'quat':
        return quat_ok
    pos_ok = metrics.comp_pos_error(curr_q[0:3], param.target_p_) < hand_cfg['pos_th']
    return pos_ok and quat_ok


def run_eval(hand, obj, target_type=None):
    cfg = HAND[hand]
    if target_type is None:
        target_type = cfg['target_type']
    args = parse_args(hand, obj, target_type)
    hand, obj, target_type = args.hand, args.obj, args.target_type
    cfg = HAND[hand]

    params_cls = importlib.import_module(f'examples.mpc.{hand}.{obj}.params').ExplicitMPCParams
    env_cls = importlib.import_module(cfg['env']).MjSimulator
    contact_cls = importlib.import_module(cfg['contact']).Contact

    tag = f'{hand}/{obj}'
    if hand == 'fingertips':
        tag = f'{tag}/{target_type}'
    mode_name = 'unknown-dyn' if args.unknown_dyn else 'original'
    print(f'[{tag}] mode={mode_name}  trials={args.trials}  headless={args.headless}')
    if args.unknown_dyn:
        print(f'  plant COM offset (m)     = {UNKNOWN_COM}')
        print(f'  plant inertia scale      = {UNKNOWN_INERTIA_SCALE}')
        print(f'  plant sliding friction   = {UNKNOWN_FRICTION}')
        print('  mass unchanged; MPC Q/mu unchanged')

    n_success = 0
    for trial_count in range(args.trials):
        param = params_cls(rand_seed=trial_count, target_type=target_type)
        param.headless_ = args.headless
        contact = contact_cls(param)
        env = env_cls(param)
        if args.unknown_dyn:
            env.apply_object_plant_mismatch(UNKNOWN_COM, UNKNOWN_INERTIA_SCALE, UNKNOWN_FRICTION)
        mpc = MPCExplicit(param)

        rollout_step = 0
        consecutive_success_time = 0
        while rollout_step < cfg['max_steps']:
            if env.dyn_paused_:
                continue
            curr_q = env.get_state()
            phi_vec, jac_mat = contact.detect_once(env)
            sol = mpc.plan_once(
                param.target_p_, param.target_q_, curr_q, phi_vec, jac_mat,
                sol_guess=param.sol_guess_)
            param.sol_guess_ = sol['sol_guess']
            env.step(sol['action'])
            rollout_step += 1
            curr_q = env.get_state()
            if _is_success(cfg, curr_q, param):
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0
            if consecutive_success_time > 20:
                break

        success = rollout_step < cfg['max_steps']
        if success:
            n_success += 1
        final_q = env.get_state()
        quat_err = metrics.comp_quat_error(final_q[3:7], param.target_q_)
        print(f'  trial {trial_count:02d}: {"SUCCESS" if success else "FAIL":7s}  '
              f'steps={rollout_step:3d}  quat_err={quat_err:.4f}  z={final_q[2]:.3f}  '
              f'rate={n_success}/{trial_count + 1}')
        if getattr(env, 'viewer_', None) is not None:
            env.viewer_.close()
            time.sleep(0.5)

    print(f'[{tag}] {mode_name} success {n_success}/{args.trials} = '
          f'{100.0 * n_success / args.trials:.1f}%')
    return n_success, args.trials

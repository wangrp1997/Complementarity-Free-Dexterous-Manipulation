import argparse
import time

import numpy as np

from examples.mpc.allegro.cube.params import ExplicitMPCParams
from planning.mpc_explicit import MPCExplicit
from envs.allegro_env import MjSimulator
from contact.allegro_collision_detection import Contact

from utils import metrics


# Plant-only mismatch. Controller (params.py) still uses mu=0.5 and the original Q.
UNKNOWN_COM = np.array([0.008, -0.006, 0.004])
UNKNOWN_INERTIA_SCALE = np.array([2.5, 4.0, 1.5])
UNKNOWN_FRICTION = 0.2


def parse_args():
    parser = argparse.ArgumentParser(description='Allegro cube MPC demo')
    parser.add_argument('--unknown-dyn', action='store_true',
                        help='mismatch plant COM / inertia / friction; keep mass and MPC model')
    parser.add_argument('--headless', action='store_true',
                        help='no MuJoCo viewer, for batch success-rate runs')
    parser.add_argument('--trials', type=int, default=20)
    return parser.parse_args()


# -------------------------------
#       loop trials
# -------------------------------
args = parse_args()
save_flag = False
if save_flag:
    save_dir = './examples/mpc/allegro/cube/save/'
    prefix_data_name = 'ours_'
    save_data = dict()

trial_num = args.trials
success_pos_threshold = 0.02
success_quat_threshold = 0.04
consecutive_success_time_threshold = 20
max_rollout_length = 500

mode_name = 'unknown-dyn' if args.unknown_dyn else 'original'
print(f'[allegro/cube] mode={mode_name}  trials={trial_num}  headless={args.headless}')
if args.unknown_dyn:
    print(f'  plant COM offset (m)     = {UNKNOWN_COM}')
    print(f'  plant inertia scale      = {UNKNOWN_INERTIA_SCALE}')
    print(f'  plant sliding friction   = {UNKNOWN_FRICTION}')
    print('  mass unchanged; MPC Q/mu unchanged')

trial_count = 0
n_success = 0
while trial_count < trial_num:

    # -------------------------------
    #        init parameters
    # -------------------------------
    param = ExplicitMPCParams(rand_seed=trial_count, target_type='rotation')
    param.headless_ = args.headless

    # -------------------------------
    #        init contact
    # -------------------------------
    contact = Contact(param)

    # -------------------------------
    #        init envs
    # -------------------------------
    env = MjSimulator(param)
    if args.unknown_dyn:
        env.apply_object_plant_mismatch(UNKNOWN_COM, UNKNOWN_INERTIA_SCALE, UNKNOWN_FRICTION)

    # -------------------------------
    #        init planner
    # -------------------------------
    mpc = MPCExplicit(param)

    # -------------------------------
    #        MPC rollout
    # -------------------------------
    rollout_step = 0
    consecutive_success_time = 0

    rollout_q_traj = []
    while rollout_step < max_rollout_length:
        if not env.dyn_paused_:
            # get state
            curr_q = env.get_state()
            rollout_q_traj.append(curr_q)

            # -----------------------
            #     contact detect
            # -----------------------
            phi_vec, jac_mat = contact.detect_once(env)

            # -----------------------
            #        planning
            # -----------------------
            sol = mpc.plan_once(
                param.target_p_,
                param.target_q_,
                curr_q,
                phi_vec,
                jac_mat,
                sol_guess=param.sol_guess_)
            param.sol_guess_ = sol['sol_guess']
            action = sol['action']

            # -----------------------
            #        simulate
            # -----------------------
            env.step(action)
            rollout_step = rollout_step + 1

            # -----------------------
            #        success check
            # -----------------------
            curr_q = env.get_state()
            if (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold):
                consecutive_success_time = consecutive_success_time + 1
            else:
                consecutive_success_time = 0

            # -----------------------
            #       early termination
            # -----------------------
            if consecutive_success_time > consecutive_success_time_threshold:
                break

    success = rollout_step < max_rollout_length
    if success:
        n_success += 1
    final_q = env.get_state()
    quat_err = metrics.comp_quat_error(final_q[3:7], param.target_q_)
    print(f'  trial {trial_count:02d}: {"SUCCESS" if success else "FAIL":7s}  '
          f'steps={rollout_step:3d}  quat_err={quat_err:.4f}  z={final_q[2]:.3f}  '
          f'rate={n_success}/{trial_count + 1}')

    # -------------------------------
    #        close viewer
    # -------------------------------
    if env.viewer_ is not None:
        env.viewer_.close()
        time.sleep(0.5)

    # -------------------------------
    #        save data
    # -------------------------------
    if save_flag:
        # save
        save_data.update(target_obj_pos=param.target_p_)
        save_data.update(target_obj_quat=param.target_q_)
        save_data.update(rollout_traj=np.array(rollout_q_traj))
        # success index
        if rollout_step < max_rollout_length:
            save_data.update(success=True)
        else:
            save_data.update(success=False)
        # save to file
        metrics.save_data(save_data, data_name=prefix_data_name + 'trial_' + str(trial_count) + '_rollout',
                          save_dir=save_dir)

    trial_count = trial_count + 1

print(f'[allegro/cube] {mode_name} success {n_success}/{trial_num} = {100.0 * n_success / trial_num:.1f}%')

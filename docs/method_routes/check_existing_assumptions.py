"""Read existing observations and inspect conditional method assumptions.

Run from repository root:
python -m docs.method_routes.check_existing_assumptions --output /tmp/method_assumptions.json
No simulation step, dynamics fitting, contact truth, or physical parameters
are used. MuJoCo is used only to evaluate known rigid-body kinematics.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from examples.mpc.ours.analyze_tactile_probes import rotation_vector


def skew(point):
    x, y, z = point
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def geometry(points):
    matrix = np.vstack([np.hstack((np.eye(3), -skew(point / 0.05))) for point in points])
    _, values, vectors = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.count_nonzero(values > 1e-8))
    return matrix, vectors[rank:].T, {"rank_at_1e-8": rank, "singular_values": values.tolist()}


def analyze(root):
    rows = []
    for folder in sorted(root.glob('allegro*')):
        meta = json.loads((folder / 'metadata.json').read_text())
        model = mujoco.MjModel.from_xml_path(meta['model_path'])
        before_data, after_data = mujoco.MjData(model), mujoco.MjData(model)
        with np.load(folder / 'inputs.npz', allow_pickle=False) as data:
            for index in np.flatnonzero((data['pair_id'] >= 0) & (data['probe_sign'] == 0)):
                before, after = data['state'][index], data['next_state'][index]
                for mjdata, state in ((before_data, before), (after_data, after)):
                    mjdata.qpos[:16] = state[7:]
                    mjdata.qpos[-7:] = state[:7]
                    mujoco.mj_kinematics(model, mjdata)
                obj = model.body('obj').id
                rotation_before = before_data.xmat[obj].reshape(3, 3)
                rotation_after = after_data.xmat[obj].reshape(3, 3)
                offsets = data['contact_records_offsets']
                contacts = data['contact_records'][offsets[index]:offsets[index + 1]]
                residuals = []
                for record in contacts:
                    site = model.site(f'ftp_{int(record[0])}').id
                    point = before_data.xpos[obj] + rotation_before @ record[1:4]
                    point_on_finger = before_data.site_xmat[site].reshape(3, 3).T @ (point - before_data.site_xpos[site])
                    finger_after = after_data.site_xpos[site] + after_data.site_xmat[site].reshape(3, 3) @ point_on_finger
                    object_after = after_data.xpos[obj] + rotation_after @ record[1:4]
                    residuals.append(float(np.linalg.norm(finger_after - object_after)))
                tactile = data['tactile'][index]
                centers = tactile[tactile[:, 0] > 0, 1:4]
                _, _, point_geometry = geometry(contacts[:, 1:4])
                _, nullspace, center_geometry = geometry(centers)
                error = rotation_vector(before[3:7], data['target_quaternion'])
                gradient = np.r_[np.zeros(3), -error]
                missing = nullspace @ (nullspace.T @ gradient)
                action = data['action'][index]
                actual = after[7:] - before[7:]
                rows.append({
                    'episode': folder.name,
                    'pair_id': int(data['pair_id'][index]),
                    'source_step': int(data['source_step'][index]),
                    'active_fingers': np.flatnonzero(tactile[:, 0]).tolist(),
                    'contact_point_count': len(contacts),
                    'all_point_sticking_geometry': point_geometry,
                    'one_center_per_finger_geometry': center_geometry,
                    'command_norm_rad': float(np.linalg.norm(action)),
                    'actual_joint_increment_norm_rad': float(np.linalg.norm(actual)),
                    'tracking_error_norm_rad': float(np.linalg.norm(actual - action)),
                    'relative_tracking_error': float(np.linalg.norm(actual - action) / np.linalg.norm(action)),
                    'frozen_contact_material_displacement_m': residuals,
                    'max_frozen_contact_material_displacement_m': max(residuals),
                    'target_rotation_error_rad': float(np.linalg.norm(error)),
                    'finger_center_target_gradient_nullspace_fraction': float(np.linalg.norm(missing) / np.linalg.norm(gradient)) if np.linalg.norm(gradient) else 0.,
                })
    return {
        'purpose': 'Read-only tests of candidate method assumptions, not fitted response models',
        'source_dataset': str(root),
        'length_scale_for_geometric_svd_m': 0.05,
        'geometry_assumption': 'Each counted point sticks in 3D; rank is conditional geometry, not evidence of sticking or force closure',
        'residual_definition': 'Transport each initial contact point as a material point on the hand and on the object using only observed q/pose and known kinematics; compare at next control boundary',
        'residual_limit': 'Includes rolling, detachment, compliance, actual slip, and discrete contact association changes; not a direct slip sensor or proof of failure at every physics step',
        'physical_parameters_used': False,
        'target_ambiguity_definition': 'Illustrative projection of the rotation-objective gradient onto the nullspace of hypothetical sticking constraints at per-finger weighted centers. Centers need not be physical contact points. This is not a measured physical uncertainty.',
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('docs/data/tactile_probe_v2_review_20260920'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.dataset)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'snapshots': len(result['rows']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()

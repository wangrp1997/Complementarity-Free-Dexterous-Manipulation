"""Read-only audit of recorded sensing, not a simulator rollout or state proof.

Run from repository root with the free environment. Output must not exist.
Labels are used only to audit omitted contacts/velocities/control history.
No learned model is fitted; no simulator integration or mj_forward is called.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def stats(values):
    x = np.asarray(values, dtype=float)
    return None if not len(x) else dict(n=len(x), minimum=float(x.min()), median=float(np.median(x)),
                                      p90=float(np.quantile(x, .9)), maximum=float(x.max()))


def ragged(data, field, row):
    offsets = data[field + '_offsets']
    return data[field][offsets[row]:offsets[row + 1]]


def group_summary(rows):
    counts = Counter()
    loads = Counter()
    presence = Counter()
    for row in rows:
        counts.update(row['contacts_by_category'])
        loads.update(row['normal_load_by_category_N'])
        presence.update(k for k, v in row['contacts_by_category'].items() if v)
    total_load = sum(loads.values())
    total_count = sum(counts.values())
    active = Counter(row['active_fingertips'] for row in rows)
    return {
        'baseline_start_frames': len(rows),
        'active_fingertip_histogram': {str(i): active[i] for i in range(5)},
        'no_fingertip_frames': active[0], 'single_fingertip_frames': active[1],
        'multi_fingertip_frames': sum(active[i] for i in range(2, 5)),
        'contact_count_by_category': dict(counts), 'category_present_frames': dict(presence),
        'summed_normal_load_by_category_N': dict(loads),
        'fingertip_contact_count_fraction': counts['fingertip'] / total_count if total_count else None,
        'fingertip_summed_normal_load_fraction': loads['fingertip'] / total_load if total_load else None,
        'any_uncovered_contact_frames': sum(r['uncovered_contacts'] > 0 for r in rows),
        'loaded_object_with_no_fingertip_frames': sum(r['active_fingertips'] == 0 and r['total_contacts'] > 0 for r in rows),
        'fingertips_only_nonempty_frames': sum(r['uncovered_contacts'] == 0 and r['total_contacts'] > 0 for r in rows),
        'fingertips_only_multi_frames': sum(r['uncovered_contacts'] == 0 and r['active_fingertips'] >= 2 for r in rows),
        'fingertip_mask_changed_by_end_frames': sum(r['mask_changed_at_end'] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('docs/data/tactile_probe_v2_review_20260920'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    all_rows, episodes, finger_rows, history_rows, matching, hashes = [], [], [], [], [], {}
    fingerprint_frames = []
    threshold_rows = {threshold: [] for threshold in (1e-8, 1e-4, 1e-2, 1e-1)}
    for folder in sorted(args.dataset.glob('allegro*')):
        for path in (folder / 'inputs.npz', folder / 'labels.npz', folder / 'metadata.json'):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = json.loads((folder / 'metadata.json').read_text())
        model = mujoco.MjModel.from_xml_path(metadata['model_path'])
        object_geoms = {model.geom(name).id for name in metadata['tactile_schema']['object_geoms']}
        fingertip_geoms = {model.geom(f'fingertip{i}').id for i in range(4)}
        categories, names = {}, {}
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f'geom_{geom}'
            names[geom] = name
            categories[geom] = ('fingertip' if geom in fingertip_geoms else
                                'ground' if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE else
                                'palm' if name == 'palm' else
                                'finger_non_tip' if name.startswith(('ff_', 'mf_', 'rf_', 'th_')) else 'other')
        with np.load(folder / 'inputs.npz', allow_pickle=False) as inp, np.load(folder / 'labels.npz', allow_pickle=False) as lab:
            baseline = np.flatnonzero(inp['probe_sign'] == 0)
            baseline = baseline[np.argsort(inp['source_step'][baseline])]
            assert len(np.unique(inp['source_step'][baseline])) == len(baseline)
            episode_rows = []
            previous = None
            for index in baseline:
                source_step = int(inp['source_step'][index])
                records = ragged(inp, 'contact_records', index)
                truth = ragged(lab, 'contact_truth_records', index)
                loaded = truth[truth[:, 3] > 1e-8]
                for threshold in threshold_rows:
                    selected_truth = truth[truth[:, 3] > threshold]
                    touched_tips, uncovered = set(), 0
                    for contact in selected_truth:
                        geom1, geom2 = map(int, contact[:2])
                        other = geom2 if geom1 in object_geoms else geom1
                        if other in fingertip_geoms:
                            touched_tips.add(other)
                        else:
                            uncovered += 1
                    threshold_rows[threshold].append(dict(active_tips=len(touched_tips), uncovered=uncovered))
                count, load, geom_count = Counter(), Counter(), Counter()
                for contact in loaded:
                    geom1, geom2 = map(int, contact[:2])
                    assert (geom1 in object_geoms) != (geom2 in object_geoms)
                    other = geom2 if geom1 in object_geoms else geom1
                    count[categories[other]] += 1
                    load[categories[other]] += float(contact[3])
                    geom_count[names[other]] += 1
                tactile = inp['tactile'][index]
                active = int(np.count_nonzero(tactile[:, 0]))
                assert count['fingertip'] == len(records) == int(lab['n_fingertip_contacts'][index])
                assert len(loaded) == int(lab['n_object_contacts'][index])
                assert np.isclose(load['fingertip'], tactile[:, 13].sum(), rtol=1e-10, atol=1e-10)
                row = dict(episode=folder.name, source_step=source_step, active_fingertips=active,
                           contacts_by_category=dict(count), normal_load_by_category_N=dict(load),
                           contact_count_by_geom=dict(geom_count), total_contacts=len(loaded),
                           uncovered_contacts=len(loaded) - len(records),
                           mask_changed_at_end=bool(np.any(tactile[:, 0] != inp['next_tactile'][index, :, 0])),
                           pair_id=int(inp['pair_id'][index]))
                all_rows.append(row)
                episode_rows.append(row)
                for finger in np.flatnonzero(tactile[:, 0]):
                    points = records[records[:, 0] == finger]
                    force = points[:, 7:10].sum(axis=0)
                    torque = (np.cross(points[:, 1:4], points[:, 7:10]) + points[:, 10:13]).sum(axis=0)
                    pair_dist = np.linalg.norm(points[:, None, 1:4] - points[None, :, 1:4], axis=-1)
                    finger_rows.append(dict(episode=folder.name, source_step=source_step, finger=int(finger),
                                            points=len(points), point_span_m=float(pair_dist.max()),
                                            strongest_point_normal_load_fraction=float(points[:, 13].max() / points[:, 13].sum()),
                                            normal_resultant_length=float(np.linalg.norm(tactile[finger, 4:7])),
                                            force_aggregate_error_N=float(np.linalg.norm(force - tactile[finger, 7:10])),
                                            torque_aggregate_error_Nm=float(np.linalg.norm(torque - tactile[finger, 10:13])),
                                            torque_not_in_centroid_cross_force_Nm=float(np.linalg.norm(torque - np.cross(tactile[finger, 1:4], force)))))
                fingerprint_frames.append(dict(episode=folder.name, source_step=source_step,
                                               object=metadata['object'], mode=metadata['mode'],
                                               state=inp['state'][index].copy(), tactile=tactile.copy(),
                                               records=records.copy(), joint_velocity=inp['joint_velocity'][index].copy(),
                                               action=inp['action'][index].copy(), next_state=inp['next_state'][index].copy(),
                                               object_velocity=lab['object_velocity_mujoco'][index].copy()))
                if previous is not None:
                    assert source_step == int(inp['source_step'][previous]) + 1
                    assert np.array_equal(inp['state'][index], inp['next_state'][previous])
                    assert np.array_equal(inp['joint_velocity'][index], inp['next_joint_velocity'][previous])
                    dt = float(inp['time'][index] - inp['time'][previous])
                    qpos_before = np.r_[inp['state'][previous, 7:], inp['state'][previous, :7]]
                    qpos_after = np.r_[inp['state'][index, 7:], inp['state'][index, :7]]
                    averaged_velocity = np.empty(model.nv)
                    mujoco.mj_differentiatePos(model, averaged_velocity, dt, qpos_before, qpos_after)
                    instant = lab['object_velocity_mujoco'][index]
                    ctrl_field = next(f for f in metadata['labels']['integration_state_fields'] if f['name'] == 'mjSTATE_CTRL')
                    ctrl = lab['integration_state'][index, ctrl_field['offset']:ctrl_field['offset'] + ctrl_field['size']]
                    reconstructed_target = inp['state'][previous, 7:] + inp['action'][previous]
                    history_rows.append(dict(episode=folder.name, source_step=source_step, dt=dt,
                                             object_translation_speed_m_s=float(np.linalg.norm(instant[:3])),
                                             object_rotation_speed_rad_s=float(np.linalg.norm(instant[3:])),
                                             backward_average_minus_instant_translation_m_s=float(np.linalg.norm(averaged_velocity[-6:-3] - instant[:3])),
                                             backward_average_minus_instant_rotation_rad_s=float(np.linalg.norm(averaged_velocity[-3:] - instant[3:])),
                                             prior_target_reconstruction_error_rad=float(np.linalg.norm(ctrl - reconstructed_target))))
                previous = index
            episodes.append(dict(episode=folder.name, object=metadata['object'], mode=metadata['mode'],
                                 rows=len(inp['action']), paired_snapshots=metadata['paired_snapshots'],
                                 activation_state_dimension=int(model.na), summary=group_summary(episode_rows)))
    # Exact/tight repeated current observations cannot establish a conditional law;
    # this search merely reports whether this tiny dataset contains comparison cases.
    for left_index, left in enumerate(fingerprint_frames):
        for right in fingerprint_frames[left_index + 1:]:
            if left['object'] != right['object'] or left['records'].shape != right['records'].shape:
                continue
            keys = ('state', 'tactile', 'joint_velocity', 'records')
            if not all(np.allclose(left[k], right[k], rtol=0, atol=1e-10) for k in keys):
                continue
            matching.append(dict(left=f"{left['episode']}:{left['source_step']}",
                                 right=f"{right['episode']}:{right['source_step']}",
                                 max_action_difference_rad=float(np.max(np.abs(left['action'] - right['action']))),
                                 same_action_at_1e_10=bool(np.allclose(left['action'], right['action'], rtol=0, atol=1e-10)),
                                 hidden_object_velocity_difference=float(np.linalg.norm(left['object_velocity'] - right['object_velocity'])),
                                 next_object_translation_difference_m=float(np.linalg.norm(left['next_state'][:3] - right['next_state'][:3])),
                                 next_quaternion_geodesic_difference_rad=float(2 * np.arccos(np.clip(abs(np.dot(left['next_state'][3:7], right['next_state'][3:7])), 0, 1)))))
    geom_counts = Counter()
    for row in all_rows:
        geom_counts.update(row['contact_count_by_geom'])
    result = {
        'purpose': 'Finite-sample necessity/coverage diagnostics, NOT Markov sufficiency, controller or learned model',
        'dataset': str(args.dataset), 'baseline_rule': 'probe_sign == 0; unique source_step within each episode; only action-start frames',
        'loaded_threshold_N': 1e-8,
        'normal_load_warning': 'Sum of positive contact-normal magnitudes across different directions/times; NOT net force, support fraction, force closure or stability',
        'truth_access': 'labels: contact geom IDs/normal magnitudes, endpoint object velocity, integration ctrl; metadata: geometry names/layout/model path. Evaluation only, never ours input.',
        'no_simulation_rollout': True, 'no_training': True,
        'overall': group_summary(all_rows), 'episodes': episodes,
        'per_point_normal_threshold_sensitivity': {
            str(threshold): dict(no_tip_frames=sum(r['active_tips'] == 0 for r in rows),
                                 single_tip_frames=sum(r['active_tips'] == 1 for r in rows),
                                 multi_tip_frames=sum(r['active_tips'] >= 2 for r in rows),
                                 any_uncovered_contact_frames=sum(r['uncovered'] > 0 for r in rows))
            for threshold, rows in threshold_rows.items()
        },
        'loaded_contact_count_by_geom': dict(geom_counts),
        'finger_summary': {
            'active_finger_observations': len(finger_rows),
            'multiple_point_finger_observations': sum(row['points'] > 1 for row in finger_rows),
            'point_count_histogram': dict(sorted(Counter(row['points'] for row in finger_rows).items())),
            **{key: stats([row[key] for row in finger_rows]) for key in (
                'point_span_m', 'strongest_point_normal_load_fraction', 'normal_resultant_length',
                'force_aggregate_error_N', 'torque_aggregate_error_Nm', 'torque_not_in_centroid_cross_force_Nm')},
        },
        'history': {
            'consecutive_baseline_pairs': len(history_rows),
            **{key: stats([row[key] for row in history_rows]) for key in (
                'object_translation_speed_m_s', 'object_rotation_speed_rad_s',
                'backward_average_minus_instant_translation_m_s', 'backward_average_minus_instant_rotation_rad_s',
                'prior_target_reconstruction_error_rad')},
            'velocity_difference_warning': 'Backward 0.1 s pose secant differs from instantaneous endpoint truth; no estimated velocity ground-truth error claim, no proof history suffices.',
        },
        'tight_current_observation_matches_at_1e_10': matching,
        'matching_warning': 'Includes cross-physics comparisons. No noise-based approximate-match threshold or sufficient repeated-action population is available; no fitted model or inference from absent matches.',
        'baseline_rows': all_rows, 'finger_rows': finger_rows, 'history_rows': history_rows,
        'source_sha256': hashes,
    }
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('overall', 'loaded_contact_count_by_geom', 'finger_summary', 'history', 'tight_current_observation_matches_at_1e_10')}, indent=2))


if __name__ == '__main__':
    main()

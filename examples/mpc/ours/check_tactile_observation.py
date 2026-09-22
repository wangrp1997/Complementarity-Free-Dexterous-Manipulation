"""Physical consistency and passive-observation checks for virtual tactile v2.

Run from the repository root:
  PYTHONPATH=. python examples/mpc/ours/check_tactile_observation.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import mujoco
import numpy as np

from ours.tactile_observation import VirtualTactile, flatten_contact_records


def _scene(object_first=False, support_name="fingertip0", clearance=-0.001):
    # Tilted support exposes contact-axis row/column mistakes. Object inertia
    # is offset from its body origin to expose accidental COM-frame torques.
    angle = 0.37
    quat = f"{np.cos(angle / 2)} 0 {np.sin(angle / 2)} 0"
    normal = np.array([np.sin(angle), 0, np.cos(angle)])
    center = (0.04 + clearance) * normal
    xyz = " ".join(map(str, center))
    support = f'<body name="support" quat="{quat}"><geom name="{support_name}" type="box" size=".1 .1 .05" pos="0 0 -.05"/>'
    if support_name == "fingertip0":
        support += '<site name="ftp_0"/>'
    support += '</body>'
    obj = f'<body name="object" pos="{xyz}" quat="{quat}"><freejoint/><inertial pos=".005 -.003 .007" mass=".2" diaginertia=".0003 .0004 .0005"/><geom name="obj" type="box" size=".04 .04 .04"/></body>'
    extra_indices = range(1, 4) if support_name == "fingertip0" else range(4)
    extras = ''.join(f'<body name="tip{i}" pos="{i + 1} 0 0"><geom name="fingertip{i}" size=".01"/><site name="ftp_{i}"/></body>' for i in extra_indices)
    xml = f'<mujoco><option cone="elliptic" jacobian="dense" tolerance="1e-12"/><default><geom condim="6" friction=".7 .1 .05" margin=".003" gap=".002"/></default><worldbody>{obj + support if object_first else support + obj}{extras}</worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qvel[:] = [.02, -.03, 0, .02, .01, .04]
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model_=model, data_=data)


def _physical_checks():
    results = []
    observations = []
    for object_first in (False, True):
        env = _scene(object_first=object_first)
        sensor = VirtualTactile()
        observation = sensor.observe(env)
        model = env.model_
        # Independently recreate the observer's evaluation state.
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, env.data_)
        mujoco.mj_forward(model, data)
        body = model.body("object").id
        object_geom = model.geom("obj").id
        rotation = data.xmat[body].reshape(3, 3)
        origin = data.xpos[body]
        contact_ids = [i for i, c in enumerate(data.contact) if c.efc_address >= 0]
        assert len(observation["contact_records"]) == len(contact_ids) == 4
        assert all((data.contact[i].geom1 == object_geom) == object_first for i in contact_ids)
        predicted_generalized = np.zeros(model.nv)
        force_sum = np.zeros(3)
        moment_sum = np.zeros(3)
        for record in observation["contact_records"]:
            force = record["force_object"]
            torque = record["torque_at_contact_object"]
            position = record["position_object"]
            np.testing.assert_allclose(np.dot(force, record["normal_object"]), record["normal_force"], atol=1e-12)
            np.testing.assert_allclose(np.linalg.norm(record["normal_object"]), 1, atol=1e-12)
            jac_position = np.zeros((3, model.nv))
            jac_rotation = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac_position, jac_rotation, origin + rotation @ position, body)
            predicted_generalized += jac_position.T @ (rotation @ force) + jac_rotation.T @ (rotation @ torque)
            force_sum += force
            moment_sum += np.cross(position, force) + torque
        # All constraints in this synthetic scene belong to the four contacts.
        constraint_jacobian = data.efc_J.reshape(data.nefc, model.nv)
        expected_generalized = constraint_jacobian.T @ data.efc_force
        np.testing.assert_allclose(predicted_generalized, expected_generalized, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(predicted_generalized, data.qfrc_constraint, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(observation["tactile"][0, 7:10], force_sum, atol=1e-12)
        np.testing.assert_allclose(observation["tactile"][0, 10:13], moment_sum, atol=1e-12)
        # Aggregate wrench at body origin must also map to the same generalized force.
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, origin, body)
        aggregate_generalized = jp.T @ (rotation @ force_sum) + jr.T @ (rotation @ moment_sum)
        np.testing.assert_allclose(aggregate_generalized, expected_generalized, rtol=1e-10, atol=1e-11)
        com_object = rotation.T @ (data.xipos[body] - origin)
        assert np.linalg.norm(np.cross(com_object, force_sum)) > 1e-4
        assert any(np.linalg.norm(r["torque_at_contact_object"]) > 1e-7 for r in observation["contact_records"])
        observations.append(observation["tactile"])
        results.append({
            "object_contact_slot": "geom1" if object_first else "geom2",
            "loaded_contacts_retained": len(contact_ids),
            "max_generalized_force_error": float(np.max(np.abs(predicted_generalized - expected_generalized))),
        })
    np.testing.assert_allclose(observations[0], observations[1], rtol=1e-9, atol=1e-10)
    return results


def _coverage_checks():
    results = {}
    for name in ("floor", "ff_distal", "palm"):
        observed = VirtualTactile().observe(_scene(support_name=name))
        assert not observed["contact_records"]
        assert not observed["tactile"].any()
        diagnostics = observed["diagnostics"]
        assert diagnostics["n_object_contacts"] == 4
        assert diagnostics["n_palm_contacts"] == (4 if name == "palm" else 0)
        assert diagnostics["n_other_object_contacts"] == (0 if name == "palm" else 4)
        results[name] = diagnostics
    near = _scene(clearance=0.0065)
    assert near.data_.ncon > 0
    assert all(c.efc_address < 0 for c in near.data_.contact)
    near_observation = VirtualTactile().observe(near)
    assert not near_observation["contact_records"]
    assert not near_observation["tactile"].any()
    assert flatten_contact_records([]).shape == (0, 14)
    results["inactive_near_contacts_excluded"] = int(near.data_.ncon)
    unloading = _scene()
    unloading.model_.opt.gravity[:] = 0
    unloading.data_.qvel[:] = 0
    unloading.data_.qvel[:3] = [np.sin(0.37), 0, np.cos(0.37)]
    mujoco.mj_forward(unloading.model_, unloading.data_)
    assert unloading.data_.nefc > 0
    unloaded = VirtualTactile().observe(unloading)
    assert not unloaded["contact_records"]
    assert not unloaded["tactile"].any()
    results["zero_load_contacts_excluded"] = int(unloading.data_.ncon)
    return results


def _integration_state(model, data):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    return state


def _passive_observation_checks():
    from envs.allegro_env import MjSimulator
    from examples.mpc.allegro.cube.params import ExplicitMPCParams

    params = ExplicitMPCParams(rand_seed=0, target_type="rotation")
    params.headless_ = True
    env = MjSimulator(params)
    model = env.model_
    reference = mujoco.MjData(model)
    mujoco.mj_copyData(reference, model, env.data_)
    sensor = VirtualTactile(params.object_names_)
    rng = np.random.default_rng(20260920)
    max_difference = 0.0
    observations = 0
    arrays = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "efc_force", "efc_J", "qfrc_constraint")
    for step in range(8):
        action = rng.uniform(-0.002, 0.002, 16)
        target = reference.qpos[:16].copy() + action
        for _ in range(params.frame_skip_):
            before = _integration_state(model, env.data_)
            array_before = {name: getattr(env.data_, name).copy() for name in arrays}
            sensor.observe(env)
            observations += 1
            np.testing.assert_array_equal(before, _integration_state(model, env.data_))
            for name, value in array_before.items():
                np.testing.assert_array_equal(value, getattr(env.data_, name))
            env.data_.ctrl[:] = target
            reference.ctrl[:] = target
            mujoco.mj_step(model, env.data_)
            mujoco.mj_step(model, reference)
            actual, expected = _integration_state(model, env.data_), _integration_state(model, reference)
            np.testing.assert_array_equal(actual, expected)
            max_difference = max(max_difference, float(np.max(np.abs(actual - expected))))
    env.close() if hasattr(env, "close") else None
    return {"control_steps": 8, "physics_steps": observations, "max_state_difference": max_difference}


def main():
    results = {
        "schema": VirtualTactile.schema_version,
        "physical_consistency": _physical_checks(),
        "coverage": _coverage_checks(),
        "passive_observation": _passive_observation_checks(),
        "status": "passed",
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

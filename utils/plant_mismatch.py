import mujoco
import numpy as np

UNKNOWN_COM = np.array([0.004, -0.003, 0.002])
UNKNOWN_INERTIA_SCALE = np.array([1.8, 2.0, 1.5])
UNKNOWN_FRICTION = 0.3


def apply_object_plant_mismatch(model, data, com_offset=None, inertia_scale=None, sliding_friction=None):
    """Shift only the MuJoCo object plant. MPC params stay unchanged."""
    if com_offset is None:
        com_offset = UNKNOWN_COM
    if inertia_scale is None:
        inertia_scale = UNKNOWN_INERTIA_SCALE
    if sliding_friction is None:
        sliding_friction = UNKNOWN_FRICTION

    body_id = model.body('obj').id
    model.body_simple[body_id] = 0
    model.body_sameframe[body_id] = 0
    model.body_ipos[body_id] = np.asarray(com_offset, dtype=float)
    model.body_inertia[body_id] = model.body_inertia[body_id] * np.asarray(inertia_scale, dtype=float)

    geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body_id]
    for geom_id in geom_ids:
        model.geom_friction[geom_id, 0] = float(sliding_friction)

    mujoco.mj_setConst(model, data)
    mujoco.mj_forward(model, data)
    return {
        'mass': float(model.body_mass[body_id]),
        'com': model.body_ipos[body_id].copy(),
        'inertia': model.body_inertia[body_id].copy(),
        'friction': float(sliding_friction),
    }

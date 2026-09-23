"""Allegro fingertip-grasp cache, adapted from Sharpa for this plant.

Sharpa's cylinder already sits in a designed in-hand pose. Ours are 17
FREE meshes, so we keep the idea (place, lock, flip gravity) and change
the parts that do not transfer:

1. AABB in the fingertip cage; elongated meshes lie across the finger row;
2. close a little, then lock ctrl = settled q (no leftover squeeze);
3. prefer four loaded tips; accept two or more; palm contact is rejected;
4. stick is pinched at pad height, not dropped onto the palm plate;
5. six-way gravity so a rest-on-the-fingers pose is rejected;
6. force / cage / tilt thresholds match 10 g Allegro meshes, not Sharpa's
   50 g cylinder.
"""

from __future__ import annotations

import pathlib

import numpy as np
import mujoco

from ours.tactile_observation import VirtualTactile

OBJECTS = [
    "airplane", "binoculars", "bowl", "bunny", "camera", "can", "cube", "cup",
    "elephant", "foambrick", "mug", "piggy_bank", "rubber_duck", "stick",
    "teapot", "torus", "water_bottle",
]

_REPO = pathlib.Path(__file__).resolve().parents[2]
XML_TEMPLATE = str(_REPO / "envs/xmls/env_allegro_{obj}.xml")
FINGERTIP_GEOMS = ("fingertip0", "fingertip1", "fingertip2", "fingertip3")
PALM_GEOM = "palm"
OBJECT_GEOM = "obj"
N_Q = 16

# Adducted cage above the palm. FREE's own REF pose is too spread / too low.
PREGRASP = np.array([
    -0.22, 1.08, 1.12, 1.08,
    0.00, 1.08, 1.12, 1.08,
    0.22, 1.08, 1.12, 1.08,
    1.08, 1.20, 0.98, 1.28,
])
FLEX_IDX = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
TIP_IDX = np.array([3, 7, 11, 15])
# More open than PREGRASP so the shaft sits in the pads, not the palm cup.
PINCH = np.array([
    -0.16, 0.82, 0.95, 1.18,
     0.00, 0.82, 0.95, 1.18,
     0.16, 0.82, 0.95, 1.18,
     1.00, 1.05, 0.92, 1.22,
])

PALM_TOP_Z = 0.012
PALM_CLEAR = 0.018
# Soft Allegro (kp=1). Cube/stick are 100 g; most other FREE meshes are 10 g.
FT_FORCE = 0.03
MIN_TACTILE_LOAD = 0.03
MAX_TACTILE_LOAD = 8.0
MIN_TIPS = 2
HOLD_STEPS = 400
SQUEEZE_STEPS = 250
SQUEEZE = 0.06
SQUEEZE_LONG = 0.18
JOINT_NOISE = 0.08
HOLLOW = frozenset({"bowl", "cup", "mug", "teapot"})
# Stick is ~13 cm × 3 cm. Align the shaft with the finger row and pinch
# it at pad height — a palm rest is not an in-hand grasp.
ELONGATED = frozenset({"stick"})
# 100 g objects: keep a small wrap command so the pinch does not go slack.
HEAVY = frozenset({"cube", "stick"})
MIN_Z_TIPS = 0.048
HOLD_BIAS = 0.04
MAX_LIN_VEL = 0.12
MAX_XY = 0.04
MAX_CAGE = 0.05
MAX_TILT = np.deg2rad(40.0)
MAX_CAGE_LONG = 0.08
MAX_TILT_LONG = np.deg2rad(55.0)
STEPS_PER_G = 160
SETTLE_STEPS = 2000  # ~4 s; a snapshot that later slides onto the palm is rejected.
GRAVITIES = (
    (0.0, 0.0, -9.81),
    (0.0, 0.0, 9.81),
    (0.0, 9.81, 0.0),
    (0.0, -9.81, 0.0),
    (9.81, 0.0, 0.0),
    (-9.81, 0.0, 0.0),
)


def load_model(obj: str) -> mujoco.MjModel:
    if obj not in OBJECTS:
        raise KeyError(obj)
    return mujoco.MjModel.from_xml_path(XML_TEMPLATE.format(obj=obj))


def _ids(model: mujoco.MjModel):
    return (
        [model.geom(name).id for name in FINGERTIP_GEOMS],
        model.geom(PALM_GEOM).id,
        model.geom(OBJECT_GEOM).id,
    )


def _clip_q(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    return np.clip(q, model.jnt_range[:N_Q, 0], model.jnt_range[:N_Q, 1])


def contact_report(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    ft_ids, palm_id, obj_id = _ids(model)
    ft = np.zeros(4)
    palm = 0.0
    other = 0.0
    loaded = set()
    for i in range(data.ncon):
        con = data.contact[i]
        if obj_id not in (con.geom1, con.geom2):
            continue
        other_g = con.geom2 if con.geom1 == obj_id else con.geom1
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        fn = abs(float(force[0]))
        if other_g == palm_id:
            palm += fn
            continue
        if fn > FT_FORCE:
            loaded.add(int(other_g))
        if other_g in ft_ids:
            ft[ft_ids.index(other_g)] += fn
        else:
            other += fn
    return {
        "fingertip_fn": ft,
        "n_fingertip": int((ft > FT_FORCE).sum()),
        "n_sensors": int(len(loaded)),
        "palm_fn": float(palm),
        "other_fn": float(other),
    }


def object_vertices(model: mujoco.MjModel) -> np.ndarray:
    """Object mesh in the free-joint body frame (geom quat + pos applied)."""
    gid = model.geom(OBJECT_GEOM).id
    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[gid]
        start, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        verts = model.mesh_vert[start:start + n].reshape(-1, 3)
    else:
        s = model.geom_size[gid][:3]
        verts = np.array(
            [[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])],
            dtype=float,
        )
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
    return verts @ rot.reshape(3, 3).T + model.geom_pos[gid]


def _rotmat_zy(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rz @ ry


def _align_long_axis(verts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map the mesh long axis onto the fingertip row (world +Y)."""
    ext = verts.max(axis=0) - verts.min(axis=0)
    long_i = int(np.argmax(ext))
    others = [i for i in range(3) if i != long_i]
    up_i = others[int(np.argmin(ext[others]))]
    sign = 1.0 if rng.random() < 0.5 else -1.0
    s_long = np.eye(3)[long_i]
    s_up = np.eye(3)[up_i]
    s_x = np.cross(s_long, s_up)
    s_x /= max(float(np.linalg.norm(s_x)), 1e-9)
    src = np.stack([s_long, s_up, s_x], axis=1)
    dst = np.stack([
        np.array([0.0, sign, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([sign, 0.0, 0.0]),
    ], axis=1)
    rot = dst @ src.T
    if np.linalg.det(rot) < 0.0:
        dst[:, 2] *= -1.0
        rot = dst @ src.T
    return rot


def _quat_from_rotmat(rm: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rm, dtype=float).reshape(-1))
    return quat


def sample_trial(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator,
                 obj: str | None = None):
    """Jitter the cage, drop the object into it, keep it above the palm."""
    seed_q = PINCH if obj in ELONGATED else PREGRASP
    q = _clip_q(model, seed_q + JOINT_NOISE * rng.normal(size=N_Q))
    data.qpos[:N_Q] = q
    data.qpos[16:19] = (0.0, 0.0, -1.0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    pads = np.array([data.site(f"ftp_{i}").xpos for i in range(4)], dtype=float)
    cage = pads.mean(axis=0)
    verts0 = object_vertices(model)
    hollow = obj in HOLLOW
    elongated = obj in ELONGATED
    if elongated:
        # Shaft across the four tips, not stabbed into the palm / standing up.
        yaw = float(rng.uniform(-0.25, 0.25))
        pitch = float(rng.uniform(-0.18, 0.18))
        rot = _rotmat_zy(yaw, pitch) @ _align_long_axis(verts0, rng)
    else:
        yaw = float(rng.uniform(-np.pi, np.pi))
        # Keep hollow walls more vertical so a fingertip pinches the rim/wall
        # instead of dropping a finger into the cavity.
        pitch = float(rng.uniform(-0.25, 0.25) if hollow else rng.uniform(-1.0, 1.0))
        rot = _rotmat_zy(yaw, pitch)
    quat = _quat_from_rotmat(rot)
    quat = quat / np.linalg.norm(quat)

    verts = verts0 @ rot.T
    # Sit the bounding box in the cage (object in the hand), not a random
    # outer vertex that throws the body origin out of reach.
    aabb = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
    pos = cage - aabb
    pos[:2] += rng.uniform(-0.003, 0.003, size=2)
    min_z = float((verts + pos)[:, 2].min())
    clear = 0.036 if elongated else PALM_CLEAR
    if min_z < PALM_TOP_Z + clear:
        pos[2] += (PALM_TOP_Z + clear) - min_z
    return q, pos, quat, yaw


def _stable(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    return bool(np.isfinite(data.qacc).all() and np.isfinite(data.qpos).all())


class _Plant:
    """Minimal adapter so VirtualTactile can read a raw MjModel/MjData pair."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model_ = model
        self.data_ = data


def tactile_report(sensor: VirtualTactile, model: mujoco.MjModel,
                   data: mujoco.MjData) -> dict:
    """Same observer the rest of the project uses; a 'reading' is a loaded tip."""
    out = sensor.observe(_Plant(model, data))
    loads = np.asarray(out["tactile"][:, 13], dtype=float)
    active = loads > MIN_TACTILE_LOAD
    return {
        "tactile_loads": loads,
        "n_tactile": int(active.sum()),
        "n_palm_contacts": int(out["diagnostics"]["n_palm_contacts"]),
    }


def _fail(note: str, z: float, **extra) -> dict:
    empty = {
        "n_fingertip": 0, "n_sensors": 0, "palm_fn": 0.0,
        "fingertip_fn": np.zeros(4), "other_fn": 0.0, "n_tactile": 0,
        "tactile_loads": np.zeros(4),
    }
    empty.update(extra)
    return {"ok": False, "note": note, "final_z": float(z), **empty}


def _physics_ok(model: mujoco.MjModel, data: mujoco.MjData,
                rep: dict, tac: dict, z: float, lin: float, xy: float,
                obj: str | None = None) -> bool:
    loads = np.asarray(tac.get("tactile_loads", [0.0]))
    min_z = MIN_Z_TIPS if obj in ELONGATED else (PALM_TOP_Z + 0.030)
    return (
        z > min_z
        and lin < MAX_LIN_VEL
        and xy < MAX_XY
        and rep["palm_fn"] < 1e-4
        and int(tac.get("n_palm_contacts", 0)) == 0
        and int(rep["n_fingertip"]) >= MIN_TIPS
        and float(np.max(loads)) < MAX_TACTILE_LOAD
    )


def _quat_angle(q: np.ndarray, p: np.ndarray) -> float:
    d = abs(float(np.dot(q, p)))
    return float(2.0 * np.arccos(min(1.0, d)))


def _hold_limits(obj: str | None) -> tuple[float, float]:
    if obj in ELONGATED:
        return MAX_CAGE_LONG, MAX_TILT_LONG
    return MAX_CAGE, MAX_TILT


def _escaped(data: mujoco.MjData, origin: np.ndarray, quat0: np.ndarray,
             palm_fn: float, max_cage: float = MAX_CAGE,
             max_tilt: float = MAX_TILT) -> bool:
    pos = np.asarray(data.qpos[16:19], dtype=float)
    return (
        float(np.linalg.norm(pos - origin)) > max_cage
        or _quat_angle(np.asarray(data.qpos[19:23], dtype=float), quat0) > max_tilt
        or palm_fn > 1e-4
        or not np.isfinite(data.qpos).all()
    )


def _gravity_hold(model: mujoco.MjModel, data: mujoco.MjData,
                  hold: np.ndarray, origin: np.ndarray, quat0: np.ndarray,
                  obj: str | None = None,
                  ctrl: np.ndarray | None = None) -> bool:
    """Six-way gravity. Resting on the fingers fails when g flips up."""
    max_cage, max_tilt = _hold_limits(obj)
    cmd = hold if ctrl is None else ctrl
    g0 = np.array(model.opt.gravity, dtype=float)
    try:
        for gravity in GRAVITIES:
            model.opt.gravity[:] = gravity
            data.qpos[:N_Q] = hold
            data.qpos[16:19] = origin
            data.qpos[19:23] = quat0
            data.qvel[:] = 0.0
            data.ctrl[:] = cmd
            mujoco.mj_forward(model, data)
            lost = 0
            for step in range(STEPS_PER_G):
                data.ctrl[:] = cmd
                mujoco.mj_step(model, data)
                if not _stable(model, data):
                    return False
                palm = 0.0
                if step % 8 == 7:
                    rep = contact_report(model, data)
                    palm = rep["palm_fn"]
                    if palm > 1e-4:
                        return False
                    no_finger = (
                        int(rep["n_fingertip"]) < 1 and rep["other_fn"] <= FT_FORCE
                    )
                    lost = lost + 1 if no_finger else 0
                    if lost >= 4:
                        return False
                if _escaped(data, origin, quat0, palm, max_cage, max_tilt):
                    return False
        return True
    finally:
        model.opt.gravity[:] = g0


def _grade(rep: dict, tac: dict) -> int:
    """Higher is better. Any palm contact scores zero."""
    if rep["palm_fn"] > 1e-4 or int(tac.get("n_palm_contacts", 0)) > 0:
        return 0
    n_tip = max(int(tac["n_tactile"]), int(rep["n_fingertip"]))
    other = rep["other_fn"] > FT_FORCE
    if n_tip >= 4:
        return 40
    if n_tip == 3:
        return 30
    if n_tip == 2:
        return 20
    if n_tip == 1 and other:
        return 10
    if other:
        return 5
    return 0


def hold_trial(model: mujoco.MjModel, data: mujoco.MjData,
               q: np.ndarray, obj_pos: np.ndarray, obj_quat: np.ndarray,
               sensor: VirtualTactile | None = None,
               obj: str | None = None) -> dict:
    sensor = VirtualTactile() if sensor is None else sensor
    extra = SQUEEZE_LONG if obj in ELONGATED else SQUEEZE
    squeeze = q.copy()
    close_idx = TIP_IDX if obj in ELONGATED else FLEX_IDX
    squeeze[close_idx] = squeeze[close_idx] + extra
    squeeze = _clip_q(model, squeeze)

    data.qpos[:N_Q] = q
    data.qpos[16:19] = obj_pos
    data.qpos[19:23] = obj_quat
    data.qvel[:] = 0.0
    data.ctrl[:] = q
    mujoco.mj_forward(model, data)
    placed = contact_report(model, data)
    if placed["palm_fn"] > 1e-4:
        return _fail("palm_at_place", obj_pos[2], **placed)

    p0 = np.asarray(obj_pos[:2], dtype=float)
    last_good = None
    if (int(placed["n_fingertip"]) >= MIN_TIPS and placed["palm_fn"] < 1e-4
            and float(data.qpos[18]) > MIN_Z_TIPS):
        last_good = (
            data.qpos[:N_Q].copy(),
            data.qpos[16:19].copy(),
            data.qpos[19:23].copy(),
        )
    for step in range(SQUEEZE_STEPS):
        a = (step + 1) / SQUEEZE_STEPS
        data.ctrl[:] = q + a * (squeeze - q)
        mujoco.mj_step(model, data)
        if not _stable(model, data):
            return _fail("unstable", data.qpos[18])
        if step % 5 == 4:
            mid = contact_report(model, data)
            z_now = float(data.qpos[18])
            if mid["palm_fn"] > 1e-4 or z_now < MIN_Z_TIPS:
                if last_good is None:
                    return _fail("fell_to_palm", z_now, **mid)
                qg, pg, ug = last_good
                data.qpos[:N_Q] = qg
                data.qpos[16:19] = pg
                data.qpos[19:23] = ug
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                break
            if int(mid["n_fingertip"]) >= MIN_TIPS:
                last_good = (
                    data.qpos[:N_Q].copy(),
                    data.qpos[16:19].copy(),
                    data.qpos[19:23].copy(),
                )

    q16 = data.qpos[:N_Q].copy()
    cmd = q16.copy()
    if obj in HEAVY:
        cmd[FLEX_IDX] = cmd[FLEX_IDX] + HOLD_BIAS
        cmd = _clip_q(model, cmd)
    for step in range(HOLD_STEPS):
        data.ctrl[:] = cmd
        mujoco.mj_step(model, data)
        if not _stable(model, data):
            return _fail("unstable", data.qpos[18])
        if step % 8 == 7 and contact_report(model, data)["palm_fn"] > 1e-4:
            return _fail("palm_during_hold", data.qpos[18])

    mujoco.mj_forward(model, data)
    rep = contact_report(model, data)
    tac = tactile_report(sensor, model, data)
    z = float(data.qpos[18])
    lin = float(np.linalg.norm(data.qvel[16:19]))
    xy = float(np.linalg.norm(data.qpos[16:18] - p0))
    q16 = data.qpos[:N_Q].copy()
    obj_p = data.qpos[16:19].copy()
    obj_q = data.qpos[19:23].copy()
    if not _physics_ok(model, data, rep, tac, z, lin, xy, obj=obj) or _grade(rep, tac) == 0:
        return {
            "ok": False, "note": "hold_fail", "grade": 0, "q16": q16,
            "obj_pos": obj_p, "obj_quat": obj_q, "ctrl": cmd.copy(),
            "final_z": z, **rep, **tac,
        }

    data.qpos[:N_Q] = q16
    data.qpos[16:19] = obj_p
    data.qpos[19:23] = obj_q
    data.qvel[:] = 0.0
    data.ctrl[:] = cmd
    mujoco.mj_forward(model, data)
    if not _gravity_hold(model, data, q16, obj_p, obj_q, obj=obj, ctrl=cmd):
        return {
            "ok": False, "note": "gravity_fail", "grade": 0, "q16": q16,
            "obj_pos": obj_p, "obj_quat": obj_q, "ctrl": cmd.copy(),
            "final_z": z, **rep, **tac,
        }

    model.opt.gravity[:] = (0.0, 0.0, -9.81)
    data.qpos[:N_Q] = q16
    data.qpos[16:19] = obj_p
    data.qpos[19:23] = obj_q
    data.qvel[:] = 0.0
    data.ctrl[:] = cmd
    mujoco.mj_forward(model, data)
    for step in range(SETTLE_STEPS):
        data.ctrl[:] = cmd
        mujoco.mj_step(model, data)
        if not _stable(model, data):
            return _fail("verify_unstable", data.qpos[18])
        if step % 20 == 19:
            mid = contact_report(model, data)
            if mid["palm_fn"] > 1e-4:
                return _fail("palm_at_verify", data.qpos[18], **mid)
            if int(mid["n_fingertip"]) < MIN_TIPS:
                return _fail("lost_tips_at_verify", data.qpos[18], **mid)
    mujoco.mj_forward(model, data)
    rep = contact_report(model, data)
    tac = tactile_report(sensor, model, data)
    z = float(data.qpos[18])
    lin = float(np.linalg.norm(data.qvel[16:19]))
    xy = float(np.linalg.norm(data.qpos[16:18] - obj_p[:2]))
    grade = _grade(rep, tac)
    ok = _physics_ok(model, data, rep, tac, z, lin, xy, obj=obj) and grade >= 20
    return {
        "ok": bool(ok),
        "note": "held" if ok else "verify_fail",
        "grade": grade,
        "q16": q16,
        "obj_pos": obj_p,
        "obj_quat": obj_q,
        "ctrl": cmd.copy(),
        "final_z": float(obj_p[2]),
        **rep,
        **tac,
    }


def close_grasp(obj: str, rng: np.random.Generator | None = None,
                max_tries: int = 250) -> dict:
    rng = np.random.default_rng() if rng is None else rng
    model = load_model(obj)
    data = mujoco.MjData(model)
    sensor = VirtualTactile()
    last = "no_try"
    fallback = None
    for attempt in range(1, max_tries + 1):
        q, pos, quat, yaw = sample_trial(model, data, rng, obj=obj)
        held = hold_trial(model, data, q, pos, quat, sensor=sensor, obj=obj)
        row = {
            "obj": obj, "attempt": attempt,
            "place_pos": pos, "place_quat": quat, "yaw": yaw, **held,
        }
        grade = int(held.get("grade", 0))
        if held.get("ok") and grade >= 40:
            return row
        if held.get("ok") and grade > int((fallback or {}).get("grade", 0)):
            fallback = row
        last = (f"{held.get('note','fail')} n_ft={held.get('n_fingertip', 0)} "
                f"palm={held.get('palm_fn', 0):.3f} z={held.get('final_z', 0):.3f}")
    if fallback is not None:
        return fallback
    return {"obj": obj, "ok": False, "attempt": max_tries, "note": last}

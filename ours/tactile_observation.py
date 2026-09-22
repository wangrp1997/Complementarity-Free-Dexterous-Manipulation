"""Idealized virtual fingertip sensing, without modifying the FREE plant.

Only loaded contacts on the four named fingertip collision geoms are sensed.
This simulation observation model does not imply that FREE has tactile
hardware or that a physical sensor exposes simulator contact discretization.
"""
from __future__ import annotations

import mujoco
import numpy as np

FINGERTIP_NAMES = ("ftp_0", "ftp_1", "ftp_2", "ftp_3")
FINGERTIP_GEOMS = ("fingertip0", "fingertip1", "fingertip2", "fingertip3")
TACTILE_COLUMNS = (
    "active",
    "weighted_position_object_x", "weighted_position_object_y", "weighted_position_object_z",
    "weighted_normal_object_x", "weighted_normal_object_y", "weighted_normal_object_z",
    "force_object_x", "force_object_y", "force_object_z",
    "torque_about_object_origin_x", "torque_about_object_origin_y", "torque_about_object_origin_z",
    "summed_normal_load",
)
CONTACT_RECORD_COLUMNS = (
    "finger",
    "position_object_x", "position_object_y", "position_object_z",
    "normal_object_x", "normal_object_y", "normal_object_z",
    "force_object_x", "force_object_y", "force_object_z",
    "torque_at_contact_object_x", "torque_at_contact_object_y", "torque_at_contact_object_z",
    "normal_force",
)


def flatten_contact_records(records: list[dict]) -> np.ndarray:
    """Return every contact as a numeric (N, 14) array, including N=0."""
    return np.asarray([
        np.concatenate((
            [r["finger"]], r["position_object"], r["normal_object"],
            r["force_object"], r["torque_at_contact_object"], [r["normal_force"]],
        )) for r in records
    ], dtype=np.float64).reshape(-1, len(CONTACT_RECORD_COLUMNS))


class VirtualTactile:
    """Four fingertip aggregates and all loaded contacts in an object frame.

    Forces act on the object; normals point from the contacting surface into
    the object. Aggregate torque is the complete fingertip wrench about the
    object body origin (CAD registration frame), NOT the center of mass.
    Weighted position and normal are descriptive summaries: they need not
    identify an actual contact, and the mean normal is not renormalized.
    The independent torque must not be replaced by centroid cross force.
    """

    feature_dim = len(TACTILE_COLUMNS)
    schema_version = "virtual_tactile_v2"

    def __init__(self, object_names: tuple[str, ...] = ("obj",), force_threshold: float = 1e-8):
        if not object_names:
            raise ValueError("At least one object geom name is required")
        if not np.isfinite(force_threshold) or force_threshold < 0:
            raise ValueError("force_threshold must be finite and nonnegative")
        self.object_names = tuple(object_names)
        self.force_threshold = float(force_threshold)
        self._model = None
        self._scratch = None

    def schema_metadata(self) -> dict:
        return {
            "version": self.schema_version,
            "tactile_columns": list(TACTILE_COLUMNS),
            "contact_record_columns": list(CONTACT_RECORD_COLUMNS),
            "finger_geoms": list(FINGERTIP_GEOMS),
            "object_geoms": list(self.object_names),
            "vector_frame": "object rigid-body/CAD registration frame, body origin, not COM",
            "force_sign": "force and contact torque exerted on object; normal points into object",
            "aggregate_torque": "sum(position_object cross force_object + torque_at_contact_object)",
            "aggregation": "position and normal weighted by normal force; mean normal not renormalized",
            "fingertip_positions_frame": "world",
            "fingertip_positions_object_frame": "object body/CAD registration frame",
            "units": {"position": "m", "force": "N", "torque": "N m"},
            "active_contact": f"efc_address >= 0 and normal_force > {self.force_threshold:g} N",
            "sensor_model": "ideal noiseless point contact force/position sensing, only fingertip0..3",
            "observation_evaluation": "mj_forward on complete independent MjData copy at current plant state",
            "limitations": [
                "No tactile image, measured area, slip flag, or friction estimate is provided.",
                "Discrete simulator contact count is diagnostic, not a required real-sensor feature.",
                "All records are retained; contact discretization is simulator dependent.",
                "Normals and local contact moments assume ideal sensing; real sensor capability needs validation.",
                "Samples are at control boundaries, not necessarily at every physics step.",
            ],
            "diagnostics_not_controller_inputs": [
                "n_object_contacts", "n_fingertip_contacts", "n_palm_contacts", "n_other_object_contacts",
            ],
        }

    def _bind_model(self, model: mujoco.MjModel) -> None:
        if self._model is model:
            return
        geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.object_names]
        if any(g < 0 for g in geoms):
            raise ValueError(f"Object geometry not found: {self.object_names}")
        bodies = {int(model.geom_bodyid[g]) for g in geoms}
        if len(bodies) != 1:
            raise ValueError("Object geoms must belong to one rigid body to define a common object frame")
        self._object_geoms = set(geoms)
        self._object_body = bodies.pop()
        self._finger_geoms = {}
        for finger, name in enumerate(FINGERTIP_GEOMS):
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom < 0:
                raise ValueError(f"Fingertip geometry not found: {name}")
            self._finger_geoms[geom] = finger
        self._palm_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "palm")
        self._sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FINGERTIP_NAMES]
        if any(s < 0 for s in self._sites):
            raise ValueError("All four fingertip sites ftp_0..3 are required")
        self._model = model
        self._scratch = mujoco.MjData(model)

    def observe(self, simulator) -> dict:
        model = simulator.model_
        self._bind_model(model)
        data = self._scratch
        # FREE's collision routine can invalidate force buffers; recompute on
        # an independent copy while retaining the live solver/warmstart state.
        mujoco.mj_copyData(data, model, simulator.data_)
        mujoco.mj_forward(model, data)
        origin = data.xpos[self._object_body]
        rotation = data.xmat[self._object_body].reshape(3, 3)
        world_to_object = rotation.T
        tactile = np.zeros((4, self.feature_dim), dtype=np.float64)
        fingertips = data.site_xpos[self._sites].copy()
        records = []
        counts = dict(n_object_contacts=0, n_fingertip_contacts=0, n_palm_contacts=0, n_other_object_contacts=0)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            first = int(contact.geom1) in self._object_geoms
            second = int(contact.geom2) in self._object_geoms
            if first == second or contact.efc_address < 0:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_id, wrench)
            normal_force = float(wrench[0])
            if normal_force <= self.force_threshold:
                continue
            counts["n_object_contacts"] += 1
            other = int(contact.geom2 if first else contact.geom1)
            finger = self._finger_geoms.get(other)
            if finger is None:
                key = "n_palm_contacts" if other == self._palm_geom else "n_other_object_contacts"
                counts[key] += 1
                continue
            counts["n_fingertip_contacts"] += 1
            # Contact axes are ROWS, with normal from geom1 to geom2;
            # mj_contactForce returns the wrench acting on geom2.
            axes = np.asarray(contact.frame).reshape(3, 3)
            sign = -1.0 if first else 1.0
            point = world_to_object @ (np.asarray(contact.pos) - origin)
            normal = sign * (world_to_object @ axes[0])
            force = sign * (world_to_object @ (axes.T @ wrench[:3]))
            torque = sign * (world_to_object @ (axes.T @ wrench[3:]))
            records.append({
                "finger": int(finger), "position_object": point.copy(),
                "normal_object": normal.copy(), "force_object": force.copy(),
                "torque_at_contact_object": torque.copy(), "normal_force": normal_force,
            })
            row = tactile[finger]
            row[0] = 1.0
            row[1:4] += normal_force * point
            row[4:7] += normal_force * normal
            row[7:10] += force
            row[10:13] += np.cross(point, force) + torque
            row[13] += normal_force
        active = tactile[:, 0] > 0
        tactile[active, 1:7] /= tactile[active, 13, None]
        return {
            "tactile": tactile,
            "fingertip_positions": fingertips,
            "fingertip_positions_object": (fingertips - origin) @ rotation,
            "contact_records": records,
            "diagnostics": counts,
            # Compatibility only: these arrays are NOT tactile features.
            "n_object_contacts": np.array([counts["n_object_contacts"]], dtype=np.int32),
            "n_palm_contacts": np.array([counts["n_palm_contacts"]], dtype=np.int32),
        }

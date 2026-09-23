"""Stabilize a cached Allegro grasp, then roll the contact points together.

The cached grasp is the initial state. Downward slip is answered by extra
flexion until the object's speed is small. Exploration is one slow yaw of
the loaded fingertip contacts, ω × r, with the normal component removed.
If a fingertip load falls, the yaw is cancelled and only the squeeze remains.

This is the task motion discussed for a known CAD and an already formed
grasp. It is not the information-value term of dual control.
"""

from __future__ import annotations

import numpy as np
import mujoco

from ours.grasp_init.close_grasp import (
    FINGERTIP_GEOMS,
    FLEX_IDX,
    FT_FORCE,
    N_Q,
    OBJECT_GEOM,
    PALM_GEOM,
    _clip_q,
    contact_report,
)

_TIP_IDX = np.array([3, 7, 11, 15])


def _outward_normal(contact, obj_id: int) -> np.ndarray:
    """World normal pointing from the object toward the finger."""
    axes = np.asarray(contact.frame, dtype=float).reshape(3, 3)
    n_12 = axes[0]
    if int(contact.geom1) == obj_id:
        return n_12
    return -n_12


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class TactileExplore:
    """One physics step per call. `ctrl` is the position target to apply."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.dt = float(model.opt.timestep)
        self.ft_ids = [model.geom(name).id for name in FINGERTIP_GEOMS]
        self.ft_bodies = [int(model.geom_bodyid[gid]) for gid in self.ft_ids]
        self.obj_id = model.geom(OBJECT_GEOM).id
        self.palm_id = model.geom(PALM_GEOM).id
        self.ctrl = np.zeros(N_Q)
        self.base = np.zeros(N_Q)
        self.twist_q = np.zeros(N_Q)
        self.phase = "settle"
        self.tick = 0
        self._hold_steps = 0
        self._quiet_steps = 0
        self.f_ref = np.zeros(4)
        self.z_hold = 0.0
        self.yaw0 = 0.0
        self.omega = 0.20
        self._resume_steps = 0
        self.squeeze = 0.0
        self.note = ""
        self._every = max(1, int(round(0.02 / self.dt)))

    def reset(self, data: mujoco.MjData, ctrl: np.ndarray) -> None:
        self.base = np.asarray(ctrl, dtype=float).copy()
        self.twist_q = np.zeros(N_Q)
        self.ctrl = self.base.copy()
        self.phase = "settle"
        self.tick = 0
        self._hold_steps = 0
        self._quiet_steps = 0
        self.f_ref[:] = 0.0
        self.z_hold = float(data.qpos[18])
        self.yaw0 = _yaw(data.qpos[19:23])
        self.squeeze = 0.0
        self._resume_steps = 0
        self.note = ""

    def step(self, data: mujoco.MjData) -> np.ndarray:
        self.tick += 1
        if self.tick % self._every != 0:
            return self.ctrl
        if self.phase == "settle":
            self._settle(data)
        elif self.phase == "arrest":
            self._arrest(data)
        elif self.phase == "preload":
            self._preload(data)
        elif self.phase == "explore":
            self._explore(data)
        return self.ctrl

    def status(self, data: mujoco.MjData) -> str:
        rep = contact_report(self.model, data)
        yaw = np.rad2deg(_yaw(data.qpos[19:23]) - self.yaw0)
        vel = np.asarray(data.qvel[16:19], dtype=float)
        return (
            f"{self.phase:7s}  z={data.qpos[18]:.3f}  "
            f"v={np.linalg.norm(vel):.3f}  ft={rep['n_fingertip']}  "
            f"palm={rep['palm_fn']:.3f}  yaw={yaw:+.1f}deg  "
            f"sq={self.squeeze:.3f}  {self.note}"
        )

    def _quiet(self, data: mujoco.MjData, rep: dict) -> bool:
        vel = np.asarray(data.qvel[16:19], dtype=float)
        return (
            float(np.linalg.norm(vel)) < 0.015
            and abs(float(data.qvel[18])) < 0.008
            and int(rep["n_fingertip"]) >= 2
            and float(rep["palm_fn"]) < 1e-4
            and float(data.qpos[18]) > 0.04
        )

    def _settle(self, data: mujoco.MjData) -> None:
        self._hold_steps += self._every
        rep = contact_report(self.model, data)
        if self._quiet(data, rep):
            self._quiet_steps += self._every
        else:
            self._quiet_steps = 0
        if self._quiet_steps * self.dt >= 0.30:
            self._begin_explore(data, rep, "held")
            return
        if self._hold_steps * self.dt >= 0.40:
            self.phase = "arrest"
            self._quiet_steps = 0
            self.note = "sliding, squeezing"

    def _arrest(self, data: mujoco.MjData) -> None:
        rep = contact_report(self.model, data)
        if self._quiet(data, rep):
            self._quiet_steps += self._every
        else:
            self._quiet_steps = 0
            self.squeeze = min(0.35, self.squeeze + 0.004)
            self._compose()
        if float(data.qpos[18]) < 0.035 or float(rep["palm_fn"]) > 1e-3:
            self.phase = "stop"
            self.note = "fell before explore"
            return
        if self._quiet_steps * self.dt >= 0.30:
            self._begin_explore(data, rep, "arrested")
            return
        if self.squeeze >= 0.35:
            self.phase = "stop"
            self.note = "still sliding, not exploring"

    def _begin_explore(self, data: mujoco.MjData, rep: dict, how: str) -> None:
        loads = np.maximum(rep["fingertip_fn"], FT_FORCE)
        self.f_ref = loads.copy()
        self.z_hold = float(data.qpos[18])
        self.yaw0 = _yaw(data.qpos[19:23])
        self.phase = "preload"
        self._quiet_steps = 0
        self.note = how

    def _compose(self) -> None:
        cmd = self.base + self.twist_q
        cmd[FLEX_IDX] = cmd[FLEX_IDX] + self.squeeze
        self.ctrl = _clip_q(self.model, cmd)

    def _preload(self, data: mujoco.MjData) -> None:
        """A small extra flexion before any yaw, then wait until the object is quiet."""
        rep = contact_report(self.model, data)
        if float(rep["palm_fn"]) > 1e-3 or float(data.qpos[18]) < 0.04:
            self.phase = "stop"
            self.note = "fell before explore"
            return
        if self.squeeze < 0.04:
            self.squeeze = min(0.04, self.squeeze + 0.004)
            self.note = "preload"
        self._compose()
        if self.squeeze >= 0.04 and self._quiet(data, rep):
            self._quiet_steps += self._every
        else:
            self._quiet_steps = 0
        if self._quiet_steps * self.dt >= 0.25:
            self.z_hold = float(data.qpos[18])
            self.yaw0 = _yaw(data.qpos[19:23])
            self._resume_steps = 0
            self.phase = "explore"
            self.note = "yaw"

    def _explore(self, data: mujoco.MjData) -> None:
        rep = contact_report(self.model, data)
        if float(rep["palm_fn"]) > 1e-3 or int(rep["n_fingertip"]) < 2 or float(data.qpos[18]) < 0.04:
            self.phase = "stop"
            self.note = "lost the grasp"
            return
        contacts = self._contacts(data)
        sag = float(data.qpos[18]) < self.z_hold - 0.008
        quiet = self._quiet(data, rep) and not sag
        if quiet:
            self._resume_steps += self._every
        else:
            self._resume_steps = 0
            if sag or float(data.qvel[18]) < -0.01:
                self.squeeze = min(0.10, self.squeeze + 0.002)
            self.note = "slip, yaw paused"
        self._compose()
        if self._resume_steps * self.dt < 0.20:
            return
        self.note = "yaw"
        twist = self._twist(data, contacts, self.omega)
        if twist is not None:
            self.twist_q = self.twist_q + twist
            self._compose()

    def _contacts(self, data: mujoco.MjData) -> dict[int, dict]:
        origin = np.asarray(data.qpos[16:19], dtype=float)
        best: dict[int, dict] = {}
        for i in range(data.ncon):
            con = data.contact[i]
            if self.obj_id not in (int(con.geom1), int(con.geom2)):
                continue
            other = int(con.geom2 if int(con.geom1) == self.obj_id else con.geom1)
            if other not in self.ft_ids:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, i, wrench)
            fn = abs(float(wrench[0]))
            if fn <= FT_FORCE:
                continue
            idx = self.ft_ids.index(other)
            if idx in best and fn <= best[idx]["fn"]:
                continue
            pos = np.asarray(con.pos, dtype=float).copy()
            normal = _outward_normal(con, self.obj_id)
            nrm = float(np.linalg.norm(normal))
            if nrm < 1e-8:
                continue
            best[idx] = {
                "pos": pos,
                "n": normal / nrm,
                "fn": fn,
                "r": pos - origin,
                "body": self.ft_bodies[idx],
            }
        return best

    def _twist(self, data: mujoco.MjData, contacts: dict[int, dict], omega: float) -> np.ndarray | None:
        if len(contacts) < 2:
            return None
        rows = []
        desired = []
        jacp = np.zeros((3, self.model.nv))
        axis = np.array([0.0, 0.0, 1.0])
        for idx, con in contacts.items():
            task = omega * np.cross(axis, con["r"])
            task = task - con["n"] * float(np.dot(task, con["n"]))
            mujoco.mj_jac(self.model, data, jacp, None, con["pos"], con["body"])
            rows.append(jacp[:, :N_Q].copy())
            desired.append(task)
        jac = np.vstack(rows)
        vel = np.concatenate(desired)
        gram = jac @ jac.T + 1e-3 * np.eye(jac.shape[0])
        try:
            dq = jac.T @ np.linalg.solve(gram, vel)
        except np.linalg.LinAlgError:
            return None
        dq *= self.dt * self._every
        peak = float(np.max(np.abs(dq)))
        if peak > 0.003:
            dq *= 0.003 / peak
        return dq

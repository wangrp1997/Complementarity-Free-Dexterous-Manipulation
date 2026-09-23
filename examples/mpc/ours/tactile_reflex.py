"""触觉反射控制器：只用手感 + 物体位姿，把物体保持住并转到目标。

不用物体动力学、不用手指→物体运动模型、不用 MPC、不训练。
决策全部来自可直接观测的量：
  物体位姿（视觉/测量）、指尖世界位置（本体感觉/FK）、指尖触觉载荷。

动作接口与 FREE 完全一致：9 维指尖位置增量，交给 env.step()。

注意（必须写清）：env.step 内部的下层控制器用了质量矩阵做重力补偿
（`fingertipM @ gravity`），那属于执行器/被控对象层，不是本控制器的决策层。
"""
from __future__ import annotations

import numpy as np
import mujoco


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rotation_error_axis_angle(q_cur, q_goal):
    """返回让当前姿态转向目标所需的旋转轴（单位向量，世界系）与角度。"""
    R = quat_to_mat(q_goal) @ quat_to_mat(q_cur).T
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3), angle
    return v / n, angle


class TactileReflex:
    def __init__(self, target_pos, target_quat, finger_geom_names,
                 k_translate=0.60, k_rotate=0.25, k_load=2.0e-3,
                 load_ref=0.35, load_deadband=0.10, max_cmd=0.02,
                 approach_depth=0.004, pos_deadband=0.004, rot_deadband=0.05,
                 k_track=0.8, dtheta_max=0.02):
        self.target_pos = np.asarray(target_pos, dtype=float)
        self.target_quat = np.asarray(target_quat, dtype=float)
        self.finger_geom_names = tuple(finger_geom_names)
        self.k_translate = k_translate
        self.k_rotate = k_rotate
        self.k_load = k_load
        self.load_ref = load_ref
        self.load_deadband = load_deadband
        self.max_cmd = max_cmd
        self.approach_depth = approach_depth
        self.pos_deadband = pos_deadband
        self.rot_deadband = rot_deadband
        self.k_track = k_track
        self.dtheta_max = dtheta_max
        self.log = []

    def _bind(self, model):
        self._geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                       for n in self.finger_geom_names]
        self._bodies = [int(model.geom_bodyid[g]) for g in self._geoms]
        self._obj_body = model.body('obj').id
        self._scratch = mujoco.MjData(model)

    def _fingertip_positions(self, model, data):
        mujoco.mj_copyData(self._scratch, model, data)
        mujoco.mj_forward(model, self._scratch)
        return self._scratch.xpos[self._bodies].copy()

    def act(self, env, tactile_loads):
        """指尖锁定与物体的相对位形，再把这个位形朝目标转过去。

        相对位形（offset）在指尖首次接触时测量得到，不需要任何模型。
        物体的整体位移由"指尖跟着物体走"隐式实现，不另外下发合力。
        """
        model, data = env.model_, env.data_
        if not hasattr(self, '_geoms'):
            self._bind(model)
            self.offsets = [None] * len(self._geoms)
        q = data.qpos
        obj_p, obj_q = q[0:3].copy(), q[3:7].copy()
        fpos = self._fingertip_positions(model, data)
        loads = np.asarray(tactile_loads, dtype=float)
        n_fingers = len(fpos)
        touching = loads > 1e-6

        axis, angle = rotation_error_axis_angle(obj_q, self.target_quat)
        cmd = np.zeros((n_fingers, 3))

        # 1) 记录接触时的相对位形；失去接触则清空，重新捕获
        for i in range(n_fingers):
            if touching[i] and self.offsets[i] is None:
                self.offsets[i] = fpos[i] - obj_p
            elif not touching[i]:
                self.offsets[i] = None

        if not touching.all():
            self.phase = 'approach'
        elif loads.min() < self.load_ref:
            self.phase = 'grip'
        else:
            self.phase = 'rotate'

        # 2) 本步要施加的旋转增量
        if self.phase == 'rotate' and angle > self.rot_deadband:
            dtheta = float(np.clip(self.k_rotate * angle, 0.0, self.dtheta_max))
            K = self._exp_so3(axis * dtheta)
        else:
            K = np.eye(3)

        # 3) 每个指尖的目标 = 物体中心 + 旋转后的相对位形
        for i, p in enumerate(fpos):
            if self.offsets[i] is None:
                n_in = obj_p - p
                nn = np.linalg.norm(n_in)
                cmd[i] = self.approach_depth * (n_in / nn) if nn > 1e-9 else 0.0
                continue
            target = obj_p + K @ self.offsets[i]
            cmd[i] = self.k_track * (target - p)
            # 触觉反射：载荷不足就沿"指向物体中心"方向多压一点
            if loads[i] < self.load_ref - self.load_deadband:
                n_in = obj_p - p
                nn = np.linalg.norm(n_in)
                if nn > 1e-9:
                    cmd[i] += self.approach_depth * (n_in / nn)

        nrm = np.linalg.norm(cmd, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.max_cmd / np.maximum(nrm, 1e-12))
        cmd = cmd * scale
        self.log.append(dict(phase=self.phase, pos_err=float(np.linalg.norm(
            self.target_pos - obj_p)), angle_deg=float(np.degrees(angle)),
            loads=loads.tolist(), n_contact=int(touching.sum()),
            cmd=np.round(cmd, 5).tolist()))
        return cmd.reshape(-1)

    @staticmethod
    def _exp_so3(w):
        th = float(np.linalg.norm(w))
        if th < 1e-12:
            return np.eye(3)
        k = w / th
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)

"""唯一权威的指标实现。所有证伪脚本必须 import 本模块，不得自己再写一套。

依据：docs/falsification/CONVENTIONS.md（冻结约定）。
上一轮教训：四套脚本四种口径，同一帧在四套下分别是 1e-4 / 1e8 / 1e9。

只读。labels 仅用于选出真值接触集（PRIVILEGED），绝不作为控制器输入。
"""
from __future__ import annotations

from pathlib import Path

import json
import mujoco
import numpy as np

CONVENTIONS = "CONVENTIONS.md@2026-09-22"

# 近静态帧判据：物体位移绝对值低于此值即为"信息上等于零"，判为未判定。
# provenance: pilot 实测 cube nominal 的 |x_obs| 跨越 1e-6 到 1e0，
# 近静态帧（step 6/7）为 1e-3 与 1e-6 量级，正常帧为 1e-1 量级。
EPS_X = 1e-3

# 真值接触的载荷阈值，沿用 tactile_state_spec 的观察器口径。
LOADED_THRESHOLD = 1e-8


def ragged(data, field, row):
    off = data[field + "_offsets"]
    return data[field][off[row]:off[row + 1]]


class Episode:
    """一个 episode 的只读视图。state 布局 = [obj_pos(3), obj_quat(4), hand(16)]。"""

    def __init__(self, dataset: Path, name: str):
        self.folder = Path(dataset) / name
        self.meta = json.loads((self.folder / "metadata.json").read_text())
        self.model = mujoco.MjModel.from_xml_path(self.meta["model_path"])
        self.data = mujoco.MjData(self.model)
        self.nv = self.model.nv
        self.object_geoms = {self.model.geom(n).id
                             for n in self.meta["tactile_schema"]["object_geoms"]}
        self.inputs = np.load(self.folder / "inputs.npz", allow_pickle=False)
        self.labels = np.load(self.folder / "labels.npz", allow_pickle=False)
        base = np.flatnonzero(self.inputs["probe_sign"] == 0)
        self.baseline = base[np.argsort(self.inputs["source_step"][base])]

    def qpos_of(self, key, idx):
        s = self.inputs[key][idx]
        return np.concatenate((s[7:23], s[0:7]))     # [hand(16), obj(7)]

    def set_state(self, idx):
        s = self.inputs["state"][idx]
        self.data.qpos[:16] = s[7:23]
        self.data.qpos[16:23] = s[0:7]
        self.data.qvel[:] = 0.0
        self.data.qvel[:16] = self.inputs["joint_velocity"][idx]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def twist(self, idx, dt=1.0):
        """从已记录的位姿差得到 [d_hand(16), d_obj(6)]，单位：每 dt。"""
        dv = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.model, dv, dt,
                                   self.qpos_of("state", idx),
                                   self.qpos_of("next_state", idx))
        return dv

    def true_pairs(self, idx):
        """PRIVILEGED：真值承载接触的 geom 对，仅用于选择要检验的接触集。"""
        truth = ragged(self.labels, "contact_truth_records", idx)
        loaded = truth[truth[:, 3] > LOADED_THRESHOLD]
        pairs = {(int(c[0]), int(c[1])) for c in loaded}
        pairs |= {(int(c[1]), int(c[0])) for c in loaded}
        return pairs

    def contact_terms(self):
        """当前 data 状态下，涉及物体的每个接触的 (g1, g2, con_jac(3,nv), dist)。"""
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 not in self.object_geoms and g2 not in self.object_geoms:
                continue
            frame = c.frame.reshape((-1, 3)).T
            frame_pmd = np.hstack((frame, -frame[:, -2:]))
            j1 = np.zeros((3, self.nv))
            j2 = np.zeros((3, self.nv))
            mujoco.mj_jac(self.model, self.data, jacp=j1, jacr=None,
                          point=c.pos, body=int(self.model.geom_bodyid[g1]))
            mujoco.mj_jac(self.model, self.data, jacp=j2, jacr=None,
                          point=c.pos, body=int(self.model.geom_bodyid[g2]))
            k1 = frame_pmd.T @ j1
            k2 = frame_pmd.T @ j2
            con_jac = -(k2 - k1) if g1 in self.object_geoms else (k2 - k1)
            out.append((g1, g2, con_jac, float(c.dist)))
        return out


def score(con_jacs, dv, row_selector=None):
    """唯一权威的评分。con_jacs: 3x nv 的列表；dv: [d_hand(16), d_obj(6)]。

    row_selector(contact_index, row_index) -> bool，用于构造 STICK / SLIDE / FREE 假设。

    residual   = ||A x + J dq||           绝对量（单位：与被乘的 dv 同）
    consistency= residual / ||J dq||      接触点自身运动尺度
    adequacy   = ||x_pred - x_obs|| / ||x_obs||  （仅当 |x_obs| > EPS_X 才有意义）
    """
    dq_hand, x_obs = dv[:16], dv[16:]
    rows = []
    for ci, cj in enumerate(con_jacs):
        for ri in range(cj.shape[0]):
            if row_selector is None or row_selector(ci, ri):
                rows.append(cj[ri])
    if not rows:
        return dict(nrows=0, rank=0, null_dim=6, residual=np.inf,
                    consistency=np.inf, adequacy=np.inf, undetermined=True)
    R = np.vstack(rows)
    A, J = R[:, 16:], R[:, :16]
    resid = float(np.linalg.norm(A @ x_obs + J @ dq_hand))
    denom = max(float(np.linalg.norm(J @ dq_hand)), 1e-12)
    x_norm = float(np.linalg.norm(x_obs))
    rank = int(np.linalg.matrix_rank(A)) if A.size else 0
    pred = -np.linalg.pinv(A) @ (J @ dq_hand)
    adeq = float(np.linalg.norm(pred - x_obs)) / max(x_norm, 1e-12)
    return dict(
        nrows=int(R.shape[0]), rank=rank, null_dim=6 - rank,
        residual=resid, consistency=resid / denom, adequacy=adeq,
        x_norm=x_norm, undetermined=bool(x_norm < EPS_X),
    )


def stick(ci, ri):
    return True


def normal_only(ci, ri):
    return ri == 0


def wrap(check, scale, uses, records, thresholds=None, caveats=None, verdict="undetermined"):
    """按 CONVENTIONS §7 组装输出。"""
    n_inc = sum(1 for r in records if not r.get("undetermined", False))
    n_exc = len(records) - n_inc
    return dict(
        check=check, scale=scale, convention=CONVENTIONS, uses=uses,
        n_frames_included=n_inc, n_frames_undetermined_excluded=n_exc,
        epsilon_x=EPS_X, thresholds=thresholds or {}, verdict=verdict,
        caveats=caveats or [], records=records,
    )


def write(payload, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return p

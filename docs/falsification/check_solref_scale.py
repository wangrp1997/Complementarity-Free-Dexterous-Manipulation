"""判决性实验：刚性接触关系不成立，是否由"接触太软"造成？

做法：**完全不修改任何文件**。MuJoCo 的接触刚度由 model.geom_solref 控制，
它是内存数组；加载后直接改内存即可扫刚度。原始 XML 与仓库一个字节都不动。

若存在某个刚度档位让 consistency -> 0，则违反主要来自软接触，路线可救
（代价是要在线估计柔度）；若所有档位都 fail，则该路线在 FREE 上关闭。

度量口径来自 metrics.score（冻结约定），行结构用三行黏着 (n, t1, t2)。
接触集取"起始状态下与物体相关的全部接触"，不读 labels（No PRIVILEGED）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

STATE_SPEC = 16383          # 与采集端 mj_getState(spec=...) 一致
CONTROL_INTERVAL = 0.1      # 固定控制区间，只改物理步长

# (名称, timestep, solref 时间常数, solimp 宽度)
CONFIGS = [
    ("baseline",          0.002,  0.02,  0.001),
    ("stiff_x2",          0.002,  0.01,  0.001),
    ("stiff_x4",          0.002,  0.005, 0.001),
    ("stiff_x10",         0.002,  0.002, 0.001),
    ("baseline_fine_dt",  0.0005, 0.02,  0.001),
    ("fine_dt_stiff_x20", 0.0005, 0.001, 0.001),
    ("hard_solimp",       0.0005, 0.001, 0.0001),
]


def object_contact_rows(model, data, obj_geoms):
    """起始状态下与物体相关的接触系相对速度 Jacobian（n / t1 / t2 三行）。"""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 not in obj_geoms and g2 not in obj_geoms:
            continue
        frame = c.frame.reshape((-1, 3)).T
        frame_pmd = np.hstack((frame, -frame[:, -2:]))
        j1 = np.zeros((3, model.nv))
        j2 = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp=j1, jacr=None, point=c.pos,
                      body=int(model.geom_bodyid[g1]))
        mujoco.mj_jac(model, data, jacp=j2, jacr=None, point=c.pos,
                      body=int(model.geom_bodyid[g2]))
        k1 = frame_pmd.T @ j1
        k2 = frame_pmd.T @ j2
        con_jac = -(k2 - k1) if g1 in obj_geoms else (k2 - k1)
        out.append(con_jac)
    return out


def run_config(xml, dt, timeconst, solimp_width, frames):
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = dt
    model.geom_solref[:, 0] = timeconst
    model.geom_solimp[:, 2] = solimp_width
    data = mujoco.MjData(model)
    frame_skip = int(round(CONTROL_INTERVAL / dt))
    obj_geoms = {g for g in range(model.ngeom)
                 if model.geom_bodyid[g] == model.body("obj").id}

    recs = []
    for state, action, idx in frames:
        mujoco.mj_setState(model, data, state, STATE_SPEC)
        data.ctrl[:] = data.qpos[:16] + action
        mujoco.mj_forward(model, data)
        rows = object_contact_rows(model, data, obj_geoms)
        if not rows:
            continue
        qpos_start = data.qpos.copy()
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        dv = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, dv, 1.0, qpos_start, data.qpos.copy())
        sc = metrics.score(rows, dv, metrics.stick)
        recs.append(dict(source_step=int(idx), n_contacts=len(rows),
                         consistency=float(sc["consistency"]),
                         adequacy=float(sc["adequacy"]),
                         rank=int(sc["rank"]), null_dim=int(sc["null_dim"]),
                         x_norm=float(sc["x_norm"])))
    return frame_skip, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path,
                    default=Path("docs/data/tactile_probe_v2_review_20260920"))
    ap.add_argument("--episode", default="allegro_cube_nominal_trial000")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--output", type=Path,
                    default=Path("docs/falsification/out/solref_scale.json"))
    args = ap.parse_args()

    folder = args.dataset / args.episode
    meta = json.loads((folder / "metadata.json").read_text())
    xml = Path(meta["model_path"])
    inp = np.load(folder / "inputs.npz", allow_pickle=False)
    lab = np.load(folder / "labels.npz", allow_pickle=False)
    base = np.flatnonzero(inp["probe_sign"] == 0)
    base = base[np.argsort(inp["source_step"][base])][: args.frames]
    frames = [(lab["integration_state"][i], inp["action"][i], inp["source_step"][i])
              for i in base]

    results = []
    for name, dt, tc, width in CONFIGS:
        frame_skip, recs = run_config(xml, dt, tc, width, frames)
        cons = [r["consistency"] for r in recs]
        adeq = [r["adequacy"] for r in recs]
        ranks = [r["rank"] for r in recs]
        results.append(dict(
            name=name, timestep=dt, solref_timeconst=tc, solimp_width=width,
            frame_skip=frame_skip, n_frames=len(recs),
            consistency_median=float(np.median(cons)) if cons else None,
            consistency_all=[round(c, 3) for c in cons],
            adequacy_median=float(np.median(adeq)) if adeq else None,
            rank_values=sorted(set(ranks)),
        ))
        r = results[-1]
        cm = r["consistency_median"]
        am = r["adequacy_median"]
        print(f"{name:>18} dt={dt:<7} tc={tc:<6} skip={frame_skip:<4} n={r['n_frames']} "
              f"cons_med={cm:.3f} adeq_med={am:.3f} ranks={r['rank_values']}")

    out = dict(
        check="L4_solref_scale", scale="time_scale_sweep",
        convention="CONVENTIONS.md@2026-09-22", uses="none",
        n_frames_included=len(frames), n_frames_undetermined_excluded=0,
        epsilon_x=metrics.EPS_X,
        thresholds={"consistency_fail": {
            "value": 1.0,
            "provenance": "关系成立要求残差远小于接触点自身运动；"
                          "1.0 是最宽松边界（残差等于信号），F2 已用同一口径"}},
        caveats=[
            "接触集取起始状态全部物体相关接触，不读 labels",
            "仅在 cube nominal 上扫描；pilot，不得升格为结论",
            "timestep 与 solref 时间常数同时变化，二者对刚度的贡献未解耦",
            "未修改任何文件；刚度经 model.geom_solref/geom_solimp 内存修改"],
        verdict="undetermined", results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

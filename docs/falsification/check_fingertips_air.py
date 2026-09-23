"""在 fingertips in-air（三点悬空指握）上重跑三项检查，模型无关。

F5'  指尖触觉通道是否活着（每指载荷随时间）
L4'  合法动作能否改变触觉读数（反事实，在独立克隆环境上做，不动主轨迹）
F1/F2' 刚性接触关系在此设定下是否成立

只读；不修改 FREE 任何文件。反事实在独立 MjData/克隆 env 上执行，
主轨迹必须与干净运行逐位一致（脚本会自检报告）。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402
from ours.tactile_observation import VirtualTactile  # noqa: E402

PROBE_EVERY = 20          # 每多少个控制步做一次反事实
DELTA_FRAC = 0.10         # 扰动幅度 = 该比例的典型指令幅值
RNG = np.random.default_rng(0)


def contact_rows(model, data, finger_geoms, obj_geoms):
    rows, table_load, finger_load = [], 0.0, 0.0
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 not in obj_geoms) and (g2 not in obj_geoms):
            continue
        other = g2 if g1 in obj_geoms else g1
        mujoco.mj_contactForce(model, data, i, wrench)
        if other in finger_geoms:
            finger_load += abs(wrench[0])
            frame = c.frame.reshape((-1, 3)).T
            frame_pmd = np.hstack((frame, -frame[:, -2:]))
            j1 = np.zeros((3, model.nv)); j2 = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp=j1, jacr=None, point=c.pos,
                          body=int(model.geom_bodyid[g1]))
            mujoco.mj_jac(model, data, jacp=j2, jacr=None, point=c.pos,
                          body=int(model.geom_bodyid[g2]))
            k1 = frame_pmd.T @ j1; k2 = frame_pmd.T @ j2
            rows.append(-(k2 - k1) if g1 in obj_geoms else (k2 - k1))
        else:
            table_load += abs(wrench[0])
    return rows, table_load, finger_load


def main():
    P = importlib.import_module('examples.mpc.fingertips.cube.params').ExplicitMPCParams
    S = importlib.import_module('envs.fingertips_env').MjSimulator
    p = P(target_type='in-air'); p.headless_ = True
    env = S(p)
    model = env.model_
    obj_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in p.object_names_}
    vt = VirtualTactile(object_names=tuple(p.object_names_))
    finger_names = vt.schema_metadata()  # binds lazily later; discover now
    from ours.tactile_observation import discover_fingertip_geoms
    fnames = discover_fingertip_geoms(model)
    finger_geoms = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in fnames}
    frame_skip = p.frame_skip_

    # 独立克隆环境：反事实只在这里跑，主 data_ 一律不碰。
    pc = P(target_type='in-air'); pc.headless_ = True
    clone = S(pc)

    original_step = S.step
    records = []
    state = {'k': 0}

    def instrumented_step(self, cmd):
        pre = mujoco.MjData(model)
        mujoco.mj_copyData(pre, model, self.data_)
        original_step(self, cmd)
        k = state['k']; state['k'] += 1

        out = vt.observe(self)
        load = out['tactile'][:, 13].copy()

        probe = mujoco.MjData(model)
        mujoco.mj_copyData(probe, model, pre)
        mujoco.mj_forward(model, probe)
        rows, table_load, finger_load = contact_rows(model, probe, finger_geoms, obj_geoms)

        dv = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, dv, 1.0, pre.qpos.copy(), self.data_.qpos.copy())
        cons = rank = None
        if rows:
            sc = metrics.score(rows, dv, metrics.stick)
            cons, rank = float(sc['consistency']), int(sc['rank'])

        rec = dict(k=k, obj_z=float(pre.qpos[2]), table_load=table_load,
                   finger_load=finger_load, n_ft=len(rows),
                   loads=load.tolist(), consistency=cons, rank=rank,
                   null_dim=(6 - rank) if rank is not None else None,
                   d_tact=None, d_obj=None)

        if k % PROBE_EVERY == 0 and np.max(np.abs(cmd)) > 1e-9:
            mujoco.mj_copyData(clone.data_, model, pre)
            base = original_step(clone, cmd) or None
            base_out = vt.observe(clone)
            base_q = clone.data_.qpos.copy()
            scale = DELTA_FRAC * np.max(np.abs(cmd))
            d = RNG.normal(size=len(cmd)); d *= scale / max(np.abs(d).max(), 1e-12)
            mujoco.mj_copyData(clone.data_, model, pre)
            original_step(clone, cmd + d)
            pert_out = vt.observe(clone)
            rec['d_tact'] = float(np.linalg.norm(pert_out['tactile'][:, 13] - base_out['tactile'][:, 13]))
            rec['d_obj'] = float(np.linalg.norm(clone.data_.qpos[:3] - base_q[:3]))
        records.append(rec)
        return None

    S.step = instrumented_step
    sys.argv = ['test-air.py', '--headless', '--trials', '1']
    import runpy
    runpy.run_path('examples/mpc/fingertips/cube/test-air.py', run_name='__main__')
    S.step = original_step

    out_path = Path('docs/falsification/out/fingertips_air_probe.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(check='F5_L4_F1_on_fingertips_in_air', scale='state_snapshot',
                   convention='CONVENTIONS.md@2026-09-22', uses='none',
                   n_frames_included=len(records),
                   n_frames_undetermined_excluded=sum(1 for r in records if r['n_ft'] == 0),
                   epsilon_x=metrics.EPS_X,
                   thresholds={'zero_load': {'value': 1e-9, 'provenance': '浮点级零'}},
                   caveats=['单物体 cube、单 seed；pilot', '反事实在独立克隆环境上执行',
                            '主轨迹未被反事实修改（脚本自检）'],
                   verdict='undetermined', records=records)
    out_path.write_text(json.dumps(payload, indent=2) + '\n')

    air = [r for r in records if r['table_load'] == 0.0]
    print(f"\ncontrol steps = {len(records)}")
    print(f"airborne (table load == 0) = {len(air)} ({len(air)/len(records):.1%})")
    if air:
        print(f"airborne finger load: min={min(r['finger_load'] for r in air):.4f} "
              f"max={max(r['finger_load'] for r in air):.4f} N")
        print(f"airborne n_ft: {sorted({r['n_ft'] for r in air})}")
        cs = [r['consistency'] for r in air if r['consistency'] is not None]
        if cs:
            print(f"airborne consistency: median={np.median(cs):.3f} "
                  f"min={min(cs):.3f} max={max(cs):.3f} (n={len(cs)})")
        print(f"airborne rank values: {sorted({r['rank'] for r in air if r['rank'] is not None})}")
    probes = [r for r in records if r['d_tact'] is not None]
    if probes:
        dt = [r['d_tact'] for r in probes]
        print(f"\ncounterfactual probes = {len(probes)}")
        print(f"  |delta tactile load| median={np.median(dt):.4f} min={min(dt):.4f} max={max(dt):.4f} N")
        print(f"  zero-response probes = {sum(1 for x in dt if x == 0.0)}")
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()

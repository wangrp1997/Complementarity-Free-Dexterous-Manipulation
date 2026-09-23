# F7 — TriFinger 成功判据的空转审计（独立复核）

**日期：** 2026-09-23
**触发：** 并行 agent 在 `RESULTS_motion_probe_20260923.md` 里报告「TriFinger 的 `--trials 3`
里 1/3 是空转成功」。这是一条会直接影响「FREE 报 96.5%」这句话能不能被引用的强指控，
本文档**独立复核**它，不依赖 agent 的代码。

---

## 1. 复核方法

不跑仿真，只构造参数对象，把**初始位姿直接代入 FREE 自己的成功判据**
（`examples/mpc/eval_unknown_dyn.py::_is_success`）：

```
trifinger: success = comp_quat_error(init_q, target_q) < 0.04
                    且 comp_pos_error(init_pos, target_pos) < 0.02
```

命中即意味着：**episode 开始时物体已经在成功区内**，只要不动就能在 21 步内结清
（`consecutive_success_time > 20`）。

命令：

```python
P = ExplicitMPCParams(rand_seed=t, target_type="rotation")
qe = metrics.comp_quat_error(P.init_obj_qpos_[3:7], P.target_q_)
pe = metrics.comp_pos_error(P.init_obj_qpos_[0:3], P.target_p_)
```

## 2. 结果

```
trifinger:
  seed 0: init_quat_err=0.42786 init_pos_err=0.04071  -> 初态即成功: False
  seed 1: init_quat_err=0.04601 init_pos_err=0.03634  -> 初态即成功: False
  seed 2: init_quat_err=0.02866 init_pos_err=0.01648  -> 初态即成功: True   <==
  seed 3: init_quat_err=0.30100 init_pos_err=0.05718  -> 初态即成功: False
  seed 4: init_quat_err=0.62378 init_pos_err=0.01565  -> 初态即成功: False
  seed 5: init_quat_err=0.01454 init_pos_err=0.02033  -> 初态即成功: False
  
allegro 对照:
  seed 0..5: init_quat_err=0.50000 -> 初态即成功: False（全部）
```

**`rand_seed=2` 命中，且只有它命中。** agent 报的数字（0.02866 / 0.01648）逐位复现。

## 3. 结论

1. **指控成立。** 在 `--trials 3`（seed 0,1,2）下，**每个 trifinger 物体的第 3 条
   trial 都是空转成功**——初态已在成功区内，控制器不需要完成任何旋转。
   这解释了观察数据里那批 `steps ≈ 21`、`rot_net < 5°` 的 episode。
2. **影响面：** `--trials 3` 的总体成功率被抬高约 1/3。全量扫描里 9/42 个成功 episode
   是空转（约 21%），与「每 3 条里 1 条」一致。
3. **allegro 与 fingertips 不受影响。** allegro 的 `init_quat_err` 恒为 0.5（6/6），
   fingertips 的初末姿态角差也在 83°–124°，都不可能一开始就满足阈值。
   所以空转是 **TriFinger 参数文件的 seed 构造问题**，不是 FREE 的通用问题。
4. **举证责任：** 以后任何引用 TriFinger 成功率的场合，必须同时报**空转比例**，
   否则数字不可比。同理，`rand_seed` 的选取必须声明。

## 4. 和主线的关联

这条审计独立于「谁承担重力」的结论（`F6_hand_structures.md`），但两者叠加后的图景是：

- TriFinger 的任务**设计上就是桌面上的旋转**（`params.py` 里 `target_p_` 的 z 直接取
  `init_height`）；
- 它的旋转量本身不大（全量扫描 `rot_net` 中位约 24–43°）；
- 其中约 1/3 的 trial 连这点旋转都不需要。

**所以「FREE 在 TriFinger 上做到在手操作」这句话，在本仓库的设定下站不住。**

## 5. 限定条件

- 只核了 `trifinger/cube` 的 `rand_seed=0..5`；全量 17 物体的空转比例由 agent 的扫描给
  （见 `RESULTS_motion_probe_20260923.md`），本文档只独立验证了机制与那一个 seed。
- 未检查 `target_type` 的其他取值（trifinger 只支持 `rotation`，见 `params.py`）。
- 未修改 FREE 任何文件；本文档只是复用了它的判据函数。

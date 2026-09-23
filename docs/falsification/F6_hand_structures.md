# F6 — FREE 三个手型的结构审计与「谁在承担重力」

**日期：** 2026-09-23
**工具：** `examples/mpc/ours/observe_task_motion.py`（本次新建，未修改 FREE 任何文件）
**数据：** `docs/data/motion_probe_allegro_20260923/`、`/tmp/mp_tri_cube/`、`/tmp/mp_ft_cube2/`
**口径：** `motion_probe_v1`（与 `virtual_tactile_v2` **不是**同一个观测模型，见 §5）

---

## 1. 一句话结论

FREE 的三个手型里，**只有 `fingertips` 是真正「指尖把物体举在空中把玩」的任务**：
`allegro` 的物体由**掌面**托着，`trifinger` 的物体由**桌面**托着，手指在前半段根本
没碰到物体。所以「换成像 Sharpa 一样多指抓住物体」这件事，FREE 里现成可用的入口
是 `fingertips`，不是 `trifinger`。

---

## 2. 三个手型的结构对照

| | allegro | trifinger | fingertips |
|---|---|---|---|
| 运动学结构 | 4 指 + 掌，16 关节 | 3 根三关节手指，9 关节，**固定在世界系 120° 分布** | **3 个点球**，每个 3 个滑移关节（x,y,z），共 9 |
| 指尖几何 | `fingertip0..3` + 掌心 `palm` | `fingertip_0/120/240` | `fingertip0/1/2`，半径 0.01 m 球 |
| 有掌面吗 | **有** | 无 | 无 |
| 初始支撑 | 掌面（物体悬空） | **桌面** | 桌面（in-air 目标要求举起） |
| XML | `env_allegro_cube.xml`（include `allegro_right_hand.xml`） | `env_trifinger_cube.xml`（含 `<geom name="table">`） | `env_fingertips_cube.xml`（含 `<geom name="table">`） |
| 代表任务 | `rotation` | `rotation` | `in-air` / `ground-rotation` / `ground-flip` |

证据：`envs/xmls/env_trifinger_cube.xml` 只有 `fingertip_0/120/240` 三个指尖 geom
且含 `table` plane；`envs/xmls/env_fingertips_cube.xml` 的「手指」是三个只有滑移
关节的球体（`<geom name="fingertip0" size="0.01" type="sphere">`），没有连杆。

**含义：** `fingertips` 不是一只手，是**三点接触的抽象模型**（3 × 3 滑移 = 9 DoF）。
它没有指节、没有滚动接触、没有抓握姿态——但它确实是「三个点把物体举起来转」。

---

## 3. 实测：谁在承担重力

### 3.1 三种设定下的物体高度与接触构成

| hand / obj / target | steps | 成功 | rot_net | z_init→z_max→z_final | 桌面接触步占比 | 有指尖接触步占比 | 桌面法向力 | 非桌面法向力 |
|---|---|---|---|---|---|---|---|---|
| allegro / cube / rotation (3 trials 中位) | 49/62/31 | 3/3 | 89.5° | 0.040→0.045→0.039 | **0.00** | 0.73 | 0.0 | 4.07 |
| trifinger / cube / rotation (trial 0) | 92 | 1/1 | 73.8° | 0.030→**0.0338**→0.030 | **0.96** | 0.42 | **0.0913** | 0.0955 |
| fingertips / cube / in-air (trial 0) | 183 | 1/1 | 84.8° | 0.030→**0.101**→0.097 | 0.33 | 0.67 | 见表下 | 见表下 |

物体质量 0.01 kg，**mg = 0.0981 N**。

### 3.2 关键判据：桌面法向力 ≈ mg 就是「桌面在托着」

- **trifinger / cube，步 7–42**：`n_table = 4`，桌面法向力 = **0.0981 N**，
  与 mg **完全相等**；同时 `n_fingertip = 0`。
  → 前半段物体纯粹躺在桌面上，三根手指**一点都没碰到它**。
- **allegro / cube**：全程 `n_table = 0`（`frac_steps_airborne_no_table = 1.0`），
  物体由掌面 + 指节托住（220 次 palm 接触 vs 184 次指尖接触）。
- **fingertips / cube / in-air，步 126 起**：`n_table = 0`、`n_fingertip = 3`、
  非桌面法向力 0.41–1.04 N，物体 z = 0.084–0.097。
  → **物体真的离地了**，重力完全由三个点接触承担。

### 3.3 逐步骤剖面（这是本节最直接的证据）

trifinger / cube / trial 0（92 步）：

```
step    z    n_ft n_tbl  L_notbl  L_tbl
   0  0.0300   0     4    0.0000  0.0000
   7  0.0300   0     4    0.0000  0.0981   <- 桌面承担 mg，手指未接触
  21  0.0300   0     4    0.0000  0.0981
  42  0.0300   0     4    0.0000  0.0981
  56  0.0319   3     1    0.4532  0.0780   <- 手指终于接触，物体仍在桌上
  70  0.0297   1     3    0.1621  0.1988
  84  0.0302   4     1    0.3437  0.0626
```

fingertips / cube / in-air / trial 0（183 步）：

```
step    z    n_ft n_tbl  L_notbl  L_tbl
   0  0.0300   0     0    0.0000  0.0000
  36  0.0300   0     4    0.0000  0.0981   <- 先躺在桌上
  54  0.0300   0     4    0.0000  0.0981
  72  0.0318   1     1    0.0716  0.0843   <- 开始接触
  90  0.0486   2     1    0.7641  0.0855   <- 离地
 126  0.0427   2     0    0.5910  0.0000   <- 桌面接触消失
 162  0.0845   3     0    1.0386  0.0000   <- 三点悬空，载荷 ~1 N
 180  0.0972   3     0    0.8393  0.0000
```

---

## 4. 指尖触觉通道活了吗（F5 的问题在新设定下的重测）

用同一份 `motion_probe_v1` traces 统计（`docs/falsification/analyze_fingertip_channel.py`）：

| hand | zero_ft | zero_load | 平均指尖接触数/步 | 接触时非桌面载荷 |
|---|---|---|---|---|
| allegro / cube | 0.246 | 0.014 | 1.30 | 3.46 N（含掌面与指节） |
| trifinger / cube | **0.576** | 0.576 | 1.38 | 0.225 N |

- `zero_ft` = 完全没有指尖接触的控制步占比。
- trifinger 的 57.6% 来自 §3.3 那段「手指还没碰到物体」的前半程——**不是弱信号，是零接触**。
- trifinger 没有掌面，非桌面接触只能是指尖，所以 `zero_load` 与 `zero_ft` 相等
  （0.576）在算术上自洽，可作为交叉验证。

**对照 F5：** F5 在 allegro 上测到 12 个 nominal 状态里 8 个（67%）指尖读数为零、
而掌面承载约 1 N。本处的 allegro `zero_ft = 0.246` 是**滚动过程中的控制步**统计，
分母与采样点都与 F5 不同，两个数字不可直接比较；但两者指向同一件事：
**allegro 的承载主力不是指尖。**

---

## 5. 触觉接口的可移植性（会阻塞后续工作的一条）

`ours/tactile_observation.py::VirtualTactile` 把 Allegro 的传感器模型写死了：
必须是 `fingertip0..3` 四个 geom、`ftp_0..3` 四个 site、以及一个名为 `palm` 的 geom。
实测（`docs/falsification/check_tactile_portability.py`）：

| hand | `fingertip0..3` | `ftp_0..3` sites | `palm` |
|---|---|---|---|
| allegro | 4/4 | 4/4 | 有 |
| trifinger | **0/4**（实际叫 `fingertip_0/120/240`） | **0/4** | 无 |
| fingertips | **3/4**（只有 `fingertip0/1/2`） | **0/4** | 无 |

结论：**`VirtualTactile` 不能原样搬到 trifinger 或 fingertips**，一实例化就会抛
`ValueError`。要在新设定上继续用触觉特征，需要扩展这个模块（新文件，不改它），
或者改用 §4 那种 `motion_probe_v1` 的接触聚合口径。

**注意两者的区别：** `virtual_tactile_v2` 是「理想指尖力/位置传感」，带 object frame
下的力矩；`motion_probe_v1` 只是「接触归类 + 法向力大小」。不要混用，也不要把后者
的载荷数字当成前者。

---

## 6. 对研究计划的含义

1. **「换成像 Sharpa 一样多指抓住」在 FREE 里的正确入口是 `fingertips`**，
   不是 `trifinger`。trifinger 的官方任务目标 z 就等于初始高度（`params.py` 里
   `self.target_p_ = np.hstack([target_xy_rand, init_height])`），**设计上就是桌面上的
   旋转**，不是举起来把玩。
2. **`fingertips` 也不是手。** 它是 3 个点球 + 9 个滑移关节的抽象模型。它给出了
   「三点接触、无掌面、无桌面、悬空操纵」这个**结构**，但不提供真实手指的滚动、
   指节接触或抓握姿态。任何结论只能主张到「三点接触抽象」这一层。
3. **三手型的承载归属是三个不同的物理机制**（掌面 / 桌面 / 三点）。讨论「触觉能替代
   模型到什么程度」时，必须先声明是在哪一个机制下说的。
4. **触觉接口需要一次重写**（§5），这是换设定后第一个实打实的工程量。

---

## 7. 限定条件

- trifinger 与 fingertips 的数字来自 **1–2 条 trial**，只在 `cube` 上，**是 pilot**。
  全量扫描由并行的两个 agent 完成，不在本文档内。
- `motion_probe_v1` 的法向力是**按接触逐条求和**，不是带符号的净力；因此
  §3 里「谁承担重力」的主证据是 **z 轨迹 + 接触是否存在 + 桌面法向力是否≈mg**，
  载荷数字只作为次级指标。trifinger 步 7–42 的 0.0981 N 恰好等于 mg 是个强巧合级
  的旁证，但单靠它不足以定性。
- `contact.detect_once` 末尾的 `mj_collision` 会让 `efc_address` 失效、之后读
  `mj_contactForce` 一律得 0。观测器因此在每次读力前重跑 `mj_forward`（见脚本
  docstring）。这**不扰动** rollout：已用 `--hand allegro --trials 3` 与官方
  `examples.mpc.allegro.cube.test` 对拍，步数 49/62/31、quat_err 完全一致；
  `--hand trifinger` 同样对拍通过（92 步、0.0047）。
- 本文档不改动 FREE 的 `envs/ models/ planning/ contact/ utils/`，只新增
  `examples/mpc/ours/observe_task_motion.py` 与 `docs/` 下的内容。

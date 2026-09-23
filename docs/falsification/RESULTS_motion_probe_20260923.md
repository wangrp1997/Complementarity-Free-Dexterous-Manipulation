# 运动观测：TriFinger / Fingertips / Allegro 三环境对比（2026-09-23）

## 0. 这次回答什么问题

之前的结论都指向同一件事：Allegro 的 on-palm 任务里物体由掌面承载，67% 的状态指尖触觉读数为零。
于是提出两条替代路线——**TriFinger 指握**和 **Fingertips 纯触点**。

这次要回答的是：**换成这两个设定后，物体到底靠什么支撑、有没有真的离开支撑面运动？**

**姊妹文档：** `docs/falsification/F6_hand_structures.md`（手型结构审计 + 触觉接口可移植性 + 逐步骤剖面）。本文档是它的全量扫描版本（14 物体 × 3 trials + 三种目标类型），两者结论一致；F6 给出更强的单步证据，本文给出跨物体的分布。

## 1. 工具与口径

唯一工具：`examples/mpc/ours/observe_task_motion.py`（只读观测器，复用 FREE 自己的 MPC 循环与成功判据，
不修改 `envs/` `models/` `planning/` `contact/` `utils/`，不改任何 `params.py`）。
汇总脚本：`docs/falsification/aggregate_motion_probe.py` → `docs/falsification/out/motion_probe_aggregate.json`。

关键指标定义（冻结）：

| 指标 | 定义 |
|---|---|
| `frac_steps_with_table_contact` | 该 episode 中，物体 geom 与 `table` geom 存在接触的控制步占比 |
| `mean_table_normal_force` / `mean_non_table_normal_force` | 每步物体接触的**法向力之和**，按是否含 `table` 分类后在步上取平均（N） |
| `max_clearance_above_table_m` | `z_max - half_extent`，见 §5 的可靠性警告 |
| 空转成功（degenerate） | `steps <= 25` 且 `rot_net_deg < 5`：初态已满足成功阈值 |

执行命令（每物体独立进程，避免单物体崩溃中断整轮）：

```shell
PYTHONPATH=. /home/rw/miniconda3/envs/free/bin/python examples/mpc/ours/observe_task_motion.py \
  --hand trifinger --objects <obj> --trials 3 --output docs/data/motion_probe_trifinger_20260923/<obj>

PYTHONPATH=. /home/rw/miniconda3/envs/free/bin/python examples/mpc/ours/observe_task_motion.py \
  --hand fingertips --objects <obj> --trials 3 --all-targets --output docs/data/motion_probe_fingertips_20260923/<obj>
```

## 2. 结果表

`f_tbl` = 桌面接触步占比，`f_ftp` = 指尖接触步占比，`F_tbl`/`F_other` = 桌面/非桌面平均法向力（N）。

| hand | object | target | succ | 空转 | steps | rot_net° | f_tbl | f_ftp | F_tbl | F_other | ftip接触数 | palm接触数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| allegro | cube | rotation | 3/3 | 0 | 47.3 | 86.9 | **0.000** | 0.723 | **0.000** | 3.621 | 184 | 220 |
| fingertips | cube | **in-air** | 3/3 | 0 | 149.7 | 98.0 | 0.577 | 0.583 | 0.059 | 0.274 | 614 | 0 |
| fingertips | cube | ground-rotation | 3/3 | 0 | 156.0 | 116.7 | 0.896 | 0.615 | 0.099 | 0.327 | 658 | 0 |
| fingertips | cube | ground-flip | 3/3 | 0 | 127.7 | 133.1 | 0.714 | 0.556 | 0.076 | 0.264 | 480 | 0 |
| fingertips | bunny | in-air | 3/3 | 0 | 248.7 | 98.5 | 0.580 | 0.655 | 0.070 | 0.293 | 1023 | 0 |
| fingertips | bunny | ground-rotation | 3/3 | 0 | 115.3 | 118.5 | 0.857 | 0.475 | 0.092 | 0.175 | 361 | 0 |
| fingertips | foambrick | in-air | 3/3 | 0 | 232.0 | 98.5 | 0.454 | 0.698 | 0.052 | 0.343 | 1119 | 0 |
| fingertips | foambrick | ground-rotation | 3/3 | 0 | 124.7 | 119.5 | 0.853 | 0.537 | 0.122 | 0.234 | 427 | 0 |
| fingertips | foambrick | ground-flip | **2/3** | 0 | 791.7 | 129.4 | 0.803 | 0.772 | 0.150 | 0.684 | 5268 | 0 |
| fingertips | stick | ground-flip | 3/3 | 0 | 299.0 | 120.4 | 0.838 | 0.724 | 0.124 | 0.312 | 1351 | 0 |
| trifinger | airplane | rotation | 3/3 | 0 | 76.0 | 42.9 | 0.943 | 0.404 | 0.131 | 0.088 | 406 | 0 |
| trifinger | binoculars | rotation | 3/3 | **1** | 102.7 | 28.0 | 0.952 | 0.209 | 0.099 | 0.047 | 298 | 0 |
| trifinger | bowl | rotation | 3/3 | 0 | 53.0 | 36.0 | 0.778 | 0.382 | 0.077 | 0.143 | 382 | 0 |
| trifinger | bunny | rotation | 3/3 | 0 | 72.3 | 37.7 | 0.968 | 0.272 | 0.115 | 0.094 | 446 | 0 |
| trifinger | can | rotation | 3/3 | **1** | 75.3 | 32.0 | 0.931 | 0.194 | 0.096 | 0.055 | 265 | 0 |
| trifinger | cube | rotation | 3/3 | **1** | 68.2 | 43.0 | 0.971 | 0.285 | 0.094 | 0.056 | 308 | 0 |
| trifinger | cup | rotation | 3/3 | **1** | 60.0 | 30.9 | 0.922 | 0.215 | 0.095 | 0.066 | 238 | 0 |
| trifinger | elephant | rotation | 3/3 | **1** | 71.0 | 28.7 | 0.927 | 0.241 | 0.091 | 0.046 | 221 | 0 |
| trifinger | foambrick | rotation | 3/3 | 0 | 55.7 | 36.1 | 0.922 | 0.323 | 0.105 | 0.058 | 191 | 0 |
| trifinger | mug | rotation | 3/3 | **1** | 95.3 | 33.4 | 0.977 | 0.197 | 0.103 | 0.043 | 377 | 0 |
| trifinger | rubber_duck | rotation | 3/3 | **1** | 94.3 | 32.6 | 0.969 | 0.132 | 0.108 | 0.035 | 221 | 0 |
| trifinger | stick | rotation | 3/3 | 0 | 183.7 | 34.6 | 0.959 | 0.400 | 0.101 | 0.075 | 1047 | 0 |
| trifinger | torus | rotation | 3/3 | **1** | 56.0 | 35.7 | 0.946 | 0.394 | 0.101 | 0.100 | 354 | 0 |
| trifinger | water_bottle | rotation | 3/3 | **1** | 148.7 | 35.5 | 0.931 | 0.179 | 0.099 | 0.071 | 551 | 0 |

未完成：`trifinger/{camera, piggy_bank, teapot}`（见 §4）。

## 3. 结论

### C1. TriFinger 不是悬空在手操作，它是"压在桌面上搓"

14 个跑通的物体里，桌面接触步占比 **0.778 – 0.977（中位约 0.94）**。
平均法向力上，**桌面 0.077–0.131 N，非桌面 0.035–0.143 N**——多数物体桌面承担的载荷更大或相当。

也就是说，**FREE 论文里那个"TriFinger in-hand manipulation"，在这份代码的设定下主要是桌面辅助的转动**，
不是物体悬空被三指捏住把玩。这解释了为什么前面看到的指尖触觉弱信号问题在 TriFinger 上依然存在
（`f_ftp` 中位约 0.27，多数步指尖根本没接触到物体）。

### C2. Allegro 是三者中唯一真正悬空的

Allegro cube：`f_tbl = 0.000`、airborne 100%、平均法向力 **3.62 N**、palm 接触数 220。
载荷量级比 TriFinger 高 30–40 倍，且完全没有桌面兜底。

### C3. Fingertips 的 `in-air` 才是真正的悬空在手操作

`in-air` 目标下物体被举到 `z ≈ 0.09–0.10`（初始 0.03），抬升 5–7 cm。
但 `f_tbl = 0.45–0.58` 并不矛盾：物体**初态就搁在地面上**（`z_init = half_extent`），
episode 前段仍有地面接触，抬起后才脱离。三种目标类型里只有 `in-air` 是悬空的，
`ground-*` 两种顾名思义是地面辅助。

### C4. TriFinger 的成功率被"空转成功"抬高

42 个跑通的 TriFinger episode 中 **9 个是空转**：21 步、净转动 <5°、直接判成功。
根因已核实（见下），**all of them are `rand_seed=2`**：

```
trifinger seed=2: init_quat_err=0.02866 (< quat_th 0.04), init_pos_err=0.01648 (< pos_th 0.02)
trifinger seed=0: init_quat_err=0.42786 ; seed=1: 0.04601
allegro   seed=0/1/2: init_quat_err=0.50000（从不空转）
fingertips seed=0/1/2: init_quat_err=0.44/0.78/0.47（从不空转）
```

**TriFinger 的 `--trials 3` 里有整整 1/3 是在测"什么都不做算不算成功"，而答案算成功。**
这是 FREE 评测脚本的性质，不是 TriFinger 的任务性质，但对任何以 TriFinger 为基准的比较都成立——
引用 TriFinger 成功率时必须同时报空转比例。

### C5. FREE 的 TriFinger 接触封装在接触数超过 `max_ncon_=15` 时直接崩

`trifinger/{camera, piggy_bank, teapot}` 三个物体跑挂，报错原文：

```
File "contact/trifinger_collision_detection.py", line 119, in reformat
    jac_mat[4 * i: 4 * i + 4] = con_jac_list[i]
ValueError: could not broadcast input array from shape (4,15) into shape (0,15)
```

不是静默丢弃，是越界赋值崩溃。这三个物体的接触数超过了 `max_ncon_ = 15`。
**任何要在 TriFinger 上做实验的方案都必须先处理这个上限**，否则大约 18% 的物体会直接跑不了。

### C6. Fingertips 是唯一没有掌面/桌面歧义的执行结构

`envs/xmls/env_fingertips_*.xml` 中 `fingertip0/1/2` 是三个带 slide joint 的 **sphere（半径 0.01）**，
没有手指连杆、没有掌面。它就是"单纯触点接触"任务。接触统计中 `palm = 0`。

## 4. 如果要做 TriFinger 的触觉研究，先决条件

1. **必须换到 Fingertips `in-air`，或者自己构造一个真正悬空的 TriFinger 初态**（当前默认初态物体贴桌）。
2. TriFinger 的 `max_ncon_` 崩溃要先解决（不改 FREE 的前提下去做，只能避开高接触物体，或在观测层截断）。
3. 报 TriFinger 成功率时必须同时报空转比例（本批 9/42）。

## 5. 已知的口径缺陷（不要当成结论）

1. **`max_clearance_above_table_m` 对网格物体不可信。** 该量用 XML 中 `size=` 解析物体半尺寸，
   网格物体（airplane / bunny / mug / …）没有 `size=`，脚本回退到默认 0.03，
   于是出现负值（如 airplane −0.0143）。**只有 `frac_steps_with_table_contact` 和力分配这两个量在所有物体上都可靠。**
2. **`contact_class_counts` 的 `other` 分类**会把非指尖的手指连杆网格计入 `other`，不要读作"外来物体"。
3. **cube 的 trial 0 被计了两次**：冒烟测试目录 `docs/data/motion_probe_trifinger_20260923T013542Z` 与
   `.../trifinger_20260923/cube` 是同一 seed 的同一次运行，数字完全一致。汇总表里 cube 记为 4/4 而非 3/3。
4. 全部结论只基于 3 个 trial / seed，**是 pilot，不是基准**。
5. 观测器在每步额外调用一次 `mj_forward`。它对固定状态是幂等的，且 `detect_once` 本来就调用它，
   但严格说这仍是"观测改变了时序"的一个潜在来源，未做对照实验排除。

## 6. 对研究方向的影响

- 之前想把 TriFinger 当作"指握、无掌面兜底"的替代设定：**这个前提不成立**，它大部分时间压在桌面上。
- 真正满足"悬空 + 无掌面 + 纯接触点"的设定是 **Fingertips `in-air`**：物体被三个球举到 5–7 cm 高，
  没有掌面、没有桌面兜底，接触全部发生在三个触点上。
- 因此如果要检验"指尖触觉是唯一信息源"这个设定，**应该换到 Fingertips `in-air`**，而不是 TriFinger。

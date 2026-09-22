# 触觉采集修复与小规模审核记录

本轮由三个 agent 分别修复观察器、修复采集器、整理状态规范，主 agent 审阅实现并独立运行物理检查和数据审核。未训练模型，未实现 `ours` 优化器。

## 交付与保存位置

- 状态规范：[tactile_state_spec.md](tactile_state_spec.md)。
- 修订计划：[research_plan_differentiable_tactile.md](research_plan_differentiable_tactile.md)。原 `research_plan.md` 未改。
- 数据目录：`docs/data/tactile_probe_v2_review_20260920/`；每个 episode 有 `inputs.npz`、`labels.npz`、`metadata.json`。
- 最终机器审核：[audit_summary_reviewed.json](data/tactile_probe_v2_review_20260920/audit_summary_reviewed.json)。原始 `audit_summary.json` 保留，其中近零导数的相对差可能误导，复核版已改为只报告绝对量和幅值。
- `run.log` 保留执行记录；`review_manifest.json` 保存代码校验值和原文件保护结果。

代码位于 `ours/` 和 `examples/mpc/ours/`。当前分支为 `research/tactile-state-model`，未提交。逐一比较了 346 个原有受版本管理文件及原计划对应的校验记录，没有内容变化。项目现有规则忽略整个 `docs/`，所以文档及数据保存在本机，但没有因切分支自动得到 Git 保护；本轮未改忽略规则。

## 已修复的采集问题

1. 接触轴按 MuJoCo 的行方向解析；根据物体位于 geom1/geom2 统一受力与法向符号。
2. 四个指尖只覆盖 `fingertip0..3`；其他指节、掌面、地面不会冒充指尖触觉。全部有载荷接触保留，排除无激活及零载荷的近接触。
3. 力、位置、法向统一到物体 body/CAD 注册坐标；力矩关于 CAD 原点，不读取未知 COM 来定义观测。多点力矩独立求和，不用加权中心叉乘合力替代。
4. 观察在完整 `MjData` 副本上计算，不修改正在执行的仿真。保存动作前后触觉、全部接触、姿态、关节速度、时间与完整基准历史。
5. 真实物理参数、其他接触、有效摩擦、完整仿真状态放在 metadata/labels；`inputs.npz` 中动作来源等诊断字段也不属于默认控制特征。前后帧用于验证，在线控制只能读取当前及过去数据。
6. 采集脚本本地修复失配初始化，保留原 FREE helper。求解失败不执行对应动作；输出目录禁止覆盖。

## 两项影响旧结果解释的发现

**失配同时重置了初态。** 当前 MuJoCo 3.13.0 中，旧失配 helper 调用 `mj_setConst` 会把 `qpos` 设为 `model.qpos0`。cube seed 0 实测手指回到零位，最大关节变化 1.85 rad，物体也回到 XML 初始姿态。旧 nominal/mismatch 差异不能全部归因于动力学。本轮在改变物理参数前保存完整积分状态，之后恢复；正式数据的两模式初始物体/关节姿态完全一致。这里没有重跑完整成功率评测。

**修改物体摩擦不代表修改了指尖有效摩擦。** 物体 geom 滑动摩擦从 0.5 设为 0.3，但手指 geom 默认 1.0；正式样本的有载荷指尖接触有效滑动摩擦均为 1.0。此配置确实改变 COM 和惯量，不能声称已经验证指尖摩擦变化。COM 设置是指定新值，不是对原质心加偏移；所有实际参数已单独保存。

## 验证证据

- 倾斜四点接触、两种 geom 顺序及偏置 COM 场景：接触力/力矩映射所得广义力与 MuJoCo 约束广义力一致，最大误差小于 `9e-16`。
- 真实 cube 场景 400 个物理步每步插入观察，与不观察的同动作轨迹逐位相同，积分状态最大差为 `0`；原 warmstart 与约束缓冲不变。
- 正式数据逐条验证了多接触合力/力矩、单位法向、四元数、时间间隔、有限数值与求解状态。每个配对组的完整初始积分状态、观测、基准动作一致；基准扰动为零，正负动作对称，分支执行时长相同。
- 姿态差采用 SO(3) 对数/转角，检查了四元数正负等价；不用四元数分量差充当旋转角。

## 本轮正式数据

三个物体、两种物理条件、每种一个 seed 0，最多 30 个基准控制周期。要求至少两个有载荷指尖、无物体地面接触、当前关节及候选目标满足界。基准动作来自 FREE；扰动为一个关节的 `±0.002/±0.004 rad`，只选择两个符号、两档幅度都合法的方向。未选中快照也保留完整基准记录与排除原因。短时长和有限 seed 不构成成功率评测。

| 物体 | 条件 | 基准周期 | 合格配对快照 | 总动作记录 |
| --- | --- | ---: | ---: | ---: |
| cube | nominal | 30 | 0 | 30 |
| cube | unknown-dyn | 30 | 0 | 30 |
| mug | nominal | 30 | 1 | 34 |
| mug | unknown-dyn | 17 | 2 | 25 |
| stick | nominal | 30 | 1 | 34 |
| stick | unknown-dyn | 30 | 0 | 30 |
| 合计 | | 167 | 4 | 183 |

每个合格快照只剩一个通过边界筛选的关节方向，因此是一个基准加四个扰动分支。183 条记录不是 183 个独立实验，局部幅度比较只覆盖 4 个快照。mug mismatch 提前结束是达到 2 个配对快照，不是任务成功。

大量候选时刻只有 0–1 个有载荷指尖，或原 FREE 的实际关节/目标越过严格边界。当前环境存在指节、掌面支撑，四指尖传感不能默认覆盖全部操作接触。未为增加样本而把指节改名指尖，也未裁剪原 FREE 动作。单独的 cube 单指烟测只用于验证配对保存链路，未混入本表。

## 局部响应的初步观察

对两档幅度分别计算中央差分，再比较平移与旋转导数：

- mug mismatch 的第 16 个基准步：两幅度平移导数相对差约 5.7%，旋转约 2.6%，所采末态的指尖接触开关一致。这只支持该快照、该方向的局部近似候选；不能推出中间时刻未切换或全局光滑。
- mug nominal 第 8 步、mug mismatch 第 12 步：旋转导数相对差约 30.8%、50.6%，末态指尖接触开关也有分支差异。需要处理接触变化，不能把所有样本套入一个已成立的局部线性假设。
- stick nominal 第 7 步：所选关节 1 属于第 0 指，而起始有载荷指尖为第 2、3 指；测得的物体响应导数接近数值零。此方向没有提供有效的物体运动激励，不能用近零分母产生的大相对差声称非线性很强。

这不是触觉状态充分性证明，也没有比较 GNN 或训练模型。下一步应结合实际感知覆盖和接触关系，定义有任务作用的动作方向，再推导局部约束/在线响应；不宜直接扩大同类采集或凭这 4 个快照承诺论文性能。

## 复现

在项目根目录使用已有环境：

```bash
/home/rw/miniconda3/envs/free/bin/python -m examples.mpc.ours.check_tactile_observation
/home/rw/miniconda3/envs/free/bin/python -X faulthandler -u -m examples.mpc.ours.collect_tactile_dataset --objects cube mug stick --trials 1 --modes nominal unknown-dyn --max-steps 30 --probe-count 2 --min-fingers 2
/home/rw/miniconda3/envs/free/bin/python -m examples.mpc.ours.analyze_tactile_probes <新生成的数据目录>
```

采集默认生成带时间戳的新目录；分析结果也默认拒绝覆盖已有文件。一次正式命令在创建输出目录前以 `139` 崩溃，没有 traceback；随后帮助命令和开启 faulthandler 的完整采集均成功。启动崩溃原因尚未定位，不声称已修复。最终数据来自退出码为 `0`、且通过独立审核的完整运行。

# 冻结约定（所有 agent 与脚本必须遵守）

**日期：** 2026-09-22
**性质：** 冻结。本文件优先于 `docs/falsification/` 下所有其他文档。冲突以本文件为准。
**为什么存在：** 上一轮并行产出了五份重叠规范、四套互相矛盾的残差口径，同一帧在四套口径下分别是 1e-4 / 1e8 / 1e9。**任何不遵守本约定的数字一律不得进入报告。**

---

## 1. 权威文档层级

| 层级 | 文件 | 说明 |
| --- | --- | --- |
| **最高** | `CONVENTIONS.md`（本文件） | 口径与规则。冲突以此为准。 |
| 规范 | `falsification_spec.md` | L1–L4 判据的唯一权威出处。 |
| 规范附件 | `F0_falsification_plan.md` **§2**（阈值登记表）、**§4**（第三类"未判定"判定） | **仍然有效**，见 §1.1；该文件其余章节归档。 |
| 结果 | `RESULTS_pilot_20260922.md`、`F1_free_physics_audit.md`、`RESULTS_*.md`、`F<n>_*.md` | 已确认结论。 |
| 归档 | `README.md`、`L0_validity_audit.md`、`falsification_protocol.md`、`F0_falsification_plan.md` 其余章节 | **已被本文件取代，只作修订痕迹保留，不得再引用作为判据来源。** |

**禁止新增规范类文档。** 新的发现写进 `RESULTS_*.md` 或 `F<n>_*.md`；新的判据必须先改本文件。

## 1.1 为什么 F0 的 §2/§4 不是归档（修订记录见 §9.6）

本文件 §8.2 强制要求"每个阈值必须做 ×0.1/×1/×10 敏感性"，而唯一登记了这些档位的地方是 `F0_falsification_plan.md` §2；§3 要求的"第三类判定：未判定"（残差谱无间隙、半宽为零但秩可疑）同样只在 F0 §4 有判别规则。把 F0 整份判为归档，会让 §3 与 §8.2 变成没有实现的空条文——这是本文件自身的矛盾。

因此按范围拆分：**F0 §2 与 §4 升级为"规范附件"，优先级仅次于 `falsification_spec.md`**；F0 其余章节继续归档。

## 1.2 非合规脚本（其数字不得引用）

本文件 §3 要求"所有脚本必须调用同一个实现"，但以下四个脚本**各自定义了一套残差口径，均未 `import metrics.py`**，其历史输出不满足本约定：

| 脚本 | 不合规原因 | 历史产物 |
| --- | --- | --- |
| `check_core_relation.py` | 未 import `metrics.py`，无 `consistency`/`adequacy` 双栏 | `out/core_relation_check.json` |
| `check_discrimination.py` | 自建 `denom_j`/`denom_x` 口径 | `out/discrimination.json` |
| `check_single_step.py` | 自建口径，且除以 \|x_obs\| 的写法违反本文件 §3/§4 | `out/single_step.json` |
| `check_row_structure.py` | 自建口径 | `out/row_structure.json` |

**这四个脚本与它们的 JSON 只作修订痕迹保留。** 任何报告不得引用其数字；要引用必须先改写成 `metrics.py` 口径后重跑。它们已暴露的问题**仍然有效**（尺度失效、近静态帧相对量伪影、真值接触集常不满秩）——那些结论由多个互相不重叠的实现共同复现，并已由主 agent 独立复现。

---

## 2. 尺度：运动学关系只能用在小尺度上

**F1 已确认：FREE 是软接触力模型，预测器含物体质量与重力。** 理想刚性关系 $\|A_hx+J_h\delta q\|$ 测的是与 FREE 软模型的失配，不是接触模式。

**适用范围（重要）：** 本节约束的是**使用刚性接触运动学关系**（$A_hx+J_h\delta q$ 及其变体）的检查。**不使用时间导数、只在状态快照上做几何/接触枚举的检查不受本节约束**——见下面第 3 条。

因此：

1. **禁止**在控制区间（约 0.1 s / 0.2 rad）尺度上用该关系给假设打分。
2. 使用该关系的检查，**唯一允许的尺度是单物理步**（dt = `timestep` = 0.002 s）。
3. ~~**(b) 已有的 ε-探针分支**~~ —— **已作废**，见 §10.1。探针是小**幅度**扰动，不是小**时间尺度**测量：16/16 探针行的 `dt` 实测均为 0.1 s，与基准帧相同，同样跨过 50 个物理步。
4. **`state_snapshot`（新增第三类）：** 不涉及时间导数、只在单个状态快照上做几何/接触枚举的检查（如覆盖率闸门 `check_coverage.py`），允许声明 `scale: "state_snapshot"`，但**必须同时声明它不使用刚性接触关系**。
   **这第三类不得被借用来给任何涉及位移/速度/预测量的判据背书。** 它只覆盖"这个状态下有哪些几何接触"这类问题。
5. 任何脚本必须在输出里显式声明 `scale ∈ {single_step, state_snapshot}`；声明 `eps_probe` 一律作废。使用刚性关系的检查必须写明时间步长。

---

## 3. 单一残差口径

一个量只有一个定义。**所有脚本必须调用同一个实现**（`docs/falsification/metrics.py`）。

**`metrics.py` 本轮冻结：任何 agent 不得修改它。** 多个 agent 并行使用时改它等于所有人同时换口径——上一轮的教训正是这个。若发现它缺少你需要的量，写进报告的 `caveats`，**不要另起一套定义，也不要就地改它**。要新增量时先改本文件、再改 `metrics.py`，一次只做一件。

| 量 | 定义 | 分母 |
| --- | --- | --- |
| `residual` | $\|A_hx_{\text{obs}}+J_h\delta q\|$ | **无**（绝对量，声明单位） |
| `consistency` | 同上 | $\|J_h\delta q\|$（接触点自身运动尺度） |
| `adequacy` | $\|x_{\text{pred}}-x_{\text{obs}}\|$ | $\|x_{\text{obs}}\|$ |

**禁止除以 $\|x_{\text{obs}}\|$ 作为"拟合优度"。** 近静态帧上会炸到 $10^9$，那是实现陷阱不是世界性质。

**必须两侧同时报告** `consistency` 与 `adequacy`：只看残差会系统性偏向"删接触"的假设（删掉行残差必然下降）。

---

## 4. 未判定帧的处理

物体位移接近零的帧在信息上等于零，**无法区分任何接触假设**。

- 判据用**绝对量**：$|x_{\text{obs}}|<\epsilon_x$ 判为 `未判定`。
- `未判定` 帧**必须从 L1/L3 的统计里剔除**，并单独报告剔除数量。
- 禁止用相对量做剔除判据。

---

## 5. 秩的处理

真值接触集实测经常不满秩（cube nominal 前 8 帧秩为 3,6,5,6,6,6,6,6）。

- **禁止**用 `rank = 6` 丢弃假设——那会把真值假设本身丢掉。
- 必须显式记录 `null_dim = 6 - rank`。
- 所有涉及 $T_h$／预测位移的量，写成**条件性结论**（"在约束能确定的子空间上……"）。
- 零空间分量既不得假定为零，也不得假定有利；需要区间时报告 `[悲观, 乐观]`。

---

## 6. 物理声明

任何预测器必须在输出里显式标注一行：

```
USES: object_dynamics   或   USES: none
```

用 FREE 的 `ExplicitModel` 作参考预测器时必须是 `USES: object_dynamics`，因为它的 `b` 含质量与重力。**不得把带动力学的预测器的成绩记在"无动力学"名下。**

---

## 7. 输出契约

所有脚本只读，产物写到 `docs/falsification/out/`。**不得修改 FREE 任何文件**（`envs/`、`models/`、`planning/`、`contact/`、`utils/`、`envs/xmls/`、`params.py`）。

每个 JSON 必须含这些字段：

```json
{
  "check": "L1|L2|L3|L4|coverage|oracle|attribution",
  "scale": "single_step|state_snapshot",
  "convention": "CONVENTIONS.md@2026-09-22",
  "uses": "none|object_dynamics",
  "n_frames_included": 0,
  "n_frames_undetermined_excluded": 0,
  "epsilon_x": 0.0,
  "thresholds": {"name": {"value": 0.0, "provenance": "怎么来的"}},
  "verdict": "pass|fail|undetermined",
  "caveats": []
}
```

**`thresholds.*.provenance` 不得为空。** 阈值只能来自：数值精度、模型误差（实测跟踪误差 34.7%–51.4%）、离散化（接触点集直径中位数约 0.047 mm）。拍脑袋的阈值一律作废。

---

## 8. 报数纪律

1. 每个数字必须带：口径、尺度、单位、样本量、剔除量。
2. 阈值必须做 $\times 0.1 / \times 1 / \times 10$ 敏感性；**结论随阈值翻转 = 结论不成立**，必须写出来。
3. 只报成功样本 = 没做。必须报告全部帧的通过/不通过/未判定。
4. `pilot`（只有 cube nominal 前 10 帧）不得写成结论。
5. 每一级的负面结论必须回答：**这个负面排除了多少搜索空间？** 只排除"本次实现"的是日志，不是结论。

## 9. 第 R2 轮：分工、文件所有权与接口冻结

**日期：** 2026-09-22
**为什么存在：** 上一轮并行的失败原因不是切分错了，而是**没有预先冻结接口**——三个 agent 各写了一份规范、四套口径、同一个输出文件。本轮先冻接口再开工。

### 9.1 文件所有权（硬边界）

每个 agent **只允许写**自己名下的文件。写入他人名下的文件视为破坏。

| 线程 | agent | 独占可写路径（**本表与真实分配一致，不是理想计划**） |
| --- | --- | --- |
| 覆盖率闸门（= 本文件 §9.4） | `falsif2_coverage` | `docs/falsification/check_coverage_gate.py`、`docs/falsification/out/coverage_gate.json`、`docs/falsification/RESULTS_coverage_gate.md` |
| 文献线 | `falsif2_lit` | `docs/task_relevant_info_lit.md` |
| 探针信号线（ε-探针尺度 L1） | `falsif2_consolidate` | `docs/falsification/check_L1_probe.py`、`docs/falsification/out/L1_probe.json`、`docs/falsification/RESULTS_L1_probe.md` |
| 规范与合并 | `falsif2_consolidate` | `CONVENTIONS.md`；`RESULTS_*.md` 的合并 |

**对齐说明（2026-09-22）：** 本节初稿写的 A1/A2/A3 路径与三条线程**实际收到的任务书不一致**，而任务书已在派发时生效。按"文档必须描述现实"的原则，本表改为记录真实分配。三条线程的产出路径互不重叠，无冲突。

**任何 agent 不得新增规范类文档。** 判据冲突一律回报主 agent，由主 agent 改本文件。
**任何 agent 不得修改 FREE 任何文件**（`envs/`、`models/`、`planning/`、`contact/`、`utils/`、`params.py`）。

### 9.2 判据来源（消除 L1 的判据真空）

`falsification_spec.md` 附录 B.4 已判定：**旧的全局残差判据 `||A_h x + J_h dq||` 作废**
（它在任何时间尺度上都没有判别力，且系统性偏好更松的约束）。

因此 L1 **不得**直接复用旧判据。允许按 B.3 列出的重建方案做，优先级：

1. **方案 1**：逐接触瞬时判据 `c = jac_c @ qvel`，按 `STICK / SLIDE / FREE` 分层判定，**不做**全局残差最小化。
   注意它需要物体瞬时速度（`labels.object_velocity_mujoco`），属 **PRIVILEGED**：
   该判据的**诊断价值**与**可部署性**必须分开陈述，不得混为一谈。
2. **方案 2**：准静态力平衡，**必须显式给出有效域**（按 `|object velocity|` 分桶），域外帧剔除并计数。

**先证明判据本身有判别力，再谈 L1 结论。** 判别力证据的最低要求：
在真值接触集上，判据必须能把真值模式与**刻意扰动**区分开（例如把某个 STICK 改写成 SLIDE/FREE，
或注入伪接触），并报告扰动前后的判据值分布。**做不到就是判据无效，L1 结论不得发表。**

### 9.3 L1 产物接口（冻结；下游 L2/L3 依赖此格式）

**权威格式 = JSONL**（不是 npz）。路径：`docs/falsification/out/l1/compatible_hypotheses.jsonl`

每行一个基线帧，字段：

```json
{
  "episode": "allegro_cube_nominal_trial000",
  "frame_index": 0,
  "source_step": 0,
  "scale": "single_step|state_snapshot",
  "undetermined": false,
  "criterion": "B3_plan1|B3_plan2",
  "n_candidates": 0,
  "n_compatible": 0,
  "hypotheses": [
    {
      "id": "h0",
      "contacts": [{"geoms": [1, 7], "mode": "STICK|SLIDE|FREE", "source": "perceived|geometric_candidate"}],
      "rank": 6,
      "null_dim": 0,
      "is_true_set": false
    }
  ],
  "covers_truth": true,
  "privileged_used": ["labels.object_velocity_mujoco"]
}
```

约束：

- `hypotheses` **必须**包含 `is_true_set: true` 的项（若该帧真值集在相容集内），否则 `covers_truth: false`。
- **禁止**用 `rank < 6` 丢弃假设；`rank` 与 `null_dim` 必填（本文件 §5）。
- 未判定帧（`|x_obs| < EPS_X`）必须写入 JSONL 且 `undetermined: true`，**不得静默剔除**。
- 同时输出 `docs/falsification/out/l1/summary.json`，含 `check/scale/convention/uses/n_frames_*/thresholds/verdict/caveats`（本文件 §7）。

### 9.4 覆盖率闸门是 L1 的第一道闸

候选假设空间若不能覆盖真值接触（96.4% 的帧含未覆盖载荷接触，这道闸很可能不过），
则 L1 的任何"无歧义"结论**作废**，必须先修表示。
覆盖率结果单独写进 `L1_report.md` 的**第一节**，不要埋在附录。

### 9.5 结果纪律（本轮补充）

1. **判据先于结论。** 没有 9.2 的判别力证据，不得写"L1 通过"。
2. **pilot 不得升格。** 仍只有 cube nominal 前若干帧时，结论只能写成"在 N 帧上的观察"。
3. **正面结果要能被打死。** 任何"存在歧义"的结论必须附带：换成更松的判据阈值后是否还成立
   （x0.1 / x1 / x10 敏感性，部分仍适用）。
4. **回报格式统一**：每个 agent 的最终回报必须含——产出文件绝对路径 / 判定 pass-fail-undetermined /
   关键数字（带样本量与剔除量）/ 用了哪些 PRIVILEGED 字段 / **这个结论排除了多少搜索空间** /
   哪些方面你的结论**不**成立。

### 9.6 本轮修订记录（falsif2_consolidate）

**2026-09-22（falsif2_consolidate）** —— 消掉本文件的内部矛盾，不新增规范文档：

1. §1 表格拆分：`F0_falsification_plan.md` 从"整份归档"改为"**§2/§4 升级为规范附件，其余归档**"，并新增 §1.1 说明理由（§8.2 的阈值三档与 §3 的"未判定"判定只有 F0 提供了实现，整份归档会让这两条变成空条文）。
2. 新增 §1.2：登记四个未 `import metrics.py` 的脚本为**非合规口径**，其历史 JSON 不得引用；同时声明它们暴露的问题仍然有效。
3. §3 增加 `metrics.py` **本轮冻结**条款。
4. **未**改动任何判据、阈值或既有结论；**未**改动 `falsification_spec.md`、`F0`、`metrics.py` 或任何脚本；**未**删除任何文档。

**遗留（明确登记，未解决）：** `F0_falsification_plan.md` 附录的 `sha256` 指纹仍为 `<待补>`——补指纹需要冻结正文，而正文仍被并行 agent 使用。

5. **（第二轮合并，2026-09-22）** 收尾别人写入的"第 1 号修正"。该节原以 `## 9.` 编号，与本文件已有的 §9 冲突，文档一度出现**两个 `## 9.`、两套 9.1–9.6**；已重编号为 **§10**（10.1–10.6）。同时**就地改写 §2 正文**：原 §2 仍把 `eps_probe` 列为允许尺度，修正只在文末单方面宣告，读者只读 §2 会得到相反信息。并把该节证据归属改为指向真实产物，原写法把第一手证据记在 `falsif2_coverage` 名下，与仓库实际产物不符。

6. **（判据级新增，2026-09-22）** 增补 `scale` 第三类 **`state_snapshot`**，回应 `RESULTS_coverage_20260923.md` §6：覆盖率闸门是**状态快照级的几何检查**，不使用刚性接触关系，原 §2 的二分类装不下它。为避免该类别被滥用，同时写明：它**不得**给任何涉及位移/速度/预测量的判据背书。§2 的适用范围限定为"使用刚性接触关系的检查"，§7 契约的 `scale` 枚举同步为 `single_step|state_snapshot`。

**本节未改动任何判据、阈值或既有结论；未改动 `falsification_spec.md`、`F0_falsification_plan.md`、`metrics.py` 或任何脚本。**

---

## 10. 第 1 号修正（2026-09-22）

### 10.1 §2 的尺度 (b) 作废

原 §2 允许两种尺度：single_step 与 eps_probe。**eps_probe 作废**（§2 正文已就地改写，不再是文末单方面宣告）。

证据（**第一手 = 本仓产物**：`RESULTS_L1_probe.md` / `check_L1_probe.py` / `out/L1_probe.json`，其中 `probe_span_equals_baseline_span = true`，16/16 探针行 `dt` = 0.1 s；**第二手**：`falsif2_lit` 独立复核六个 episode 的 metadata 后确认）：

- 六个 episode 的 metadata 实测 physics_timestep = 0.002、frame_skip = 50，即每条记录跨 **0.1 s**。
- 探针分支与基准帧的时间跨度**完全相同**：探针是小**幅度**扰动，不是小**时间尺度**测量，同样跨过 50 个物理步。

推论：**现有 183 行数据里，没有任何一行处在模型可能有效的尺度上。** 因此 L1 无法在现有数据上完成。

后果：要跑 L1，必须先按**单物理步**重新采集（记录相邻 mj_step 之间的位移），这是新的前置条件。

### 10.2 §2 的唯一有效尺度

只剩 single_step（dt = physics_timestep = 0.002 s）。任何脚本声明 scale: eps_probe 一律作废。

### 10.3 同时作废的两条

1. **"把 L1 搬到 ε-探针即可解决尺度问题"** —— 排除。探针不改变尺度。
2. **"±0.002/0.004 rad 探针默认满足一阶线性"** —— 逐快照实测否掉（不对称最高 46%；幅度标度最低 0.85，即更大扰动给出更小响应）。4 个配对快照里 1 个无信号、1 个非线性，样本量撑不起统计。

### 10.4 探针数据的另一个硬限制

探针只覆盖 6 个 episode 中的 3 个，只探过 joint 1 与 joint 13。**不得据此做任何统计推断。**

### 10.5 需要更新认知的一条

用固定阈值把真值接触分成 STICK/SLIDE 的做法**被排除**：三档 provenance 下 STICK 比例从 4.3% 跳到 43%，且直方图无空隙。接触模式不能靠单阈值切分。

### 10.6 写作纪律（文献线结论带来的）

- **"无需物体动力学"不得作为卖点。** 概念上 dual control 自 1960 年代即为中心问题；同一实验室的 Adaptive CI-MPC（ICRA 2024）已做过接触隐式 MPC + 在线残差学习；且 F1 已证明 FREE 的预测器含质量与重力。
- **"task-relevant"不得作为差异化的词。** Tishby 的信息瓶颈已形式化 relevant information，Bajcsy 1988 是整片领域，且该词**直接出现在 WM-Craftnet 原文里**。
- 必须正面处理的两篇最接近工作：TANDEM（arXiv:2203.00798，触觉探索与决策联合优化，但任务是触觉物体识别）、arXiv:2210.13403（在手操作+触觉+任务驱动+探索/完成权衡，但探索的是全局物体形状，不是"会不会改变下一步动作"）。
- 目前唯一值得写成 claim 的缝隙：**存在一类接触歧义，其区分动作在可执行动作集内不存在**（单步不可分）。这是**界**不是方法，且 L4 挂掉时自动升级为主结果。

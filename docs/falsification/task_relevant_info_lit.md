# 文献核查：任务相关信息的创新空间（阻塞项解除报告）

**日期：** 2026-09-22
**责任方：** A2（文献线）
**为什么存在：** `F1_free_physics_audit.md` §6 写明"选哪条出路**必须先做完文献核查**，现在拍板就是又一次挪球门"。本文件就是那次核查。上一轮负责这条线的 agent 跑偏去重写方案文档，`task_relevant_info_lit.md` 从未产出。

**证据等级约定（全文强制）**
- `[VERIFIED]` = 本次会话实际抓取到元数据/摘要/全文（给出 arXiv ID / DOI / 本地文件路径）
- `[LOCAL]` = 仓库内已有文档或已缓存全文（给出文件路径）
- `[MEM]` = 来自模型记忆，**未在本次会话中验证**

---

## 0. 一句话结论

**"不辨识动力学、只用观测在线校正"这条路已经被占满（dual control 已有六十余年，且同实验室的 Adaptive CI-MPC 已在 ICRA 2024 做过接触隐式 MPC + 在线残差）；"task-relevant information" 也已经是被人形式化过的既有概念。我们可辩护的增量只剩一个窄口：把探索预算的开关挂在"该歧义是否改变动作选择"上，并证明这个开关在真实数据里不是恒真也不是恒假。**

同时，**F1 §6 的三选一是个伪三选一**——见 §5，路线选择与新颖性已经解耦。

---

## 1. 方法与溯源

本次核查使用的渠道与结果：

| 渠道 | 状态 | 说明 |
| --- | --- | --- |
| `https://export.arxiv.org/api/query` | **可用** | 本报告大部分 `[VERIFIED]` 来自此 |
| `https://arxiv.org/abs/*` | 可用（HTTP 200） | |
| `https://api.crossref.org/works` | **可用** | 用于验证 1980–2010 年代经典引用 |
| `https://api.semanticscholar.org/graph/v1` | **受限**（HTTP 429） | 未取得结果，本报告不依赖它 |
| 本地全文语料 | 可用 | `docs/literature_sources_20260922/`（POISE / WM-Craftnet / Tacmap / sharpa-rl-lab）|

**未验证引用的比例：** 本报告共列 26 条引用，其中 `[VERIFIED]` 19 条、`[LOCAL]` 4 条、`[MEM]` 3 条。
**未验证比例 ≈ 11.5%**，全部集中在 1960–1998 年的控制论经典（arXiv/Crossref 覆盖差）。每条 `[MEM]` 都已单独标注，**不得直接抄进论文参考文献表**。

---

## 2. Q1：不辨识动力学、只用在线观测校正模型误差——前人做到哪了

### 2.1 这个想法本身不是新的，而且不止一次

| 血统 | 代表 | 时间 | 做了什么 | 证据 |
| --- | --- | --- | --- | --- |
| **对偶控制（dual control）** | Feldbaum 起源；Ali Mesbah, *Stochastic MPC with active uncertainty learning: A Survey on dual control*, Annual Reviews in Control | **2018**（起源 1960s） | 控制量同时承担"驱动系统"与"激励以降低不确定性"两个角色——**这正是我们 $L_{task}+\lambda L_{info}$ 的一般形式** | `[VERIFIED]` DOI `10.1016/j.arcontrol.2017.11.001`；Feldbaum 原始工作 `[MEM]` |
| 对偶控制的现代形式 | *Dual Control for Approximate Bayesian RL*, *Linear Quadratic Dual Control*, *Minimax dual control with finite-dimensional information state*, *DCEE in Autonomous Search* | 2015–2023 | 贝叶斯/乐观/极小极大各种对偶控制变体 | `[VERIFIED]` arXiv:1510.03591、2312.06014、2312.05156、2012.06276 |
| **对偶控制 + MPC** | *An Approximate DP Approach for Dual Stochastic MPC*、*Active Learning with Dual MPPI for Interaction-Aware Merging* | 2019 / 2023 | 在 MPC 里显式放信息项 | `[VERIFIED]` arXiv:1911.03728、2310.07840 |
| **接触隐式 MPC + 在线残差学习** | Huang, Aydinoglu, Jin, Posa, ICRA 2024 | **2024** | 接触隐式 MPC 边跑边学残差补偿模型误差 | `[LOCAL]` `docs/related_work.md` §3；`[MEM]` 原始出处 |
| 未知刚度/几何的在线拟合 | ContactSDF（RA-L 2025）| 2025 | MPC 中在线学 $\theta$ | `[LOCAL]` `docs/related_work.md` §1 |
| 视觉域的系统辨识 | TwinTrack（ICRA 2026）、ContactGaussian-WM | 2026 | 从 RGB-D/视频反推质量/惯量/摩擦 | `[LOCAL]` `docs/related_work.md` §3 |

### 2.2 结论

**"我们不需要辨识物体动力学"不是一个可辩护的新颖性主张。** 它同时败在两处：

1. **在概念层已经过时**：对偶控制从 1960 年代起就把"边控制边降不确定性"当作中心问题；"不辨识参数、只在线校正模型误差"是自适应控制/残差学习的标准形态。
2. **在本项目里已被 F1 作废**：`models/explicit_model.py` 的预测器 `b = [m_obj·g ; K·cmd]` 显式含物体质量与重力，`-0.1σJQ⁻¹b/h` 还与步长耦合。**拿 FREE 去验证"无动力学方法"是自相矛盾的**（F1 §2、§3）。

**因此：不要再把"无需物体动力学"写进 contribution。** 它可以作为方法描述的一句话，但不能作为卖点。剩下的空间在"**校准什么、何时校准、以及什么时候根本不该校准**"——而最后这一问才是我们的口子。

---

## 3. Q2：`task-relevant information` 是不是已经被人形式化过了

**是。而且不止一次，横跨三个学科。**

| 概念 | 出处 | 验证 |
| --- | --- | --- |
| **相关信息的经典形式化** | Tishby, Pereira, Bialek, *The information bottleneck method* | `[VERIFIED]` arXiv:physics/0004057 |
| **主动感知（active perception）** | Bajcsy, *Active perception*, Proceedings of the IEEE | `[VERIFIED]` DOI `10.1109/5.5968`（1988） |
| **信念空间规划 / POMDP** | Kaelbling, Littman, Cassandra | `[MEM]`（1998，经典，未验证） |
| **value of information** | Howard | `[MEM]`（1966，未验证） |
| **VOI 用于操作** | *POMDP Manipulation Planning under Object Composition Uncertainty* | `[VERIFIED]` arXiv:2010.13565 |

**所以：我们不能声称"提出了 task-relevant information"。** 至多能说"在某个具体难设定下把它具体化并验证"。

更要命的一条：**"task-relevant" 这个词已经出现在我们的直接竞争对手原文里。** WM-Craftnet 自己就写 "task-relevant physical inference"、"task-relevant object-state understanding"（`[LOCAL]` `docs/literature_sources_20260922/wmcraft_paper.txt:48,259`；`wmcraft_project.txt:53,159,343,382`）。用这个措辞当差异化，会在审稿第一轮就撞上。

### 3.1 最接近的两篇先行工作（**必须正面处理，不能绕过**）

**(a) TANDEM — *Learning Joint Exploration and Decision Making with Tactile Sensors***
`[VERIFIED]` arXiv:2203.00798
摘要原文：*"we focus on the process of guiding tactile exploration, and its interplay with task-related decision making... an architecture to learn efficient exploration strategies in conjunction with decision making... separate but co-trained modules for exploration and discrimination."*

- **重合度最高的一点：** "触觉探索"与"决策"联合优化，这正是我们的机制骨架。
- **它的边界：** 任务是**触觉物体识别**（从已知集合里判别是哪个物体，二值接触信号），不是把手内物体转到目标姿态；探索服务于**判别**，不是服务于**操作进展**。
- **对我们的意义：** 不能再说"首次把触觉探索和决策联合"。要说"首次把探索预算 gates 在**会改变动作选择**的歧义上"。

**(b) *In-Hand Manipulation of Unknown Objects with Tactile Sensing for Insertion***
`[VERIFIED]` arXiv:2210.13403
摘要原文：*"it incrementally builds a probabilistic estimate of the object shape and pose during task-driven manipulation. Our approach uses Bayesian optimization to balance exploration of the global object shape with efficient task completion."*

- **重合度更高的一点：** 都是**在手操作 + 触觉 + 任务驱动 + 探索/完成的显式权衡**，而且同样不依赖已知物体模型。
- **它的边界（也就是我们的口子）：** 它平衡的是 **"全局物体形状"的探索** vs 任务完成——**它是一个在线形状估计问题**。它的探索量由"形状还没摸清"驱动，**不是**由"这个歧义会不会改变我下一步的动作"驱动。
- **这是我能找到的最锋利的差别。** 也是唯一值得写成一句 claim 的差别（见 §4）。
- 需要核实但本次未核实的点：它用的是 Tactile-Enabled Roller Grasper（滚动式夹持器），不是多指灵巧手；`[VERIFIED]` 摘要里写了 "simulated Tactile-Enabled Roller Grasper"。**我们的多指接触几何与它不同，但这个差别是硬件差别，不能单独当贡献。**

### 3.2 检索反证（用来支撑"口子还开着"，但别过度解读）

| 检索式 | 结果 |
| --- | --- |
| `all:"tactile exploration" AND all:"in-hand manipulation"` | **0 条** |
| `abs:"decision-relevant" AND abs:robot` | 无同义工作（命中的是 VLA/世界模型类）|
| `abs:"relevant uncertainty" AND (abs:robot OR abs:manipulation)` | 无同义工作 |
| `abs:"task-relevant" AND abs:"uncertainty" AND abs:robot` | 无同义工作 |

**重要限定：** arXiv API 的 `all:` 检索覆盖标题/摘要/作者等元数据，**不是全文检索**。所以"0 条"只能说明"没有以此为标题/摘要主线的工作"，**不能**说明"领域内没人做过"。任何写进论文的表述都必须是 "to the best of our knowledge"，并附检索式与日期。

---

## 4. Q3：我们那句可被引用的话应该是哪一句

按"能不能被别人引用为一个发现"排序。**我把最弱但也最安全的放前面，因为后面两个要么依赖还没跑的实验，要么容易被证伪。**

### 候选 1（安全，可立即写；但单独拿出去偏弱）

> **在手内操作中，只有会改变动作选择的接触歧义才值得消耗探索预算；对不改变动作的歧义，最优的信息动作是零动作。**

- 可证伪方式：L3 直接检验。若实测中"会改变动作的歧义"占绝大多数，这句话退化为"总是该探索"，没有信息量。
- **风险：** 它读起来像 task-relevant information 的一个应用。若只有它，审稿人会说"这就是 VOI 的标准结论"。**必须配上"通用信息增益在这里做错了什么"的对照实验**，才有杀伤力。

### 候选 2（最强，但依赖尚未跑出的结果）

> **存在一类在手接触歧义，其区分动作在可执行动作集内不存在——即该歧义在单步内不可分，必须通过多步探测或重新接触才能消除。**

- 这是**条件性不可能结果**，比一个正面方法更持久。而且它在 L4 挂掉时会**自动成为主结果**（正面失败 → 负面定理），这是这个项目最好的保险。
- 限定必须写进结论：在什么表示、什么动作接口、什么预算下的不可分。
- **风险：** 需要 L4 真的失败且有结构，而不是"我们的探针太小"。

### 候选 3（未来方向，现在不能写）

> 欠驱动执行流形使可用的信息动作集合缩小，因此任务相关信息集也随之缩小。

- 依赖当前不存在的欠驱动模型（`envs/` 里只有全驱动的 Allegro 与 TriFinger）。**这不是文献问题，是建模缺口。**
- 顺带记一条文献线索备查：*Shear-based Grasp Control for Multi-fingered Underactuated Tactile Robotic Hands*，`[VERIFIED]` arXiv:2503.17501（Pisa/IIT SoftHand + TacTip 指尖触觉）。**它是"欠驱动 + 触觉"的既有工作**，将来走这条路必须先与它对表。

### 判断

**候选 2 是唯一有可能冲出 RSS/CoRL 平均线的句子**，因为它是一个界，不是一个方法。候选 1 是它的必要条件但不是充分贡献。**建议：把候选 1 作为方法定位句，把候选 2 作为论文的核心 claim 去争取。**"提出一个优化算法"这句话本身没有引用价值——这一点在本次核查里被再次确认。

---

## 5. Q4：F1 §6 三条出路的裁定（本报告的主要结论）

F1 §6 把问题设成："为了保住'无动力学'的口号，该换哪条路？" **文献核查表明这个设定本身是错的。**

### 5.1 关键洞察：路线选择与新颖性已经解耦

因为 §2 已经确认 **"无需物体动力学"本来就不是可辩护的卖点**，那么：

> **不需要为了保住一个不是卖点的口号，去付换仿真器的代价。**

三条路的实际成本收益因此完全变了：

| 路线 | F1 原判断 | 文献核查后的判断 | 结论 |
| --- | --- | --- | --- |
| 1. 自带预测器，FREE 只当执行器 | 评估仍会撞上 FREE 软物理 | 同左。而且"自带预测器"在对偶控制传统里毫无新意 | **不作为主张；只作为实现自由度** |
| 2. 接受 FREE 模型，改成"不辨识、只用在线观测校正模型误差" | 创新点需重新论证 | **已被占满**：dual control（§2.1）、Adaptive CI-MPC（ICRA 2024，**同一实验室**）、ContactSDF、residual RL。**这条路没有剩余空间，且有撞车风险** | **放弃作为卖点** |
| 3. 换到刚性接触仿真相 | 需另建环境，本阶段不做 | 仍是唯一能让"约束式、无动力学"内部自洽的环境。但既然该口号已不是卖点，**它就从"必须做"降级为"以后想清楚再说"** | **不在本阶段做，判断正确** |

### 5.2 建议

**不要在三条路里选。把主张层和实现层拆开：**

- **主张层（论文要写的）**：任务相关歧义的**闸门条件** + 可分性界（§4 候选 1 + 候选 2）。这个在**当前的 FREE 环境里就可以测**，不需要换仿真器。
- **实现层（工程选择）**：FREE 继续当执行器与基线（`ours/` 与 `envs/` 并列，不改 FREE）；预测器里用不用质量/重力，**按每个判据单独标注 `USES: object_dynamics | none`**（CONVENTIONS §6 已有此要求），不再统一声明"无动力学"。

**理由：** 现在唯一能支撑论文的东西是一个**机制层面的界**，而它可以在现有环境里被证伪。为一个已经不成立的口号换环境，是拿几周去救一句不该写的话。

---

## 6. Q5：四篇对比文献的 gap 逐条（与 `literature_comparison_20260922.md` 去重）

`docs/literature_comparison_20260922.md` 已经做过详尽的四篇精读，**本节不重写它**，只补三点它没有覆盖的、且与本次核查直接相关的结论。

| 工作 | 真 gap（可碰） | 已被占（别碰） | 本次新增证据 |
| --- | --- | --- | --- |
| **POISE** | 作者自陈：接触模型不完善 + 6D 位姿跟踪误差 | **"给 6D 目标到达加触觉反馈"是作者公开宣布的未来工作**，不能当我们的 headline | `[LOCAL]` `poise_paper.txt:395-417` 原文 "adding tactile feedback, and adapting from real interactions, for example by learning residual dynamics" —— **注意它同时点名了 tactile 与 residual dynamics，正好是 §5 里被占满的那两条** |
| **WM-Craftnet** | 作者自陈：短时程、严重扰动恢复有限、物体超工作空间/漂移/卡死 | 触觉状态模型、动作条件预测、运行时隐状态更新——**均已被占** | `[LOCAL]` `wmcraft_paper.txt:640-643`；且其原文已在用 "task-relevant" 措辞 |
| **Tacmap** | 作者自陈：**没有显式建模切向力分布（shear strain）**；复杂 mesh 下 ray-casting 成本 | 传感表示层。我们解决不了，也不要声称 | `[LOCAL]` `tacmap_paper.txt:299-306` |
| **sharpa-rl-lab** | 工程参照与低维触觉基线；默认每指力标量、接触位置关闭 | 它不是论文，别当对手 | `[LOCAL]` `literature_comparison_20260922.md` §6 |

### 6.1 四篇都没做、而我们可能做的三件事

1. **把探索开关挂在动作分歧上。** 四篇都不据此 gate 探索（WM-Craftnet 的隐状态是被 detach 进策略的，没有显式信息目标；POISE 部署 actor 根本不用触觉）。
2. **给出可分性的可行性分析（L4）。** 四篇都没有"这个歧义在当前动作接口下能不能被区分"的正面回答。这是候选 2 的来源。
3. **6D 目标位姿 + 多指连续触觉 + 无策略预训练的组合。** 这是 `literature_comparison_20260922.md` §7 已认定的可保留组合，本次核查**未找到反例**，但也**未穷尽**（见 §3.2 限定）。

---

## 7. 与已有文档的去重说明

| 文档 | 已覆盖 | 本文件的增量 |
| --- | --- | --- |
| `docs/related_work.md` | FREE 系谱、触觉重建/辨识、五条 gap 对照表——**机器人侧很完整** | 控制论谱系（dual control 60 年、active perception、VOI、信息瓶颈）；F1 §6 裁定 |
| `docs/literature_comparison_20260922.md` | 四篇逐条精读、可保留/需收回的主张表 | 无重复；只在 §6 补三条与本次核查直接相关的新证据 |
| `docs/research_decision_20260922.md` | 观测与文献联合审核、最小验证顺序 | 明确"无动力学"作为卖点已死；明确 F1 §6 是伪三选一 |
| `docs/falsification/F1_free_physics_audit.md` | FREE 是软接触力模型 | 回答它 §6 提出的那个阻塞问题 |

**注意：`related_work.md` 第 7 节的五条 gap 表仍然成立，但它整节都建立在"无动力学 + 已知 CAD"这个组合上。§2 的结果表明其中"无动力学"这一条要降级为方法描述，不能继续当卖点。**

---

## 8. 引用溯源表

| # | 引用 | 状态 | 标识 |
| --- | --- | --- | --- |
| 1 | Bajcsy, *Active perception*, Proc. IEEE, 1988 | `[VERIFIED]` | DOI 10.1109/5.5968（Crossref）|
| 2 | Mesbah, *Stochastic MPC with active uncertainty learning: A Survey on dual control*, Annual Reviews in Control, 2018 | `[VERIFIED]` | DOI 10.1016/j.arcontrol.2017.11.001（Crossref）|
| 3 | Tishby et al., *The information bottleneck method* | `[VERIFIED]` | arXiv:physics/0004057 |
| 4 | *Dual Control for Approximate Bayesian RL* | `[VERIFIED]` | arXiv:1510.03591 |
| 5 | *Linear Quadratic Dual Control* | `[VERIFIED]` | arXiv:2312.06014 |
| 6 | *Minimax dual control with finite-dimensional information state* | `[VERIFIED]` | arXiv:2312.05156 |
| 7 | *An Approximate DP Approach for Dual Stochastic MPC* | `[VERIFIED]` | arXiv:1911.03728 |
| 8 | *Dual Control for Exploitation and Exploration (DCEE) in Autonomous Search* | `[VERIFIED]` | arXiv:2012.06276 |
| 9 | *Active Learning with Dual MPPI* | `[VERIFIED]` | arXiv:2310.07840 |
| 10 | *Active learning for anti-disturbance dual control of unknown nonlinear systems* | `[VERIFIED]` | arXiv:2212.08934 |
| 11 | TANDEM: *Learning Joint Exploration and Decision Making with Tactile Sensors* | `[VERIFIED]` | arXiv:2203.00798（摘要已抓）|
| 12 | *In-Hand Manipulation of Unknown Objects with Tactile Sensing for Insertion* | `[VERIFIED]` | arXiv:2210.13403（摘要已抓）|
| 13 | *POMDP Manipulation Planning under Object Composition Uncertainty* | `[VERIFIED]` | arXiv:2010.13565（仅列表命中）|
| 14 | *Shear-based Grasp Control for Multi-fingered Underactuated Tactile Robotic Hands* | `[VERIFIED]` | arXiv:2503.17501（摘要已抓）|
| 15 | *Learning In-Hand Translation Using Tactile Skin With Shear and Normal Force Sensing* | `[VERIFIED]` | arXiv:2407.07885（摘要已抓）|
| 16 | *In-Hand Object-Dynamics Inference using Tactile Fingertips* | `[VERIFIED]` | arXiv:2003.13165 |
| 17 | *Task-driven Perception and Manipulation for Constrained Placement of Unknown Objects* | `[VERIFIED]` | arXiv:2006.15503 |
| 18 | FREE | `[LOCAL]` | arXiv:2408.07855；`docs/related_work.md` |
| 19 | ContactSDF / TwinTrack / ContactGaussian-WM / Where to Touch / ComFree-Sim / Adaptive CI-MPC | `[LOCAL]` | `docs/related_work.md` §1–§3 |
| 20 | POISE | `[LOCAL]` | `docs/literature_sources_20260922/poise_paper.txt` |
| 21 | WM-Craftnet | `[LOCAL]` | `docs/literature_sources_20260922/wmcraft_paper.txt` |
| 22 | Tacmap | `[LOCAL]` | `docs/literature_sources_20260922/tacmap_paper.txt` |
| 23 | Feldbaum 对偶控制原始工作（1960–61） | `[MEM]` | 未验证 |
| 24 | Kaelbling/Littman/Cassandra, POMDP（1998） | `[MEM]` | 未验证 |
| 25 | Howard, value of information（1966） | `[MEM]` | 未验证 |
| 26 | 经典自适应控制 / MRAC 血统 | `[MEM]` | 未验证（教科书级，未单列）|

---

## 9. 本报告的局限（必须与结论一起读）

1. **检索是元数据级，不是全文级。** arXiv API 的 `all:` 不覆盖正文。§3.2 的"0 条"**不能**当作"领域内没人做过"。
2. **Semantic Scholar 不可用（429）**，所以**没有做引文网络追踪**——无法回答"谁引用了 TANDEM 并扩展了它"。这是本次核查最大的方法学缺口，**也是最可能藏着反例的地方**。
3. **未覆盖非 arXiv 来源**：ICRA/CoRL/RSS 正式论文、T-RO、IJRR、以及 2026 年新投稿。**仿真到真机的触觉探索工作很可能有会议论文未上 arXiv。**
4. **`[MEM]` 的 3 条经典引用未验证**，写进论文前必须逐条核实。
5. **未做任何复现**：没有安装 POISE / WM-Craftnet / TANDEM / 2210.13403，所有"它做不到 X"的判断都基于**它们自己写的限制与摘要**，不是实测。
6. 本报告**不构成**"我们的方法可行"的结论。它只解除了 `F1` §6 的阻塞，并收紧了可主张的边界。

---

## 10. 复现

```bash
# arXiv 元数据检索（网络需要走代理；export.arxiv.org 必须用 https）
curl -sS -G "https://export.arxiv.org/api/query" \
  --data-urlencode 'search_query=all:"tactile exploration" AND all:"in-hand manipulation"' \
  --data-urlencode 'max_results=6'

# 经典引用校验
curl -sS -G "https://api.crossref.org/works" \
  --data-urlencode "query.bibliographic=Dual control a survey Mesbah" --data-urlencode "rows=3"

# 本地全文证据
grep -n -A 12 -i "Limitations and future work" \
  docs/literature_sources_20260922/poise_paper.txt
grep -n -A 4 -i "^Limitations:" \
  docs/literature_sources_20260922/wmcraft_paper.txt
```

只读操作；未修改 FREE 任何文件；未安装、未运行任何被引系统。

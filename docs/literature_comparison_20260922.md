# 四项手内操作工作的创新边界核查（2026-09-22）

**结论：研究路线仍可推进，但“已知 CAD、视觉看姿态、实机不输入精确物体动力学、转到目标姿态”已不能单独作为新颖性主张。POISE 已直接覆盖相近甚至更完整的 6D 任务；WM-Craftnet 已覆盖多模态触觉状态、动作条件接触演化与闭环适应。当前可能保留的核心是：不预训练控制策略，通过多指触觉在线消除会改变目标动作选择的接触歧义，并证明信息动作改善后续操作。这个机制尚待建立和验证。**

本次精读三篇完整论文及附录、官方项目页，以及 `sharpa-rl-lab` README/仿真与部署代码；另外发现并读取了 WM-Craftnet 独立官方代码仓库 README，两篇官方博客也已缓存供交叉核验。没有安装这些系统、运行复现或修改本项目控制器。原始缓存位于 [literature_sources_20260922](literature_sources_20260922/)，下载记录见 [fetch_manifest.json](literature_sources_20260922/fetch_manifest.json)。以下“未建立/未报告”限定于已核验来源，不能推广为整个领域没有人做过。

## 1. 比较时固定我们的设定

依据原计划的任务相关触觉探索主线和当前 [路线 B](method_routes/route_b_contact_belief.md)：已知物体 CAD、视觉物体姿态及目标姿态、手运动学与本体观测；部署不输入真实质量/COM/惯量/摩擦，不以辨识这些参数为目标。候选观测为四指尖多接触位置与力、短历史、多指关系；没有触觉图像。动作仍是 FREE 的 16 维关节目标增量。希望在线可微优化，不预训练控制策略；GNN 可选。

“不预训练策略”不等于“无需任何传感器标定、手执行模型或先验”。当前允许观测与真实传感器的对应、未覆盖指节/掌面支撑、可执行信息动作以及未来接触风险均未闭合。原计划中“不知道整体几何”的表述与目前 CAD 已知设定不同；此处不把重建已知 CAD 当作探索收益。

## 2. 核心对照

| 项目 | 实际目标与部署观测 | 训练/物理参数边界 | 与我们的重合 | 本次能支持的区别 |
| --- | --- | --- | --- | --- |
| [POISE](https://arxiv.org/abs/2609.13761) | 固定腕、掌坐标下目标 SE(3)，同时平移与旋转。Actor 用视觉姿态/速度/置信度、本体、目标、BPS 几何；已知 mesh 配合 FoundationPose。**部署 actor 无触觉输入。** | 仿真 PPO；随机质量、COM、惯量、摩擦、PD、视觉噪声等；实机 actor 不输入真实物理参数。 | 已知 CAD + 视觉闭环 + 多物体手内目标姿态；抓取保持、接触转换与扰动恢复。 | 我们候选区别在于真实触觉参与的在线接触认知和信息动作，以及无策略预训练；不能只称“未知动力学目标姿态操作”。 |
| [WM-Craftnet](https://arxiv.org/abs/2609.07002) | 主评测是围绕指定 ±x/±y/±z 轴连续旋转，轴任务分别训练 WSM；另有工具平移演示。输入本体、**二值触觉接触**、腕部深度、动作历史、递归隐状态；无需部署时给 object-ID/精确物体姿态/CAD descriptor。 | RSSM 世界模型预测/重建 + PPO；9 物体先验复用于 49 物体训练。手关节 PD 在迁移前标定；物体参数随机化，实机不输入真实参数。 | 多指触觉融合、任务相关状态、动作条件接触/运动预测、在线状态更新、未知物体与扰动闭环适应。 | 论文 WSM 给 actor 提供上下文，不通过在线信息目标优化探测动作。我们的显式接触歧义与任务相关信息优化仍有区别，但不能把“在线状态更新”或“触觉状态模型”本身当独占贡献。 |
| [Tacmap](https://arxiv.org/abs/2602.21625) | 主要贡献是触觉仿真/实机统一表征；PPO 手内转球为应用验证，并非通用目标 SE(3) 控制。提供 **合力 F、接触位置 P、形变深度图 M**。 | 实机用标定数据训练图像→F/M 的网络，控制策略只在仿真训练；转球零样本实机迁移。未提供可支持任意物体物理泛化的完整范围/消融。 | 力、接触位置与几何触觉联合用于操作，已经存在。 | 我们处理在线决策，Tacmap处理传感表示/渲染；可互补。读取 MuJoCo 点力并不等于解决它的触觉迁移问题。 |
| [sharpa-rl-lab](https://github.com/sharpa-robotics/sharpa-rl-lab) | SharpaWave 手内旋转 demo，README 默认推荐半径 24 mm、高 60 mm 圆柱。部署本体、目标历史、五指触觉；默认用每指力标量，接触位置关闭。 | 仿真 RL→`ProprioAdapt` 蒸馏→硬件部署；仿真有 mass/COM/friction/PD 随机化与 privileged buffer；部署不读真实物体参数。 | 触觉力闭环与不输入实物参数的 sim2real；短/长观测历史。 | 是工程参考与低维触觉基线，**不是论文，也不是 POISE 或 WM-Craftnet 的完整实现。** 没有可据此宣称的通用 6D 结果或系统性 gap。 |

所有方法都使用仿真动力学来生成训练/评估数据；这与“部署控制器必须知道每个现实物体的真实质量/摩擦”是两回事。仅凭它们做域随机化，不能推出其部署依赖精确参数。我们的仿真也需要物理参数，这同样不意味着控制器读取它们。

## 3. POISE：任务重合最直接，但它确实尚未使用触觉闭环

证据来自 [全文](https://arxiv.org/html/2609.13761v1) / [PDF](https://arxiv.org/pdf/2609.13761)。

- **任务/观测：** §I，PDF p.1，原文 “jointly constrains ... 3D translation and 3D rotation”；§III，p.2 的 actor 使用三帧视觉与本体历史、BPS。额外 contact forces 在 critic，不能当作 actor 的触觉感知。§IV-B，p.3，BPS 描述已知表面几何。
- **真实参数不是部署输入：** §IV-E/Table I 与 §V-A，p.4。训练质量 10–100 g、惯量倍率 1–4、COM 每轴偏移至 5 mm、摩擦倍率 0.5–2；硬件 “only deployable proprioceptive and visual estimates are provided to the actor”。FoundationPose 使用 “a known object mesh”。
- **泛化范围：** §V-C，p.6，cube/hexagonal prism/square bifrustum 三个形状族、九个形状尺寸组合联合训练；hammer 采用扩大的平移 curriculum。不能把演示的 hammer 说成无任何训练的 unseen-object 泛化，也不能说所有实机物体共用一个未经调整的策略。
- **安全性质：** §IV-D 的抓取保持奖励近似 friction-cone wrench coverage，加掉落/力矩/功率惩罚；是奖励和经验成功率，非任意未知摩擦下的不掉落证明。我们也没有此证明。
- **作者明确 gap：** §VI，p.7：没有指定期望手/接触配置；现实表现受 “imperfect contact modeling and errors in 6D pose tracking” 影响。未来明确提出 “adding tactile feedback” 以及从真实交互适应，例如 residual dynamics。
- **复现状态：** [项目页](https://junxiaolin.github.io/poise-website/) 的代码按钮截至本次访问是 “Coming soon”，虽有交互演示，不能称完整代码已可复现。

**我们可能回应的部分：** 在视觉给姿态的条件下，用触觉发现接触约束变化，并让信息获取改变动作选择；这直接对应作者希望加入触觉/真实交互适应的方向。**尚不能说已解决：** 视觉估姿误差仍是我们的输入误差；没有期望抓取配置目标时也没有解决其手型问题；没有受控比较不能声称比 POISE 更稳健、更省样本或更安全。单纯给 POISE 加触觉反馈已是作者公开路线，不足以构成我们的完整新颖性论证。

## 4. WM-Craftnet：对“触觉状态模型”的覆盖最强

证据来自 [全文](https://arxiv.org/html/2609.07002v1) / [PDF](https://arxiv.org/pdf/2609.07002)。arXiv 页面标注 Accepted to CoRL 2026；本报告以公开稿方法与实验为依据。

- **输入与任务：** §3.1，p.3 和 Table 8，p.13，明确 “binary tactile/contact measurements”；contact force、物体 pose/velocity 是 reward/critic/diagnostic 信息，不是部署 actor 的连续力输入。主要目标为指定轴有符号旋转，不能等同任意 6D 目标到达。§4.3，p.7，写明 “a separately trained WSM for each axis”。§B.3，p.15 的 screwdriver 目标平移是额外演示，不构成已全面评测的通用 SE(3) 控制器。
- **确实在线推断状态：** §3.2–3.3，p.4，RSSM 利用新观测与前动作更新 posterior/recurrent state。不能说它“不在线适应”；应区别**运行时隐状态更新**与**部署时更新模型权重**。论文展示的是前者，权重训练在仿真 PPO rollout/replay 与下游训练阶段；没有给出每个新实物部署时进行在线梯度更新的实验依据。
- **与我们最有关的差别：** §3.3/3.5，pp.4–5，世界模型的 deterministic latent “detached before entering the policy network”，并作为 PPO actor 上下文。文中未建立显式“会改变任务决策的接触假设集合—候选动作信息增益—在线可微动作优化”；其 reward 是旋转、漂移、接触、能耗等，不是我们候选的信息动作机制。**但没有显式信息项不代表策略绝无隐式探测行为。**
- **训练与未知物理：** §3.4/§A.1，pp.4–5/12，9 物体预训练的 WSM 在 49 物体 PPO 训练中继续适应新 rollout；不能说 49 物体全部是零样本测试。§A.3/Table 7，pp.12–13，辨识的是手关节响应/PD；物体 mass 训练 0.2–0.4 kg、测试 0.1–1.4 kg，摩擦倍率 0.5–2。它已做部分超训练质量区间验证，不能把这一点当尚无人验证的 gap。
- **作者明确 gap：** §5，p.9：“current tasks remain short-horizon, and severe-perturbation recovery remains limited”；物体超手工作空间、大偏移超恢复区域、漂移/卡住导致失效。不是只在固定初态/熟悉物体上工作；§4 已报告 unseen-object 与扰动实验。
- **复现状态：** 有独立 [官方仓库](https://github.com/sharpa-robotics/WM-Craftnet) 和 [checkpoint 链接](https://huggingface.co/SharpaIT/WM-Craftnet)。README 的 [WSM prediction heads](https://github.com/sharpa-robotics/WM-Craftnet#wsm-prediction-heads) 明确 “The released checkpoint uses proprioception + depth”；完整辅助头配置须匹配。本次仅确认公开代码/说明存在，未运行完整实验。

**我们可能回应的部分：** 让已知 CAD、多指连续力与接触几何支持显式、任务相关的接触不确定性更新；在无策略预训练的条件下，用少量真实在线交互调整动作。**尚不能说已解决：** 工作空间限制无法靠认知模型消除；极端扰动、接触切换和长时操作也正是我们尚未建成的能力。没有大范围实验，不能把基于训练分布的方法概括为“不能适应未知物体”。

## 5. Tacmap：传感层互补，不能把它说成没有力反馈

证据来自 [全文](https://arxiv.org/html/2602.21625v2) / [PDF](https://arxiv.org/pdf/2602.21625)。

- **有 F/P/M 三路信息：** §III-A，p.3，原文 “net force F, contact position P, and deform map M”。仿真 F 来自物理引擎，实机 F 来自原始触觉图像→ResNet，并用外部高精度力传感器标定；实机 P 是有效接触区域的几何质心。不能声称它只有触觉图像或只有二值触觉。
- **何处训练：** §III-C，p.4，有自动压入标定装置及 image-to-deform 翻译网络训练。§V-D，pp.6–7，PPO 控制策略只在仿真训练，球体连续旋转直接部署，“without any real-world fine-tuning or domain adaptation”。这不等于整套系统从未使用真实标定数据。
- **CAD/物理边界：** 渲染需要物体 mesh 和相对姿态；现实触觉翻译从图像推断 F/P/M。不能把“渲染器知道 mesh”推成“实机 actor 必须给精确 CAD 和物体质量”。文中没有足够控制实验/随机化表来证明其对任意未知物理参数普遍有效，同样也不能反推它必须输入这些参数。
- **作者明确 gap：** §VI，p.7：“does not explicitly model the tangential force distribution (shear strain) across the elastomer surface”；另有复杂 mesh 下 ray-casting 成本。这里缺的是**密集切向形变/剪切分布**，并不是全部合力或接触位置缺失。
- **实验范围与复现：** 有 normal indentation 的 force/geometry 对齐、并行渲染效率和转球零样本应用。全文与本次读取的[官方博客](https://www.sharpa.com/blogs/research/tacmap-breaking-the-sim-to-real-deadlock-in-tactile-simulation-with-a-geometric-language)未确认独立可下载的完整 Tacmap 发布包；不能把 `sharpa-rl-lab` 默认力标量 demo 当成 Tacmap deform-map 全部实现。

**与我们关系：** 若未来实机提供可靠连续力/接触位置，Tacmap 类传感管线可以作为上游。我们用多指力做决策可能补充其应用层，但既没有开发切向形变渲染器，也未解决图像到物理量的 sim2real 标定，不能声称“解决 Tacmap 的 shear gap”。尤其不能把 MuJoCo 多个求解器接触点等同于一个实际传感器可独立测出的多点力。

## 6. sharpa-rl-lab：以代码为准的工程边界

证据：[README](https://github.com/sharpa-robotics/sharpa-rl-lab)、[仿真配置](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/main/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env_cfg.py)、[部署配置](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/main/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env_cfg.py)、[部署实现](https://github.com/sharpa-robotics/sharpa-rl-lab/blob/main/rl_isaaclab/tasks/inhand_rotate/sharpa_wave_deploy_env.py)。代码以 2026-09-22 缓存为准，`main` 链接后续可能变化。

- README 定位为 “reinforcement learning sim2real rotation demo”；流程是生成 grasp cache、训练策略、`ProprioAdapt` 蒸馏、实机部署。没有论文式实验覆盖/统计指标可直接用来论证它的失败条件。
- `compute_observations()` 使用 22 关节位置、22 当前目标、5 触觉力值、15 接触位置值；三帧 actor 输入，另保留长历史。`get_tactile_info()` 实际每指取一个 `f_norm`；默认 `binary_contact=False`、`enable_contact_pos=False`，后者注释为 “Not tested yet”。所以默认接触位置块是零，不能把 API 预留等同于已验证完整接触几何控制。
- 仿真配置默认随机质量 0.01–0.25 kg、COM 每轴 ±0.01 m、摩擦倍率 0.5–2、PD 倍率 0.5–2。`sharpa_wave_env.py` 把随机摩擦、质量等写入 privileged buffer，而部署 `get_observations()` 返回观测与历史；不能说部署需读取真实物体参数。
- README 的实机准备、力标定/阈值/缩放和硬件 SDK 说明表明它是可用的工程起点；本次没有执行训练或硬件步骤。它与 WM-Craftnet 独立仓库不可混为一谈。

## 7. 哪些主张保留，哪些需要收回

| 候选主张 | 审核判断 |
| --- | --- |
| 第一次只知道 CAD/视觉姿态，不输入真实质量摩擦，将物体手内转到目标姿态 | **不能保留。** POISE 的部署条件已高度重合，且目标为完整 6D。 |
| 第一次用多指触觉/历史更新任务相关状态并控制 | **不能保留。** WM-Craftnet 与更早的触觉 RL/估计已有大量重合。 |
| 加入接触力、接触位置或图结构即构成主要创新 | **不足。** Tacmap/demo 已含力/位置；确定性图边不增加观测信息，需证明其算法作用。 |
| 不依赖离线策略预训练，在线用接触认知进行可微动作选择 | **可保留为方法定位。** 四项没有完整覆盖该组合，但无训练本身不是充分创新或性能优势。 |
| 只探索会改变目标动作选择的接触歧义，并以真实触觉闭环验证收益 | **最值得验证的机制。** 需明确未知变量、可区分观测、可行动作与信息目标；还需与更广的主动感知/双重控制相关工作比较。 |
| 解决 RL 的泛化问题、Tacmap 的触觉迁移问题、任意未知动力学安全操作 | **目前均不能声称。** 我们没有这些相应证据和保证。 |

**邻近程度不是单一排序：** 按任务目标，POISE 最近；按触觉状态/接触认知机制，WM-Craftnet 最近；Tacmap 是传感表示邻近工作；`sharpa-rl-lab` 是工程参照。把其中任一项完全排除出相关工作都会使论证薄弱。

## 8. 最小验证顺序

1. **先证明观测有意义。** 按实际传感覆盖比较二值接触、每指合力、力+接触位置、多接触/短历史；统计未覆盖支撑。不能让 richer MuJoCo oracle 输入自动赢过论文中真实低维触觉，再把优势归因于算法。
2. **建立一个可辨别且任务相关的接触场景。** 相同已知 CAD/姿态下，列出会改变目标动作选择的接触假设；用允许观测与实际执行动作证明能区分它们。不能仅用仿真隐藏标签判别，不能把不满足执行/接触约束的数学方向当信息动作。
3. **证明信息改变行为并改善任务。** 固定内部模型、观测、初态、动作/时间/计算预算，对比任务贪心、一般信息探索、任务相关探索。测目标姿态达成、时间/交互量、掉落/失接触/峰值载荷；后验熵下降不代替任务收益。
4. **再检验论文对应的 gap。** 针对 POISE 的接触模型误差和触觉缺失，做视觉/本体与增加触觉的受控实验；针对 WM 的极端扰动/新接触，检验在线信息机制是否帮助恢复。物体超工作空间、视觉完全失效不应被默认列为我们能够解决的情形。
5. **最后才能比较迁移与成本。** 已知 CAD、同一个手、明确传感器误差与覆盖下，比较初始化、预训练成本、在线适应交互和部署成功率；同时覆盖训练范围内/外物理变化与不同初态。域随机化 RL 已有强基线，不能只挑一种旧混淆失配结果来立论。

本次结论不是“换个名字就与 RL 不同”，而是给出可检验的边界：**我们若证明无需策略预训练的任务相关主动触觉，能在相同观测和预算下改善目标操作，就是一项具体贡献；目前只有这个候选机制与待验证缺口，没有已经解决这些论文不足的结果。**

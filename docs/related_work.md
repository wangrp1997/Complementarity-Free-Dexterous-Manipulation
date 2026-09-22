# Related Work

对照问题（见 `research_plan.md`）：

> 视觉提供物体当前姿态与目标姿态；在手旋转过程中用主动触觉在线获取**转到该目标所需的**几何/接触信息；动作同时服务到达、信息增益与安全。  
> \(u_t^*=\arg\min_u[L_{\mathrm{task}}+\lambda L_{\mathrm{info}}+\beta L_{\mathrm{safe}}]\)。  
> 不是先完整 3D 重建再操作，也不是把论文做成“在线学一个更准的运动模型”。

本文档整理 2026-09 检索到的、与 FREE 及上述问题直接相关的文献。引用次数以 Semantic Scholar 对 Jin, RSS 2025 的记录为准（约 34 次）。

---

## 1. 无互补 / 显式接触模型 + 转到目标姿态

这类工作把接触从互补约束改成闭式、可微模型，再做接触隐式 MPC，任务通常是**给定目标位姿的重定向**。默认几何已知、动力学参数手调或离线给定，**无指尖触觉阵列，无“探索多少由任务决定”**。

| 工作 | 年份 | 做了什么 | 相对本问题的缺口 |
|---|---|---|---|
| **FREE** — Jin, *Complementarity-Free Multi-Contact Modeling and Optimization for Dexterous Manipulation*, RSS 2025, [arXiv:2408.07855](https://arxiv.org/abs/2408.07855) | 2025 | Anitescu 对偶 QP + 对角 \(K(q)\)，闭式接触步进；fingertips / TriFinger / Allegro；全文汇总成功率 96.5%，CI-MPC 50–100 Hz | 模型已知；\(L_{\mathrm{task}}\) 追目标，无 \(L_{\mathrm{info}}\) / 触觉信念。我们 Allegro 原版可复现；COM/惯量/摩擦错配后 17 物体合计 82.9%→13.8% |
| **ContactSDF** — Yang & Jin, RA-L 2025, [arXiv:2408.09612](https://arxiv.org/abs/2408.09612) | 2025 | SDF 近似碰撞+步进；on-MPC 学 \(\theta\)；真机 Allegro 约 2 min、30–60 Hz | 几何已知且**凸**；用多段带目标 rollout 拟合预测残差；学习阶段无显式安全；无触觉 |
| **On-Palm Dexterity** — Xie, Yang, Jin, ICRA 2025 Workshop | 2025 | 把 complementarity-free 扩到全动态，翻面/滑动；托盘真机 | 抗外扰，不是在线接触几何信念；无触觉 |
| **Where to Touch** — Xie, Xiang, Posa, Jin, [arXiv:2601.10930](https://arxiv.org/abs/2601.10930) | 2026 | 上层 RL 选接触位置+物体子目标，下层 complementarity-free MPC | 几何泛化 / sim-to-real；非触觉探索；非掌内多指阵列 |
| **ComFree-Sim** — Borse, Xie, Huang, Jin, [arXiv:2603.12185](https://arxiv.org/abs/2603.12185) | 2026 | FREE 接触模型的 GPU/Warp 引擎；6D 摩擦；对标 MJWarp 吞吐 2–3×；LEAP 真机 MPPI | 可学 \(K,D\) 是接触阻抗，不是任务相关几何信念；无触觉 |
| **ContactNNLS** — Li, Luo, Yang, Pan, Huang, TIE 2026 | 2026 | NNLS 显式接触 + MPC；声称比现有显式模型更一般；真机 Allegro 掌上重定向 | 模型形式改进；未见到与 FREE 逐物体对照，也无触觉 / \(L_{\mathrm{info}}\) |

FREE 原文自己的对照是 complementarity / QP **implicit MPC**（速度与成功率），不是未知接触几何，也不是触觉。

### 这类各自怎么保证“能转、辨识时安全”（答案：基本不保证）

| 工作 | 辨识何时发生 | 辨识时还在转到目标吗 | 安全靠什么 | 缺口（写清楚） |
|---|---|---|---|---|
| FREE | 不辨识；\(K,\mu,m\) 手调，对所有物体同一套 | 一直在转目标 | 软代价（保持抓取、控制别太大）；模型对才不掉 | 模型错了就掉。我们未知动力学列 13.8% 就是这个 |
| ContactSDF | **边追目标边拟合** \(\theta\)：多段随机目标 rollout，约 2 min 真机 | 是，但允许前面许多段失败 | 仍是 MPC 软代价；**无 \(L_{\mathrm{safe}}\)**，失败算学习预算 | 能转是因为初始模型够近 + 可以掉多次；不是同一次里“探索且保证不掉” |
| On-Palm Dexterity | 不辨识 | 转到目标（翻/滑） | 动态 MPC + 阻抗；抗外扰实验 | 接触几何当已知；无触觉探索 |
| Where to Touch | 不辨识物体物理；RL 学“摸哪里” | 子目标序列，最终到位 | 下层 MPC | 摸哪里由视觉几何+RL，不是指尖阵列；无在线接触信念 |
| ComFree-Sim | 可选学 \(K,D\)（接触软硬） | 控制实验在模型给定后 | 引擎速度换闭环频率 | 学的是阻抗不是任务相关接触几何 |
| ContactNNLS | 不辨识 | 转到目标 | 同 FREE 一类软约束 | 只换显式接触写法 |

---

## 2. 模型基在手操作：规划 + 力/接触跟踪

| 工作 | 年份 | 做了什么 | 相对本问题的缺口 |
|---|---|---|---|
| **Jiang et al.** — *Robust Model-Based In-Hand Manipulation with Integrated Real-Time Motion-Contact Planning and Tracking*（仓库 `in_hand_manipulation_2`） | 2025 | CQDC + Crocoddyl DDP + 力运动跟踪；仿真 Rotate Sphere 100/100（≤90°、60 s、最小误差 < 8°）；真机可抗人手干扰 | 几何已知，惯量/接触**近似估计**；错了靠下层跟踪补，不是主动触觉探索任务相关几何。引用 FREE，但未把 FREE Allegro 表当数字基线重跑 |
| **Jiang et al.** — *Robust In-Hand Reorientation With Hierarchical RL-Based Motion Primitives and Model-Based Regrasping* | 2026 | 分层 RL 原语 + 模型基再抓 | 仍是到达目标；不是触觉信息增益闭环 |

这类说明：**模型（近似）已知时，规划+跟踪能转到目标。** 不回答“接触几何一开始不清楚、要边转到目标边摸”。

Jiang 的安全不是在线辨识保证的：惯量/摩擦只近似一次，之后错了靠力跟踪硬跟。模型错大了跟踪也救不了；也没有“这一步会掉就别探索”。

---

## 3. 未知动力学 / 未知物体：视觉系统辨识与跟踪

原组后续确实在谈未知物理，但传感器和目标与本问题不同：用 **RGB-D / 视频** 拟合质量、惯量、摩擦或几何残差，服务**跟踪或再规划**，不是掌内触觉信念。

| 工作 | 年份 | 做了什么 | 强假设 / 缺口 |
|---|---|---|---|
| **TwinTrack** — Yang, Xie, Wang, Tadepalli, Ben Amor, Lin, Jin, ICRA 2026, [arXiv:2505.22882](https://arxiv.org/abs/2505.22882) | 2026 | Real2Sim + Sim2Real：GS 重建几何，学质量/惯量/摩擦 + **常数胀缩**几何残差；RGB-D + 本体感觉，>20 Hz 跟踪 | 必须看得见；接触来自本体感觉而非指尖阵列；输出是 6D 位姿不是 \(u_t^*\)；作者承认摩擦/惯量激励不足则不可辨识，掉落 vs 掌内参数对不上 |
| **ContactGaussian-WM** — Wang, Jin, Cao, Xie, Hong, [arXiv:2602.11021](https://arxiv.org/abs/2602.11021) | 2026 | 视频反传 complementarity-free 模型，学 \(M,\mu,K,D\)；各向同性高斯球作碰撞 | 标定相机、分割、交互视频；离线建世界模型；掌内遮挡时视觉损失不可靠 |
| **Adaptive CI-MPC with Online Residual Learning** — Huang, Aydinoglu, Jin, Posa, ICRA 2024 | 2024 | 接触隐式 MPC + 在线残差 | 残差补偿模型误差，不是多指触觉几何信念 |
| **Task-Driven Hybrid Model Reduction** — Jin & Posa, T-RO 2024 | 2024 | 只保留任务需要的接触模态 | 任务相关在“模态约化”，不是触觉探索 |

**和本问题的关键差别：** 他们做的是视觉/滚动数据上的系统辨识。本问题假设视觉已给出姿态，未知的是**转到目标还缺哪些接触/几何信息**，用触觉在同一次重定向里补。

### 这类各自怎么“先拟合 \(m,I,\mu\)”（拆开辨识和操作）

| 工作 | 具体做什么 | 此时在转到目标吗 | 安全 / 能转？ |
|---|---|---|---|
| TwinTrack | **先掉落或先看一段接触运动**：RGB-D 盯轨迹，反推质量、惯量、摩擦；几何残差只学一个整体胀缩常数。评测里明确有“物体下落碰环境”和“别人已经在做的掌内操作”两套场景 | **否。** 输出是 6D 位姿跟踪，不是 \(u_t^*\)。掌内场景里手由别的控制器在动 | 辨识本身不负责抓住/转到目标。摩擦/惯量激励不够就估不准；掉落 vs 掌内两套参数对不上 |
| ContactGaussian-WM | **先看视频**：拍推、碰的录像，反传 complementarity-free 模型，调 \(M,\mu,K,D\) 直到渲染对上录像 | **否。** 离线建世界模型，拟合完再预测/规划 | 辨识在录像上，没有“手里正转目标时保证不掉” |
| Adaptive CI-MPC | 控制过程中学残差，名义接触模型仍在 | 是 | 安全靠名义模型别太差 + 软约束；学的是运动残差不是接触几何信念 |
| Hybrid Model Reduction | 离线/任务前丢掉用不到的接触模态 | 之后再转 | 不是在线触觉探索 |

一句话：这类把“认出多重、滑不滑”和“转到目标姿态”拆成两步。前一步可以专门掉、专门拍、激励不够就估漂；后一步假定参数已经有了。

---

## 4. 触觉：重建、动力学估计、柔顺接触

| 工作 | 年份 | 做了什么 | 相对本问题的缺口 |
|---|---|---|---|
| **NeuralFeels / NeuralFeels-MuJoCo** | 2023– | 指尖视觉触觉 + 物体几何重建 | 偏 **完整 3D 重建**；`research_plan.md` 明确不做 “先摸完整再操作” |
| **Sundaralingam et al.** — *In-Hand Object-Dynamics Inference using Tactile Fingertips*, [arXiv:2003.13165](https://arxiv.org/abs/2003.13165) | 2020 | BioTac 估摩擦（主动滑）和惯量（因子图） | 先辨识动力学再操作；不是任务相关接触几何 + \(L_{\mathrm{info}}\) |
| **Tactile Probabilistic Contact Dynamics Estimation of Unknown Objects**, [arXiv:2409.17470](https://arxiv.org/abs/2409.17470) | 2024 | DeepSDF + 粒子滤波 + 信息增益探索，估几何与物理参数 | 主动触觉探索有，目标是**接触动力学估计**，不是转到给定目标姿态的在手操作 |
| **Hydrosoft** — Oller, Dang, Fazeli, 2025 | 2025 | 柔顺触觉的 hydroelastic 模型；引用 FREE | 接触本体模型，不是探索–信念–重定向闭环 |
| **tactile_envs** — Ferrazza et al. | — | Gym 触觉方块/Egg/Pen | 环境，不是本方法 |
| **XHand Manipulation** | — | 三指 + 指尖触觉 + 在手旋转，sim-to-real | 偏策略/工程平台 |
| **TacTID** — Song et al., IEEE Sensors 2024 | 2024 | 足式地形的视觉触觉 | 非在手 |

有触觉、有探索的工作，多数停在**重建或辨识**；有在手转动的工作，多数不用主动触觉来决定“下一步摸什么才能对准目标”。

### 这类辨识/探索时安不安全、能不能转

| 工作 | 探索/辨识在干什么 | 还在转到目标吗 | 安全 | 缺口 |
|---|---|---|---|---|
| NeuralFeels | 主动摸，把形状摸完整 | 重建完才谈操作 | 探索协议，不是 \(L_{\mathrm{safe}}\) | 完整 3D，违反“探索量由目标决定” |
| Sundaralingam 2020 | **先专门打滑**：加压再加切向力直到滑，用刚滑的力估 \(\mu\)；再晃动物体估惯量 | **否。** 标定流程，做完再操作 | 选不太拧的抓法、靠 BioTac 力估计；不是转到目标时的安全 | 先辨识动力学，不是任务相关接触面/让位 |
| Tactile Prob. Dyn. 2024 | 信息增益动作，粒子滤波估几何+物理 | **否。** 目标是把动力学估准 | 探索为信息，不为到达 | 有 \(L_{\mathrm{info}}\) 味道，缺 \(L_{\mathrm{task}}\) 转到给定姿态 |
| Hydrosoft | 柔顺接触怎么建模 | 不涉及本任务闭环 | — | 模型论文 |
| tactile_envs / XHand / TacTID | 环境或别的任务 | — | — | 不是本方法 |

---

## 5. 其他引用 FREE、但未当数字基线重跑的工作

Semantic Scholar 列出的引用里，多数把 FREE 当作“显式/无互补接触”的 related work，**没有复现 Allegro 17 物体成功率表**。择要：

- Suh, Pang, Zhao, Tedrake, *Dexterous contact-rich manipulation via the contact trust region*, IJRR 2025 — 接触信任域规划。
- Chen et al., *Robust Differentiable Collision Detection for General Objects*, [arXiv:2511.06267](https://arxiv.org/abs/2511.06267) — 可微碰撞。
- Li, Gong, Chalvatzaki, *IMPACT*, 2026 — 接触隐式轨迹优化求解器。
- Liu et al., *Geometry-Informed Contact Optimization and MPC*, RA-L 2026。
- Zhang et al., *Simultaneous Contact Selection and Planning with Cascaded Optimization*, 2026。
- Raicevic et al., *Object-Informed MPPI for Non-Prehensile Manipulation*, 2026。
- Li et al., *Certified Gradient-Based Contact-Rich Manipulation*, 2026。
- Huang et al., *A Unified Complementarity-based Approach for Rigid-Body Manipulation*, 2026。
- Nechyporenko et al., *Compliant Sphere Lattice Contact*, 2026。
- Leve et al., *Scaling Whole-Body Multi-Contact Manipulation*, 2025。
- Esteban et al., *Reduced-Order Model Guided CI-MPC for Humanoid Locomotion*, 2025。
- Hung et al., *AVO: Amortized Value Optimization for Contact Mode Switching*, 2025。
- Yang et al., *CADRE*, 2025。
- Liu et al., *DexTrack*, 2025。
- Hsieh et al., *DexMan*, 2025。
- Yuan et al., *UniBYD*, 2025。
- Zhao et al., *EaDex*, 2026。
- Yu et al., *RGMC Champion Solution*（大范围精确在手移动），2025。
- Li et al., *What Foundation Models can Bring for Robot Learning in Manipulation*（综述），2024。

这些不构成“FREE Allegro + 未知接触几何 + 触觉”的统一评测协议。

---

## 6. 本仓库已有的诊断（不是方法贡献）

在官方 Allegro 评测流程上（`params.py` 未改，四元数误差 < 0.04 连续 20 步）：

- 原版 17 物体 × 20 trial：282/340 = 82.9%（cube 20/20）。论文 96.5% 是全文多手多任务汇总，不能直接对。
- 仅错 plant 的 COM / 惯量 / 摩擦（质量不改），**且保持同一抓取/物体初态**：268/340 = 78.8%。cube 20/20。stick 从 13/20 掉到 2/20，是少数真正被动力学打掉的物体。
- 早期 47/340 = 13.8% 作废：`mj_setConst` 把 `qpos` 写回 XML 默认，手指张开、物体不在掌上，失败混了抓取变化。

说明：这一层温和 COM/惯量/摩擦错配，在**相同初态**下多数物体仍能转到目标；FREE 对这层并不像 13.8% 那么脆。任务相关接触几何未知仍是另一层，不能用这张表代替。

---

## 7. Gap（对照 `research_plan.md`）

还没有工作同时满足：

1. **任务**：在手转到**给定目标姿态**（连续转 = 同一 \(L_{\mathrm{task}}\) 把目标沿轴前移）。
2. **分工**：视觉（或仿真真值）只负责当前姿态与目标姿态；**不**先做完整 3D mesh。
3. **未知**：一开始不清楚转到目标还需要哪些接触面/边缘/让位关系（以及与之耦合的摩擦等）。
4. **传感**：多指尖触觉阵列，信息在**指间关系** \(G_t\) 里，不只是单点力。
5. **动作**：同一次重定向里 \(\arg\min(L_{\mathrm{task}}+\lambda L_{\mathrm{info}}+\beta L_{\mathrm{safe}})\)，探索量由到达目标所需信息决定。

按方法切在哪一条（对照上面五条）：

| 方法 | ①转到目标 | ②视觉只管姿态 | ③任务相关接触未知 | ④多指触觉 | ⑤同一次 \(L_{\mathrm{task}}+L_{\mathrm{info}}+L_{\mathrm{safe}}\) | 辨识时保证安全且还能转？ |
|---|---|---|---|---|---|---|
| FREE / On-Palm / ContactNNLS / ComFree-Sim | 是 | 几何已知，不靠触觉 | 否，参数给定 | 否 | 否，只有 task | 不辨识；模型对才安全 |
| ContactSDF / Adaptive CI-MPC | 是 | 几何已知/凸 | 学残差，不是接触面信念 | 否 | 否；无 info，无硬 safe | **否**：边学边试，失败算预算 |
| Jiang | 是 | 几何已知 | 否，近似一次 + 跟踪补 | 力跟踪，非阵列探索 | 否 | 不在线辨识 |
| TwinTrack | 否（跟踪） | 否，还重建几何 | 估 \(m,I,\mu\) | 否 | 否 | **拆开**：先掉落/先看运动 |
| ContactGaussian-WM | 拟合后再规划 | 否，从视频学几何+物理 | 学 \(M,\mu,K,D\) | 否 | 否 | **拆开**：先看视频 |
| Where to Touch | 是 | 视觉几何 | 否 | 否 | 否（RL 选摸哪，无 safe+info 同式） | 不辨识物理 |
| NeuralFeels | 重建后 | 否，要完整形状 | 完整 mesh | 是 | 否 | 先摸完整 |
| Sundaralingam | 辨识后 | — | 估 \(\mu,I\) | 是（BioTac） | 否 | **拆开**：先打滑、先晃 |
| Tactile Prob. Dyn. | 否 | 否 | 估几何+物理 | 接触传感 | 有 info，无转到目标 | 探索为估计 |

没有一行五条全是。尤其没有人保证：**同一次转到目标时，探索动作既安全又不中断转向目标。**

因此该问题设定上仍有 gap。方法结果尚未产生，不能声称已经填上。

# Tactile Exploration for Online Geometric\-Aware In\-Hand Manipulation

核心不是“在线学习一个更准的运动模型”，而是：

$\boxed{\text{Tactile Exploration}\leftrightarrow\text{Geometric/Contact Belief}\leftrightarrow\text{In-Hand Manipulation}}$

### 生物学动机

人类的主动触觉并不是“先感知、后操作”的串行过程。人在缺少视觉时，会主动移动手指探索物体表面，并根据接触、力和表面几何逐步形成形状认识；而且探索动作本身会随物体特征和当前感知需求发生变化。经典的 Nature 工作证明了主动触觉中的力信息能够参与形状感知；Nature Neuroscience 的研究则表明，指尖触觉神经活动能够快速编码复杂的空间接触事件。

所以可以抽象成：

$\boxed{ \text{动作既改变物体状态，也获取关于物体的信息} }$

也就是：

> **Manipulation is simultaneously a means of acting on the object and a means of acquiring information about the object\.**
> 
> 

---

### 你的核心问题

不要再把问题定义成：

$\text{Tactile}\rightarrow\text{Motion Prediction}\rightarrow\text{Action}$

而定义成：

$\boxed{ \text{如何在视觉提供物体姿态的前提下，利用在手旋转中的主动触觉，在线获取把物体稳定、安全转到目标姿态所需的几何/接触信息，并据此持续调整操作？} }$

对应：

$\text{Observation} \rightarrow \text{Belief Update} \rightarrow \text{Information-aware Manipulation} \rightarrow \text{New Observation}$

完整闭环：

$T_t^{1:N} \rightarrow z_t \rightarrow G_t \rightarrow M_t \rightarrow u_t^* \rightarrow T_{t+1} \rightarrow M_{t+1} \rightarrow\cdots$

其中：

- $T_t^{1:N}$：多个指尖的触觉阵列

- 视觉：物体当前姿态与目标姿态

- $z_t$：结构化多指触觉信息

- $G_t$：多指之间的接触关系

- $M_t$：当前对物体几何/接触状态的**与转到目标姿态相关的认知**

- $u_t^*$：下一步操作动作

---

### 为什么不是传统的 3D 重建？

这里是一个很重要的区别。

已有工作已经在做 **vision \+ touch 的主动 3D shape reconstruction**，即通过主动选择触觉位置逐渐恢复物体形状。

你不应该再做：

$\text{Tactile Exploration} \rightarrow \text{Complete 3D Reconstruction} \rightarrow \text{Manipulation}$

而是：

$\boxed{ \text{Task-relevant Geometry} \rightarrow \text{In-Hand Reorientation to Target Pose} }$

机器人不需要知道整个物体的 mesh。视觉（仿真里也可用物体姿态真值）负责**现在朝哪、目标朝哪**；接触怎么走、稳不稳，交给触觉。

先做**有视觉版本**。无视觉（靠触觉估姿态）作为更难的后续。

任务就是：把物体在手里**稳定、安全地转到给定目标姿态**（和用工具时一样，终点是能用的朝向）。

连续转和目标姿态不冲突，同一套方法一起做：`L_{\mathrm{task}}` 盯住目标即可。目标固定就是转到该姿态；目标沿转轴不断前移就是连续转。主任务仍是目标姿态，连续转作为同一控制器下的技能实验（评测用 TTF、累计转角）。

为此它只需要逐渐知道：

$\{\text{当前接触面、边缘、接触关系、可旋转方向、手指让位关系}\}$

然后马上用于下一步旋转，逐步对准目标姿态。

因此：

$\boxed{ \text{探索多少，由转到目标姿态所需的接触信息决定} }$

这比“把物体摸完整”更有研究味道。

---

### 多指触觉在这里就有真正作用了

每个指尖不是孤立的：

$x_i= [p_i,F_{n,i},F_{t,i},A_i,\dot F_i,s_i,\ldots]$

构成：

$G_t=(V_t,E_t)$

其中节点是各指尖的触觉状态，边描述指尖之间的空间、力和接触关系。

关键不是为了“用 GNN 而用 GNN”，而是因为：

$\boxed{ \text{物体几何信息往往体现在多个接触之间的关系中} }$

例如两个指尖同时接触两个不同表面，仅看单个指尖很难判断接触是否还能支撑转向目标；但它们之间的相对位置、法向和力分布可以提供更强的几何约束。当前朝向由视觉给出。

所以 GNN/关系编码可以作为**多指触觉结构表征手段**，最终是否需要 GNN，可以通过实验决定。

---

### 在线学习模型放到“内部”

你之前的在线 nonlinear model 仍然可以保留：

$z_t,u_t \rightarrow f_{\theta_t} \rightarrow \text{predicted contact/motion consequence}$

并在线更新：

$\theta_{t+1} = Update(\theta_t,z_t,u_t,z_{t+1})$

但它不再是论文故事的主角。

它只是帮助机器人回答：

> **“如果我现在这样动，我大概会得到什么新的接触/运动结果？”**
> 
> 

然后支持下一步动作选择。

---

### 真正的核心可以放到动作选择

最终：

$u_t^* = \arg\min_u \left[ L_{\mathrm{task}} + \lambda L_{\mathrm{info}} + \beta L_{\mathrm{safe}} \right]$

三个东西同时考虑：

$\underbrace{L_{\mathrm{task}}}_{\text{目标姿态误差；连续转时把目标沿轴前移}} + \lambda \underbrace{L_{\mathrm{info}}}_{\text{这个动作能让我知道什么}} + \beta \underbrace{L_{\mathrm{safe}}}_{\text{会不会掉、卡住、过度挤压}}$

于是一个动作可能不是因为“转得最快”而被选择，而是因为：

> **它既能把物体稳定、安全地转向目标姿态，又能主动获得下一步所需的几何信息。**
> 
> 

这就是 **Exploration\-as\-Manipulation** 最核心的地方。

---

### 在手转到目标姿态 Demo 就很好理解了

视觉给出物体当前姿态和目标姿态。物体几何/接触一开始也不清楚。

任务是：

> 在手里把物体稳定、安全地转到给定目标姿态。
> 

同一套闭环也做连续转：把目标沿转轴一段一段前移即可，不另起方法。

机器人：

$\text{视觉读姿态} \rightarrow \text{触摸} \rightarrow \text{判断当前接触几何} \rightarrow \text{旋转/重新接触} \rightarrow \text{获得新触觉} \rightarrow \text{更新几何认知} \rightarrow \text{继续转向目标}$

而不是：

$\text{先把物体摸一遍} \rightarrow \text{建立完整模型} \rightarrow \text{再开始转到目标姿态}$

这正好把“主动触觉”和“在手操作”真正绑在了一起。无视觉版本留作后续。

当然，**某个具体物体本身不是创新点**；立方体、不规则物体都可以当验证平台。真正的论文问题应该是：

$\boxed{ \text{Can a dexterous robot use vision for object pose and tactile exploration for contact, to reorient an object to a target pose in hand?} }$

---

### 最终研究标题

我觉得目前这个标题是比较准确的：

**Tactile Exploration for Online Geometric\-Aware In\-Hand Manipulation**

中文可以叫：

**主动触觉探索驱动的在线几何感知在手操作**

核心一句话就是：

> **先做有视觉版本：视觉负责物体当前姿态和目标姿态，机器人无需先完整感知物体，而是在转到目标姿态的过程中利用主动触觉获取接触信息并在线调整操作。同一套方法把目标沿轴前移即可做连续转。无视觉作为更难的后续设定。**
> 
> 

这条线相比单纯的“触觉 → 在线动力学模型 → MPC”确实更完整，因为它把**仿生动机、触觉探索、几何认知、在线学习和在手操作**统一到了一个闭环里。


### 仿真环境参考

1. **Jiang — Robust Model\-Based In\-Hand Manipulation**
[https://github\.com/Director\-of\-G/in\_hand\_manipulation\_2](https://github.com/Director-of-G/in_hand_manipulation_2?utm_source=chatgpt.com)
MuJoCo 灵巧手在手操作环境，包含接触动力学建模、运动\-接触规划与触觉反馈控制。

2. **FREE — Complementarity\-Free Dexterous Manipulation**
[https://github\.com/asu\-iris/Complementarity\-Free\-Dexterous\-Manipulation](https://github.com/asu-iris/Complementarity-Free-Dexterous-Manipulation?utm_source=chatgpt.com)
基于 MuJoCo 的无互补约束接触动力学与模型预测控制灵巧操作环境。

3. **NeuralFeels\-MuJoCo**
[https://github\.com/andomeder/neuralfeels\-mujoco](https://github.com/andomeder/neuralfeels-mujoco?utm_source=chatgpt.com)
MuJoCo 触觉仿真环境，面向指尖触觉感知、物体几何重建与多指操作。

4. **tactile\_envs**
[https://github\.com/carlosferrazza/tactile\_envs](https://github.com/carlosferrazza/tactile_envs?utm_source=chatgpt.com)
基于 MuJoCo/Gymnasium 的触觉操作环境，包含方块旋转、Egg、Pen 等典型灵巧操作任务。

5. **XHand Manipulation**
[https://github\.com/Hu\-xiao\-max/xhand\_manipulation](https://github.com/Hu-xiao-max/xhand_manipulation?utm_source=chatgpt.com)
XHand 三指灵巧手 MuJoCo 环境，结合指尖触觉感知与在手旋转，并支持 Sim\-to\-Real。

6. **mjlab\_hand**
[https://github\.com/ruoyiqiao/mjlab\_hand](https://github.com/ruoyiqiao/mjlab_hand?utm_source=chatgpt.com)
基于 MuJoCo 的多灵巧手操作框架，支持 Allegro、LEAP、Shadow 等手型及 In\-Hand Rotation、ContactExplorer 等任务。

7. **MuJoCo Dex Bench**
[https://github\.com/SameerSphere/mujoco\-dex\-bench](https://github.com/SameerSphere/mujoco-dex-bench?utm_source=chatgpt.com)
面向五指灵巧操作的 MuJoCo benchmark，提供多种手型和转笔、插孔等 dexterous manipulation 任务。

> （注：部分内容可能由 AI 生成）

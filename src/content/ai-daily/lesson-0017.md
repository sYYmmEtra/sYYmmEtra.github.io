---
id: lesson-0017
source:
  file: lessons/2026-07-21.md
  section: 1
  hash: sha256:70d19f70b76200ad760c2c643c497a765b61f3617e29b404df51b6e1d491c66a
lesson: 17
date: 2026-07-21
track: B
depth: L2
titleZh: 对齐算法：RLHF / PPO / DPO
titleEn: "Alignment Algorithms: RLHF, PPO, and DPO"
summaryZh: 新主题(合龙day14 β·KL + day9 奖励模型)。对齐=给"最像互联网的续写机"装人类偏好方向盘;经典RLHF三段=SFT出π_ref→Bradley-Terry从偏好对(x,y_w,y_l)学标量奖励RM(=day9学出来的验证器,故有reward hacking洞)→RL最大化 E[r_φ]−β·KL(π_θ‖π_ref);β·KL缰绳防Goodhart(离π_ref越远罚越大),是day14反向KL(mode-seeking→解释RLHF后风格趋同)。PPO显存痛=policy/冻结π_ref/RM/critic四模型同驻(70B≈280B)→2025-26转GRPO砍critic用组内归一化(DeepSeek-R1)。DPO核心=带KL的RL有闭式解π*∝π_ref·exp(r/β),反解r=β·log(π*/π_ref)+β·logZ,塞回BT时logZ因只依赖奖励之差而抵消→奖励=策略自身相对π_ref的对数比(隐式奖励r̂),得纯离线二分类损失−logσ(r̂_w−r̂_l),甩掉RM与RL循环,梯度自带难例权重。反直觉=DPO里KL没消失,是烘焙进β·log(π_θ/π_ref)这项损失结构
summaryEn: Explains how RLHF combines supervised fine-tuning, preference-trained reward models, and KL-regularized policy optimization. It examines PPO’s resource costs and reward-hacking safeguards, then derives DPO as an offline preference-classification objective whose implicit reward preserves the reference-policy constraint without a separate reward model or reinforcement-learning loop.
slug: lesson-0017
tags:
  - rlhf
  - ppo
  - dpo
  - preference-optimization
  - kl-regularization
sourceStatus: unreviewed
sourceStatusHash: sha256:70d19f70b76200ad760c2c643c497a765b61f3617e29b404df51b6e1d491c66a
metadataStatus: current
metadataSourceHash: sha256:70d19f70b76200ad760c2c643c497a765b61f3617e29b404df51b6e1d491c66a
featured: false
---
# 📅 2026-07-21 · 讲17 · 轨道B · 对齐算法：RLHF / PPO / DPO · 深度 L2

> **TL;DR**：预训练只造出"最像互联网的续写机",不是助手;对齐就是**给这台续写机装上人类偏好的方向盘**。经典 RLHF = SFT→奖励模型→带 `β·KL` 缰绳的 RL(逐字接住讲14的 `β·KL(π_θ‖π_ref)` 与讲9的奖励模型),而 **DPO 的魔法是:KL 正则问题有闭式最优解,于是可以把"奖励"反解成"策略自己",从而甩掉奖励模型和整个 RL 循环,退化成一个偏好对上的二分类损失。**

---

## 一、核心概念精讲（L1 精炼 → L2 主体）

### L1 · 为什么要对齐(30 秒直觉)

预训练目标(讲14: 最小化 `KL(真实文本分布 ‖ 模型)` = 极大似然)只教会模型一件事:**给定前文,续写出最像训练语料的下一个 token**。它是一台"模仿者",不是"助手"——你问它问题,它可能续写出**另一个问题**(因为互联网上问题后面常跟问题)。

对齐(alignment)= 在这台模仿者之上,叠一层**"人类更喜欢哪个回答"**的信号,把它从"续写分布"掰向"有用+无害+诚实的助手分布"。经典范式三段走:

```
   预训练大模型
        │  ① SFT(监督微调):几万条 (指令→示范回答),先学会"回答的格式"
        ▼
   π_ref  ←── 这就是对齐的起点 & 参照系(reference policy),之后全程冻结
        │  ② 训一个奖励模型 RM:从人类偏好对学一个打分器 r_φ(x,y)
        │  ③ RL:让模型去"多拿 RM 高分",但用 KL 拴住别跑太远
        ▼
   对齐后的 π_θ
```

> 术语:**policy(策略)** = 语言模型本身,在 RL 视角下它是一个"给定状态(prompt+已生成)、输出动作(下一 token)概率"的策略函数。**reference policy `π_ref`** = SFT 后冻结的那份,当参照系用。

### L2 · 机制拆解

#### (1) 奖励模型 RM:把"偏好"变成一个可微分数(接讲9)

人类很难给"这个回答值 7.3 分"这种绝对分,但**很容易做二选一**("A 比 B 好")。所以标注数据是**偏好对** `(x, y_w, y_l)`(w=win 胜出,l=lose 落败)。怎么从"相对排序"学出"绝对分数"?——**Bradley-Terry 模型**:

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$

即"A 胜过 B 的概率 = sigmoid(两者分差)"。用极大似然(又是交叉熵,讲14)训 `r_φ`,它就学出了一个标量打分器。**这正是讲9说的"学出来的验证器(ORM)"**,也正因为它是学出来的、有洞,才有讲9的 reward hacking 隐患。

#### (2) RL 阶段:带缰绳的爬坡——本讲与讲14 的合龙点

拿到 `r_φ` 后,RL 阶段要最大化的目标是:

$$\max_{\pi_\theta}\; \mathbb{E}_{x,\,y\sim\pi_\theta}\big[\,r_\phi(x,y)\,\big]\;-\;\beta\,\underbrace{D_{\mathrm{KL}}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\mathrm{ref}}(\cdot|x)\big)}_{\text{讲14 的反向 KL,mode-seeking}}$$

**第一性原理——为什么要这条 `β·KL` 缰绳?** 只有第一项的话,策略会为了刷 RM 高分不择手段:RM 是个有洞的近似(讲9),策略是不知疲倦的优化器,二者相乘 = **Goodhart / reward hacking**——模型会滑向"RM 爱打高分但其实是胡言乱语"的角落,同时**分布崩塌**(collapse 成几句万能高分废话)。`KL(π_θ‖π_ref)` 是一根弹簧:**离参照系越远、惩罚越大**,逼模型"在 SFT 分布的邻域内小步改进"。`β` 是缰绳松紧:大 β=保守(几乎不动)、小 β=激进(易 hack)。

注意这里是**反向 KL**(讲14: mode-seeking,倾向锁定少数高奖励模式),这解释了 RLHF 后模型"回答风格趋同、更套路化"的经验现象。

**PPO 怎么把目标变成更新?** 用 Proximal Policy Optimization:采样→用 RM 打分→算优势→带 clip 的梯度上升,clip/KL 一起构成"信任域"防止一步跨太大炸掉。**工程账单很痛**:显存里同时驻 4 个模型——① 训练中的 policy ② 冻结的 π_ref ③ 奖励模型 RM ④ critic(价值网络)。70B 的 policy ≈ **280B 参数量级**的权重+优化器状态(呼应讲12"显存才是真账单")。这就是 2025–26 转向 **GRPO**(DeepSeek-R1 用)的直接动因:砍掉 critic、用"组内归一化"估优势,省一个模型的显存。

#### (3) DPO:把 RL 循环整个拆掉的一步代数魔术

DPO(Direct Preference Optimization)问了一个绝妙的问题:**上面那个带 KL 的优化问题,最优解长什么样?** 答案是闭式的(经典结果):

$$\pi^*(y|x)=\frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\,\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big)$$

"最优策略 = 参照策略被奖励指数地重新加权"。其中 `Z(x)=Σ_y π_ref·exp(r/β)` 是配分函数,要遍历所有可能回答,**算不出来**——这本来是死路。DPO 的关键一跳:**把这个式子反解成 `r`**:

$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)}+\beta\log Z(x)$$

再把它塞回第(1)步的 Bradley-Terry 里——**因为 BT 只依赖奖励之差 `r(x,y_w)−r(x,y_l)`,那个算不出来的 `β log Z(x)` 直接抵消了!** 于是"奖励"被彻底表达成了"策略自己相对 π_ref 的对数比",奖励模型这一环被吸收进策略里。把 `π*` 换成待训练的 `π_θ`,对观测到的偏好做极大似然,得到 **DPO 损失**:

$$\mathcal{L}_{\mathrm{DPO}}=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\underbrace{\beta\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}}_{\hat r_\theta(x,y_w)}-\underbrace{\beta\log\tfrac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}}_{\hat r_\theta(x,y_l)}\Big)\Big]$$

**读懂它**:括号里 `r̂_θ = β·log(π_θ/π_ref)` 叫**隐式奖励(implicit reward)**——模型自己就是奖励模型。整个式子就是"让胜出回答的隐式奖励 > 落败回答的隐式奖励"的一个 sigmoid 二分类。**没有 RM、没有采样、没有 RL 循环、纯离线**,显存里只剩 policy + 冻结的 π_ref 两个模型。

**梯度的漂亮之处**:`∇L_DPO` 前面天然带一个权重 `σ(r̂_l − r̂_w)`——当模型**当前排错了**(给落败回答的隐式奖励反而更高)时权重大、猛推;排对了则权重小、轻推。**自适应难例加权,无需手调**。

### 🔬 完整走一遍(一个偏好对怎么被 DPO 拉动)

- **prompt** `x`:"用一句话解释递归。"
- **y_w(胜)**:"递归就是函数调用自己,直到撞上一个不再调用自己的 base case。"
- **y_l(败)**:"递归是一种编程技术,非常有用,很多算法都用它,你应该学一学。"(正确但空洞)

前向一遍,拿到四个对数概率(示意,单位 nats,值为整句 log-prob):

| | `log π_θ` | `log π_ref` | 隐式奖励 `r̂=β·(logπ_θ−logπ_ref)`, β=0.1 |
|---|---|---|---|
| y_w | −12.0 | −12.5 | 0.1×(+0.5)=**+0.05** |
| y_l | −9.0 | −9.4 | 0.1×(+0.4)=**+0.04** |

分差 `r̂_w − r̂_l = 0.05 − 0.04 = +0.01` → `σ(0.01)≈0.5025` → 单样本 loss `= −log 0.5025 ≈ 0.688`。因为分差几乎为 0、模型还没学会偏好,loss 接近 `log2`,梯度权重 `σ(−0.01)≈0.4975` **很大**。这一步梯度会**同时抬高 `log π_θ(y_w)`、压低 `log π_θ(y_l)`**——注意它拉的是"**相对 π_ref 的比值**",不是绝对概率:π_ref 那一列冻结不动,纯当锚点。多轮之后,`r̂_w − r̂_l` 被推到 +3 以上,`σ→0.95+`,loss 趋近 0,这个对就"学会"了。这正是缰绳的体现:模型可以拉开 w/l 的差距,但每个回答本身不能离 π_ref 太远(否则 `log(π_θ/π_ref)` 爆炸,别的对会把它拽回来)。

### ⚠️ 常见误区

1. **"DPO 里没有 KL 了"**——错。KL 没消失,是被**解析地烘焙进了损失结构**:`β·log(π_θ/π_ref)` 这一项就是 KL 约束的化身,`β` 还是那个缰绳松紧。DPO ≠ 无正则,而是把正则从"训练时的动态惩罚"变成"闭式解里的固定形态"。
2. **"DPO 全面替代 RLHF"**——不成立。DPO 是**离线**的、只在数据集覆盖的回答上学;它拿不到"策略跑偏后新采样出的坏样本"的在线反馈,故有 **length exploitation(偏爱长回答)**、对 π_ref 依赖敏感等毛病。在线 RLHF/GRPO 在可持续采样打分的场景仍更强。二者是"离线简单 vs 在线强但贵"的取舍,不是替代。
3. **"奖励模型 = 一个更聪明的模型"**——它只是个**标量打分头**,只会说"这个比那个好",不产生文本。
4. **把 DPO 的 `β` 当学习率**——它是**KL 缰绳强度**(偏好差异被放大的尺度),不是步长;调大反而让模型更贴 π_ref、更保守。

---

## 二、最新动态 / 论文速览

> ⚠️ 已联网,arXiv 编号均**格式自洽**(YYMM 与所述月份一致),但**逐条 PDF 未逐一开阅核内容**;且受检索所限,**均非近 30 天**(最新为 2026-05),取其为该主题最相关的权威近作。请以"方向指针"看待。

1. **From RLHF to Direct Alignment: A Theoretical Unification** · arXiv 2601.06108 · 2026-01 —— **为什么重要**:DPO、IPO、KTO、SimPO…方法爆炸让人选不动手。此文论证它们其实塌缩为**三条正交轴**的选择:① 偏好模型(用什么似然,BT? KT?)② 正则机制(如何约束偏离 π_ref)③ 数据分布(在线 vs 离线)。**与今天的关系**:正好是今天"RLHF vs DPO"这张地图的坐标系——你会发现今天讲的 `β·KL` 就是"轴②",Bradley-Terry 就是"轴①"。

2. **Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization** · arXiv 2510.01555 · 2025-10 —— **为什么重要**:指出 GRPO 里常用的 "k3 as loss" KL 估计其实是**有偏的一阶近似**,而 PPO 的 "k1 in reward" 才是反向 KL 的原则性损失;离线实现还因忽略重要性采样而有偏。**与今天的关系**:直接把讲14 的"反向 KL / KL 当损失还是当奖励"落到 RLHF 工程实现的对错上,是理解 `β·KL` 缰绳"怎么算才不偏"的进阶。

3. **A Unified Pair-GRPO Family: From Implicit to Explicit Preference Constraints** · arXiv 2605.06375 · 2026-05 —— **为什么重要**:把 DPO 的**隐式偏好**约束与 GRPO 的**显式 RL** 缝到一个家族里,追求更稳更通用的对齐。**与今天的关系**:正是今天"DPO(离线隐式) vs PPO/GRPO(在线显式)"取舍的最新弥合尝试,指明前沿不是二选一而是融合。

4. **Autoregressive Direct Preference Optimization** · arXiv 2602.09533 · 2026-02 —— **为什么重要**:一个新 DPO 变体,重申 DPO"甩掉显式奖励模型省算力"的定位并在自回归结构上改进。**与今天的关系**:说明 DPO 家族到 2026 仍在活跃演化,今天的标准 DPO 只是这条支线的起点。

---

## 三、🎯 留给明天的钩子

- **今天点到没展开的**:PPO 的信任域/clip 具体怎么算优势、GRPO 如何"组内归一化"省掉 critic;DPO 的 length exploitation 与 IPO/KTO/SimPO 各修哪个洞(对应动态①的三条轴)。
- **建议下次深入(三选一)**:
  - **B 轨复访升级 · 本主题 L2→L3**:PPO↔GRPO↔DPO 的**工程取舍**(在线 vs 离线、显存账、reward hacking 抵抗力、length bias 治理),把动态①的"三轴框架"讲透 + 挑一个变体(SimPO 去 π_ref / KTO 单边偏好)完整推一遍。
  - **B 轨新题 · 微调理论 SFT / PEFT / LoRA 数学原理**:回填学习者最熟的 LoRA 到"低秩更新 ΔW=BA 为何 work",与今天的 SFT 起点接上,补齐"对齐前一段"。
  - **C 轨 · 验证器/奖励模型 L3→L4(讲9 续)**:GRPO/PPO 如何把奖励变成梯度更新的完整闭环 + GenPRM 生成式验证,与今天 RL 阶段直接对接。

---

## 四、📚 延伸阅读

- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*(DPO 原始论文,今天(3)节推导的出处,强烈建议精读它的附录闭式解推导)。
- Ouyang et al., *InstructGPT / Training LMs to follow instructions with human feedback*(RLHF 三段式的奠基之作,今天(1)(2)节的工程原型)。
- Sebastian Raschka, *The State of Reinforcement Learning for LLM Reasoning*(magazine.sebastianraschka.com,PPO→DPO→GRPO 演化的高质量科普综述)。

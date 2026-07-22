---
id: lesson-0014
source:
  file: lessons/2026-07-18.md
  section: 1
  hash: sha256:7c9c3ca1f32ea3f460e67fd8be3be2947a4f5395e589331257ae78cacb2ce7fb
lesson: 14
date: 2026-07-18
track: B
depth: L2
titleZh: 信息论基础：熵 / 交叉熵 / KL 散度
titleEn: "Information Theory Fundamentals: Entropy / Cross-Entropy / KL Divergence"
summaryZh: 信息=意外(−log p);熵H(p)=平均意外=最优编码比特下界(香农);交叉熵H(p,q)=用错模型q编码真实p的账单,one-hot下=NLL=MLE(回填day5),PPL=exp(CE);KL(p‖q)=H(p,q)−H(p)≥0=多付的账,不对称非距离,监督训练=最小化KL(真实‖模型);前向KL=mode-covering(SFT)/反向KL=mode-seeking(RLHF的β·KL(π_θ‖π_ref)牵引绳,合龙day17),经典方向直觉2025-26被实证反驳;InfoNCE=(N+1)选1交叉熵=互信息I(q;k⁺)下界(合龙day11);τ直接调分布熵。反直觉=KL不是距离(不对称)
summaryEn: Explains self-information as surprise, entropy as average surprise and the lower bound for optimal coding, and cross-entropy as the cost of encoding a true distribution with a model. Derives KL divergence as excess cost, emphasizing its nonnegativity and asymmetry, and connects these ideas to NLL, MLE, perplexity, InfoNCE, temperature, SFT, and RLHF.
slug: lesson-0014
tags:
  - information-theory
  - entropy
  - cross-entropy
  - kl-divergence
  - infonce
sourceStatus: unreviewed
sourceStatusHash: sha256:7c9c3ca1f32ea3f460e67fd8be3be2947a4f5395e589331257ae78cacb2ce7fb
metadataStatus: current
metadataSourceHash: sha256:7c9c3ca1f32ea3f460e67fd8be3be2947a4f5395e589331257ae78cacb2ce7fb
featured: false
---
# 📅 2026-07-18 · 讲14 · 轨道B · 信息论基础：熵 / 交叉熵 / KL 散度 · 深度 L2

> **TL;DR**：信息 = 意外程度（surprise），一个事件的信息量是 `−log p`；**熵 H(p)** 是这份意外的平均值，也是"最优编码所需的平均比特数"这一物理下界。**交叉熵 H(p,q)** 是"你用错误的世界模型 q 去编码真实世界 p"要付的账单，永远 ≥ 熵，多付的那部分正好是 **KL 散度 D(p‖q)=H(p,q)−H(p)≥0**。这一条恒等式就是你已经学过的三件事的共同源头：day5 语言模型的**交叉熵损失**、day11 对比学习的 **InfoNCE softmax**，以及即将学的 RLHF **KL 惩罚**——它们都只是这把尺子在不同场景的化身。

---

## 一、核心概念精讲（L2）

### 0. 为什么软件工程师该认真学这一课
你天天写的 loss 函数叫 `cross_entropy`，天天听到的 `KL penalty`、`perplexity`、`InfoNCE`，**全部是信息论里的同一族量**。不把它们打通，你会把 day5、day11 当成三件互不相干的工程 trick 死记；打通后它们坍缩成一个恒等式。这一讲就是回填 day11 结尾我给你留的钩子——"InfoNCE 的 softmax 到底从哪来"。

### 1. 第一性原理：信息 = 意外（self-information）
先问最根本的问题：**一条消息携带多少"信息"？** 香农（Shannon）的答案反直觉但极其扎实——信息量 = **意外程度**。

- "太阳明天从东边升起"：p≈1 → 你毫不意外 → **信息量≈0**。
- "明天本地日食"：p 极小 → 你大吃一惊 → **信息量巨大**。

于是把"意外"定义成概率的**减函数**，再要求"两个独立事件一起发生的信息量 = 各自信息量之和"（可加性），唯一满足这两条的函数就是对数：

```
自信息 self-information:  I(x) = −log p(x)      （p 越小，−log p 越大；p=1 时为 0）
```

- **单位（中英对照）**：`log₂` → **bit（比特）**；`ln`（自然对数）→ **nat（奈特）**。ML 里损失几乎都用 nat（因为 `exp/ln` 好求导），只在讲"信息量/压缩"时换算成 bit。1 nat ≈ 1.44 bit。
- **直觉锚点**：`−log₂ p` = "要把这个事件从 `1/p` 个等概率可能里精确指认出来，需要问几个是非题（几比特）"。p=1/8 → 3 个是非题 → 3 bit。

### 2. 熵 H(p)：平均意外 = 编码长度的物理下界
单个事件的意外是 `−log p`。一整个分布的**平均意外**就是熵：

```
H(p) = E_{x∼p}[−log p(x)] = −Σ_x p(x) log p(x)
```

**两个必须记住的解读：**
1. **不确定性度量**：分布越"平"（均匀）熵越大，越"尖"（集中）熵越小。one-hot 分布 H=0（毫无悬念）；均匀分布熵最大。
2. **香农源编码定理（第一性原理级结论）**：H(p) 是"用最优编码方案压缩 p 的样本，每个样本平均所需比特数"的**理论下界**——你再聪明的压缩算法也压不到 H 以下。**熵不是抽象数字，它是压缩的物理极限。** 这也是"高熵=信息多=难压缩"的由来（随机噪声不可压缩，因为它熵满）。

> **完整走一遍（熵）**：设某位置真实的下一个词分布 p = [0.5, 0.25, 0.125, 0.125]（对应 猫/狗/鱼/鸟）。
> H(p) = −(0.5·log₂0.5 + 0.25·log₂0.25 + 0.125·log₂0.125 + 0.125·log₂0.125)
> = −(0.5·(−1) + 0.25·(−2) + 0.125·(−3) + 0.125·(−3))
> = 0.5 + 0.5 + 0.375 + 0.375 = **1.75 bit**。
> 物理含义：这个位置平均要 1.75 个是非题才能定下来；最优编码给"猫"1 bit、"狗"2 bit、"鱼/鸟"各 3 bit（正好 `−log₂p`，这就是 Huffman 编码在干的事）。

### 3. 交叉熵 H(p,q)：拿错模型去编码要付的账
现实里你**不知道真实分布 p**，只有一个模型 `q`（你的神经网络）。如果你用 q 的"最优编码"（给每个词分配 `−log q(x)` 比特）去编码真正来自 p 的数据，平均要付：

```
H(p,q) = E_{x∼p}[−log q(x)] = −Σ_x p(x) log q(x)
```

这就是**交叉熵**：*用 q 的密码本，编码 p 的世界*。注意期望仍对**真实 p** 取，但代价用 **q** 的自信息。

**关键联系（回填 day5）**：训练语言模型时，每个位置真实标签是 one-hot（真实下一个 token 是"猫"→ p=[1,0,0,0]）。代入：
```
H(p,q) = −Σ p_i log q_i = −1·log q(猫) − 0·(…) = −log q(猫)
```
**交叉熵损失退化成"目标 token 的负对数似然 NLL"**——这正是 day5 里我们说的 loss。而"最小化交叉熵"⇔"最大化训练数据的对数似然"⇔**MLE（极大似然估计）**。三个名字，一件事。

- **Perplexity（困惑度）= exp(交叉熵)**：把 nat 换回"等效分支数"。CE=1 bit → PPL=2，意思是"模型的犹豫程度相当于在 2 个等概率选项里瞎猜"。PPL 是交叉熵的人话版。

### 4. KL 散度 D(p‖q)：多付的那部分账 = 两个分布的"距离"
你用错模型 q 编码，比用真模型 p 编码，**多付了多少**？做个减法：

```
D_KL(p‖q) = H(p,q) − H(p) = Σ_x p(x) log [ p(x) / q(x) ]
```

- **含义**：因为"信错了分布"而额外付出的比特数 = q 偏离 p 的程度。
- **两条铁律**：
  1. **D_KL ≥ 0**，当且仅当 p=q 时为 0（Gibbs 不等式）。→ 所以 `交叉熵 ≥ 熵`，账单永远不小于物理下界。
  2. **不对称：D(p‖q) ≠ D(q‖p)**。所以它**不是真正的距离**（不满足对称性和三角不等式），叫"散度 divergence"而非"distance"。这个不对称性下面会变成 RLHF 的关键设计选择。

**联系（再回填 day5）**：one-hot 标签下 H(p)=0，于是 `D_KL(p‖q) = H(p,q) − 0 = 交叉熵`。**监督训练里最小化交叉熵，本质就是最小化 `KL(真实分布 ‖ 模型分布)`**，把模型分布往数据分布上拽。

### 5. 前向 KL vs 反向 KL：mode-covering vs mode-seeking
既然不对称，"把谁放前面"就是一个真实的工程决策，效果截然不同（设 p=真实/目标，q=我们要学的模型）：

| | 前向 KL `D(p‖q)` | 反向 KL `D(q‖p)` |
|---|---|---|
| 期望对谁取 | 真实 p | 我们的 q |
| 惩罚模式 | **p>0 而 q→0 时爆炸**（惩罚"漏掉"） | **q>0 而 p→0 时爆炸**（惩罚"乱编"） |
| 行为 | **mode-covering / 覆盖**：q 被迫盖住 p 的每个峰，宁可摊平也不遗漏 | **mode-seeking / 抓主峰**：q 可只抓 p 的一个峰，果断丢掉其余 |
| 典型场景 | **MLE / SFT**（最大似然、交叉熵监督） | **RLHF 的 KL 惩罚**：`reward − β·KL(π_θ‖π_ref)`，让策略别跑太远又允许它锐化到高奖励模式 |

**直觉**：前向 KL 是"课代表"——老师（p）讲的每个点都要照顾到，容易讲得稀（over-smooth）；反向 KL 是"押题王"——赌一个最可能的答案往死里打，容易钻牛角尖（塌缩到单一模式，over-confident）。RLHF 用反向 KL 当"牵引绳"：`π_ref` 是微调前的基座，`β·KL(π_θ‖π_ref)` 惩罚项防止策略为刷奖励而胡言乱语（day9 讲的 reward hacking 的第一道缰绳）。
> ⚠️ **诚实标注（学界争论）**：上面"前向=覆盖、反向=抓峰"是经典教科书直觉；但 2025–2026 有多篇工作（见下节）**实证反驳**它并非 KL 方向的内在属性，mode 多样性还受正则强度、reward 缩放等共同支配。当直觉用，别当定理背。

### 6. 合龙 day11：InfoNCE 就是一个交叉熵
day11 我们说对比学习的损失 InfoNCE 是"(N+1) 选 1 的 softmax 分类"。现在可以说清它**为什么**长这样：

```
InfoNCE = −log [ exp(sim(q,k⁺)/τ) / Σ_j exp(sim(q,k_j)/τ) ]
```

- 括号里的 softmax 是模型给出的分布 `q`（"正样本是第几个"的概率）；真实标签是 one-hot（正样本那一个）→ 这**就是一个交叉熵 / NLL**！所以 InfoNCE = "把正样本认出来"这个分类任务的交叉熵。
- 更深一层：InfoNCE 是 **query 与 正样本 之间互信息 `I(q;k⁺)` 的一个下界**（`I ≥ log N − InfoNCE`）。**互信息** `I(X;Y)=D_KL(p(x,y) ‖ p(x)p(y))` = "联合分布比独立假设多出多少信息" = 两个变量的相关度。所以最小化 InfoNCE ⇔ 最大化互信息下界 ⇔ 让 query 和它的正样本"相互预测性"最强。day11 的对比学习，本质是在最大化互信息。

### 常见误区 / 易错点
1. **"熵越大越好 / 越坏"** —— 都不对。熵只是不确定性度量。生成时你**又想要高熵**（多样性、别塌缩，见下节 RLVR entropy collapse）**又想要低熵**（置信、别乱编）。是权衡不是优化目标。
2. **把 KL 当距离**：不对称 + 无三角不等式。"A 离 B 的 KL"和"B 离 A 的 KL"是两码事，代码里写反方向会得到完全不同的训练行为。
3. **混淆交叉熵和熵**：`H(p,q)` 涉及两个分布，`H(p)` 只有一个。loss 里的 `cross_entropy` 永远是前者（真实 vs 预测）。
4. **以为 `−Σ q log q` 是交叉熵**：那是把 q 自己当真实分布的熵。交叉熵的期望必须对**真实 p**（标签）取，代价对**模型 q** 取——两个角色不能同源。
5. **温度 τ 的位置**：day11 的 τ 把 logits 除小 → 分布变尖 → 熵变低。温度直接调的是分布的熵，这也是"信息论 ↔ 采样"的接口。

---

## 二、最新动态 / 论文速览

`✅ 已联网检索；下列为 arXiv 真实检索条目，编号 YYMM 与日期自洽。因网络策略无法逐一打开单篇页面，正文标注「(页面未逐一核验)」，请引用前自行点开确认。`

1. **PAEC: Position-Aware Entropy Calibration for LLM Reasoning in RLVR** · arXiv 2606.08543 · 2026-06 (页面未逐一核验) —— **为什么重要**：RLVR（day9 讲的可验证奖励强化学习）训练中会发生 **entropy collapse（熵塌缩）**：策略分布过早变尖，pass@1 升但 pass@k 停滞，探索能力枯竭。本文按 token 位置校准熵。**与今天的关系**：这里的"熵"就是本讲的 `H(π)`——熵不是黑板概念，而是决定推理模型会不会"钻死牛角尖"的一线旋钮；今天学的"熵=平均意外/多样性"直接解释了塌缩为何=灾难。

2. **A Comedy of Estimators: On KL Regularization in RL Training of LLMs** · arXiv 2512.21852 · 2025-12 (页面未逐一核验) —— **为什么重要**：论证 **GRPO 把 KL 项放进 loss（用 Schulman 的 k3 估计器）会产生有偏梯度**，并不优化它声称的反向-KL 目标，而且这个 bug 已扩散到主流开源库。**与今天的关系**：正是本讲"前向/反向 KL 不对称 + KL 惩罚"在真实训练框架里的落地与坑；提醒你"KL 放 reward 里"和"KL 放 loss 里"数学上不等价。

3. **Future-KL Regularized GRPO (FRPO)** · arXiv 2601.10201 · 2026-01 (页面未逐一核验) —— **为什么重要**：指出 GRPO 常把 KL 实现成"逐 token 的局部 loss 惩罚"，漏掉了自回归带来的策略梯度信号；对反向 KL 给出"未来 KL 修正"（对 per-token log-ratio 做反向累加），无需 critic，实验里同时提升 pass@16 并维持更高熵、更低策略漂移。**与今天的关系**：把"反向 KL 当牵引绳"这件事做对，直接呼应第 5 节的 RLHF KL 惩罚设计。

> 三条同时指向一个事实：**熵和 KL 不是历史概念，而是 2026 年推理模型训练最前沿的核心旋钮**。今天这把尺子，就是读懂 day9 后续（对齐/RL）的钥匙。

---

## 三、🎯 留给明天的钩子

- **今天点到没展开的**：①香农源编码定理只给了直觉，没证；②互信息 `I(X;Y)` 只当 InfoNCE 的副产品一笔带过，它自己是特征选择、mutual-information 神经估计（MINE）的核心；③**校准 calibration**（模型的置信 vs 真实正确率）与熵、交叉熵深度相关，是评测的硬骨头。
- **建议下次深入**：
  - **本轨道 B 复访升级**：可把今天的 KL 当跳板，正式开「**对齐算法：RLHF / PPO / DPO**」——把 `reward − β·KL(π_θ‖π_ref)` 从"惩罚项"讲到"如何变成梯度更新"，直接吃掉本讲第 5 节和第二节全部三条动态（这是最连贯的下一步，深度 L2）。
  - 或 B 轨「**概率与统计：MLE、贝叶斯、校准 calibration**」，把"交叉熵=MLE"往上游推到估计理论，并补齐 calibration。
  - 或回头把 day5 的**交叉熵损失**做 L3 复访（label smoothing 的信息论解释、focal loss、logit 温度与校准）。

---

## 四、📚 延伸阅读（可选）
- Shannon (1948), *A Mathematical Theory of Communication*：熵/信道容量的原始论文，前 5 页就能读懂"信息=意外"的动机。
- Cover & Thomas, *Elements of Information Theory*, 第 2 章：熵、交叉熵、KL、互信息的标准教材式推导（Gibbs 不等式的证明在这）。
- Schulman, *Approximating KL Divergence* (blog)：k1/k2/k3 三个 KL 估计器的来历，读懂上面第 2、3 条动态的前置。

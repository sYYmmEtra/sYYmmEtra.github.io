---
id: lesson-0002
source:
  file: lessons/2026-07-06.md
  section: 2
  hash: sha256:4a5b566d0951d7028fb37b190d9ed37b139efb404fa1e8e7ba958c3bba59d2e5
lesson: 2
date: 2026-07-06
track: B
depth: L2
titleZh: 注意力机制与 Transformer 架构
titleEn: Attention Mechanisms and Transformer Architecture
summaryZh: 自注意力=softmax(QKᵀ/√d)·V，一次基于内容的软性检索；QKV 同源投影，√d 补偿点积方差防 softmax 饱和，因果掩码即"只看前文"，O(n²) 是瓶颈
summaryEn: Self-attention performs content-based soft retrieval with scaled query-key scores and value aggregation. This lesson explains shared QKV projections, why scaling prevents softmax saturation, how causal masking restricts access to earlier tokens, and why the quadratic sequence-length cost becomes the central architectural bottleneck.
slug: attention-transformer-architecture
tags:
  - attention
  - transformers
sourceStatus: unreviewed
sourceStatusHash: sha256:4a5b566d0951d7028fb37b190d9ed37b139efb404fa1e8e7ba958c3bba59d2e5
metadataStatus: current
metadataSourceHash: sha256:4a5b566d0951d7028fb37b190d9ed37b139efb404fa1e8e7ba958c3bba59d2e5
featured: false
---
# 📅 2026-07-06（第 2 讲）· 轨道B · 注意力机制与 Transformer 架构 · 深度 L2

> **TL;DR**：注意力就是"**让每个 token 用一个可学习的相似度，去加权聚合序列里所有其他 token 的信息**"。昨天说 prompt 是在"条件化自回归分布 p(下一个词 | 前文)"——今天就打开这个黑箱：Transformer 靠**堆叠的注意力层**把"前文"里相关的信息动态拉到当前位置，从而算出那个分布。整条自注意力的核心只有一行公式：`softmax(QKᵀ/√d)·V`。

## 一、核心概念精讲（L2）

### L1 直觉层（快速过）——注意力在解决什么问题

先立一个最小心智模型。在 Transformer 之前，处理序列（如句子）主要靠 RNN/LSTM：信息像"接力棒"一样从左到右一个个传递。问题有二：(1) **长程依赖衰减**——第 100 个词想用第 1 个词的信息，得穿过 99 步传递，早就糊了；(2) **无法并行**——第 t 步必须等第 t-1 步算完。

注意力（Attention）的破局思路是：**取消接力，改成"广播 + 点名"**。每个位置不再依赖邻居逐步传递，而是**一步之内直接"看向"序列中所有位置**，按相关性打分，把最相关的信息加权取回来。

> **一个精准类比**：把一句话想象成一场会议。当前 token 是一个**正在提问的人**，他心里有个"我想找什么"的查询（Query）；在座每个人胸前都挂着一张"我是谁"的名牌（Key）和一份"我肚子里的干货"（Value）。提问者拿自己的 Query 去和每个人的 Key 比对相似度→算出"该听谁的"权重→按权重把大家的 Value 汇总成一句"综合回答"，作为自己这一层的新表示。**注意力 = 一次基于内容的软性数据库检索**（区别于按地址索引的硬检索，它是按"相似度"加权取回所有条目）。

关键洞见：Query、Key、Value **都是从同一批输入向量、用三个不同的可学习权重矩阵线性变换出来的**（这叫**自**注意力 self-attention，因为查询源和被查源是同一个序列）。模型要学的，就是"怎么把一个词投影成好的 Query / 好的 Key / 好的 Value"。

### L2 机制层（今日主体）——把公式一层层拆开

#### 1) 三个投影：从 token 到 Q/K/V

输入是一段长度为 `n` 的序列，每个 token 先经过词嵌入 + 位置编码，得到一个 `d_model` 维向量。堆成矩阵 `X ∈ ℝ^{n×d_model}`。三个可学习矩阵把它投影成三份：

```
Q = X · W_Q     # Query，形状 n×d_k，“我在找什么”
K = X · W_K     # Key，  形状 n×d_k，“我能提供什么标签”
V = X · W_V     # Value，形状 n×d_v，“我实际携带的信息”
```

`W_Q, W_K ∈ ℝ^{d_model×d_k}`，`W_V ∈ ℝ^{d_model×d_v}`。注意 Q 和 K 必须同维 `d_k`（要做点积），V 可以不同维。

#### 2) 打分：Query 与 Key 做点积

两个向量的**点积**衡量它们的相似度/对齐程度（方向越一致、模越大，点积越大）。让第 `i` 个 Query 和第 `j` 个 Key 点积，就得到"位置 i 应该对位置 j 投入多少注意力"的原始分数。一次矩阵乘法 `Q·Kᵀ` 就同时算完了所有 n×n 对：

```
scores = Q · Kᵀ        # 形状 n×n，scores[i][j] = 第 i 个 token 对第 j 个的关注度原始分
```

**这里就是 O(n²) 复杂度的来源**——n 个 Query 各自要和 n 个 Key 比一遍，算力和显存都随序列长度平方增长。这是今天"最新动态"里所有工作想干掉的那个瓶颈。

#### 3) 缩放：为什么要除以 √d_k（一个必须讲透的细节）

公式里那个 `/√d_k` 不是随手加的。假设 Q、K 的每个分量是均值 0、方差 1 的独立随机变量，那么它们的点积 `q·k = Σ_{l=1}^{d_k} q_l k_l` 是 `d_k` 个独立项之和，其**方差正比于 d_k**（标准差 ∝ √d_k）。d_k 一大（比如 64、128），点积的数值就会被放得很大很分散。

问题在于下一步的 **softmax**：当输入里有个别值特别大时，softmax 会"饱和"——几乎把全部权重压给最大的那一项，输出趋近 one-hot。饱和区的**梯度接近 0**，训练就停滞了。除以 √d_k 正好把点积的标准差拉回 1 附近，让 softmax 工作在"有区分度但不饱和"的健康区间。**这就是"是什么 + 为什么"**：√d_k 是对点积方差随维度线性增长的精确补偿。

#### 4) 归一化：softmax 把分数变成权重

对 scores 的**每一行**做 softmax（沿 j 方向），把原始分变成一组和为 1 的非负权重：

```
A = softmax(scores / √d_k)     # 形状 n×n，A[i] 是第 i 个 token 的“注意力分布”，一行加起来=1
```

`A[i][j]` = 第 i 个 token 最终分给第 j 个 token 的权重。这一步让"注意力"成为一次**软性加权平均**（soft，可微，能反向传播），而非硬性选择。

#### 5) 聚合：用权重加权求和 Value

```
Output = A · V        # 形状 n×d_v。第 i 行 = Σ_j A[i][j]·V[j]，即按注意力权重把所有 Value 混合
```

合起来就是那一行著名公式（Vaswani et al. 2017）：

```
Attention(Q,K,V) = softmax( Q·Kᵀ / √d_k ) · V
```

#### 6) 多头注意力（Multi-Head）——为什么要拆成多个头

只做一次上述运算，模型只能学到**一种**"相关"的定义。实际语言里，"相关"有很多种：语法上的主谓一致、指代消解、局部搭配、长程主题……于是把 `d_model` 切成 `h` 份（比如 8 头，每头 d_k = d_model/h），**每个头用自己独立的 W_Q/W_K/W_V**，并行做一遍注意力，再把 h 个输出拼接（concat）后过一个输出矩阵 `W_O` 融合：

```
head_i = Attention(X·W_Q^i, X·W_K^i, X·W_V^i)
MultiHead(X) = Concat(head_1, …, head_h) · W_O
```

直觉：**多头 = 让模型在多个不同的"表示子空间"里同时看不同类型的关系**，最后汇总。计算量和单头基本持平（总维度不变），但表达力大增。

#### 7) 因果掩码（Causal Mask）——把"注意力"变成"只能看过去"

上面是双向的（每个 token 能看到全序列）。但 GPT 类**自回归**模型在预测第 t 个词时，绝不能偷看第 t 个及之后的词（否则就是抄答案）。做法极简：在 softmax **之前**，把 scores 矩阵**上三角（j > i 的位置）全部设成 −∞**。softmax 里 e^{−∞}=0，这些未来位置的权重自动归零。

**这正是昨天那句话的机制落点**：`p(下一个词 | 前文)` 里的"前文"约束，物理上就是这个下三角掩码——它保证每个位置只聚合它左边（含自己）的信息。堆几十层这样的带掩码注意力 + 前馈网络（FFN）+ 残差 + LayerNorm，最后一层接一个到词表大小的线性投影 + softmax，就输出了下一个词的概率分布。

#### 一个"完整走一遍"的最小例子

设序列 3 个 token："猫 / 追 / 老鼠"，`d_k = d_v = 2`（玩具尺寸）。假设经过投影后得到（数字是编的，只为看清流程）：

```
        Q                K                V
猫   [1, 0]          [1, 0]          [10, 0]
追   [0, 1]          [0, 1]          [ 0,10]
老鼠 [1, 1]          [1, 1]          [ 5, 5]
```

现在算**"追"这个 token（Q=[0,1]）的输出**，且用**因果掩码**（"追"只能看"猫"和自己"追"，看不到"老鼠"）：

1. **打分**（Q_追 · Kᵀ）：
   - 对"猫"：[0,1]·[1,0] = 0
   - 对"追"：[0,1]·[0,1] = 1
   - 对"老鼠"：被掩码 → −∞
2. **缩放**（÷√2 ≈ 1.414）：0 → 0；1 → 0.707；−∞ → −∞
3. **softmax**（[0, 0.707, −∞]）：
   - e^0=1，e^0.707≈2.03，e^{−∞}=0；总和≈3.03
   - 权重 A_追 ≈ [0.33, 0.67, 0]
4. **加权求和 Value**：
   - 0.33·[10,0] + 0.67·[0,10] + 0·[5,5] = [3.3, 6.7]

解读：**"追"这个位置最终的新表示 ≈ [3.3, 6.7]**——它把 67% 的注意力放在自己身上、33% 放在"猫"身上，完全没看"老鼠"（被掩码），聚合出一个"以自身动词语义为主、带一点主语'猫'信息"的向量。整个序列的所有位置并行地各算一遍这个流程，就是一层自注意力的全部工作。你看到的机制，在真实的 GPT 里逐字如此，只是 d 更大、层更深、数字是学出来的。

#### 常见误区 / 易错点

- **误区①："注意力权重 = 模型的解释/证据"**。注意力权重能告诉你"信息从哪流向哪"，但学界（Jain & Wallace 2019 等）证明它**不等于可靠的因果解释**——同样的预测常有多套权重能达成，别把注意力热力图当成"模型为什么这么想"的铁证。
- **误区②："QKV 是三种不同的东西"**。在自注意力里它们**同源**，都是 X 的线性投影，区别只在三个权重矩阵。搞混这点就理解不了"自"注意力。
- **误区③：忘了 √d_k 的必要性**。省掉它在 d_k 小时看似没事，d_k 一大就会因 softmax 饱和导致训练不收敛——这是新手复现 Transformer 最隐蔽的坑之一。
- **误区④："注意力自带位置信息"**。**不带**。`softmax(QKᵀ)V` 对输入顺序是置换等变的——打乱 token 顺序，输出只是跟着换位置，注意力本身分不清"猫追老鼠"和"老鼠追猫"。位置信息完全靠外挂的**位置编码**（正弦编码 / 可学习 / RoPE）注入，这也是长上下文外推研究（RoPE 外推）的战场。

## 二、最新动态 / 论文速览

> 今天的主题是 2017 年的经典架构，但"如何绕开它 O(n²) 的瓶颈"是 2026 年最活跃的研究线之一。以下均经检索核实来源。

1. **Efficient Attention Mechanisms for Large Language Models: A Survey** · arXiv:2507.19595（2026-02 最新修订）
   - 为什么重要：这是目前对"高效注意力"最系统的综述，把方向清晰切成两大类——**线性注意力**（用核近似/递归把 O(n²) 降到 O(n)）与**稀疏注意力**（只让每个 token 关注一个子集）——并讨论二者混合。想建立"注意力优化全景图"，这是最好的入口。
   - 与今日主题的关系：正是针对本讲拆出的那个 `Q·Kᵀ`（O(n²)）瓶颈开的药方，读完你就知道今天这行公式在工业界被怎样"魔改"。

2. **Multi-Head Latent Attention (MLA)** · DeepSeek-V2/V3 采用的关键技术（近两年，工程落地代表）
   - 为什么重要：推理时真正压垮显存的是 **KV cache**（每生成一个词都要缓存历史所有 token 的 K、V）。MLA 把 K、V 压缩到一个低维**潜在向量**再按需还原，KV cache 显存降一个量级，是当前长上下文/低成本推理的主力方案之一。
   - 与今日主题的关系：直接作用在本讲的 K、V 两个矩阵上——理解了 KV 是什么，才懂 MLA 在压什么。

3. **MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head** · arXiv:2601.07832（2026-01）
   - 为什么重要：线性注意力一直有"快但表达力弱、打不过标准 Transformer"的老毛病。这篇用 token 级多头的思路补回表达力，代表 2026 年"让线性注意力真正可用"的努力方向。
   - 与今日主题的关系：正是对本讲第 6 节"多头"机制的迁移改造，把多头思想搬到线性注意力上。

4. **Beyond Attention: New Possibilities for AI Architectures** · IEEE Computer Magazine（2026-01）
   - 为什么重要：跳出"优化注意力"，讨论**是否该继续以注意力为主导范式**——点名状态空间模型（SSM/Mamba）等替代品。是感知"注意力会不会被取代"这一趋势争论的好材料。
   - 与今日主题的关系：给今天的主角画了一条"未来可能的边界线"，衔接轨道 C 里"Mamba / 注意力替代品"这条线。

## 三、🎯 留给明天的钩子

- **今天点到但没展开的**：位置编码到底怎么设计（正弦编码 vs RoPE），以及为什么 RoPE 能做长度外推；残差连接 + LayerNorm/RMSNorm 在 Transformer block 里的确切位置（Pre-LN vs Post-LN）与其对训练稳定性的影响；FFN 层（其实占了参数大头）在干什么。
- **建议下次深入**：
  - 轨道 A（下一次 A）→ 承接昨日钩子，讲**结构化输出与 Function Calling / Tool Use**（把"条件化分布"用到工具调用上）。
  - 轨道 B 复访本主题时 → 升级到 **L3 工程权衡层**：FlashAttention 的 IO-aware 分块思想、多头的 KV cache 显存账、MHA/MQA/GQA/MLA 的取舍谱系。
  - 轨道 C（首次 C）可选 → **状态空间模型 / Mamba**，正好接上第 4 条动态里"注意力替代品"的争论。

## 四、📚 延伸阅读（可选）

- 📄 Vaswani et al. (2017) *Attention Is All You Need*（经典原文，本讲公式出处，务必读一遍）
- 🔗 [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)（史上最清晰的可视化图解，配今天的机制看）
- 🔗 [The Annotated Transformer — Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/)（逐行 PyTorch 实现，想动手就照它敲一遍）
- 📄 *Efficient Attention Mechanisms for LLMs: A Survey*（arXiv:2507.19595，想看前沿全景）

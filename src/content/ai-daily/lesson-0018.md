---
id: lesson-0018
source:
  file: lessons/2026-07-22.md
  section: 1
  hash: sha256:89564ae3d5c5d0a036669776c63309fbc0e169d1c73523d52f455cdf95183038
lesson: 18
date: 2026-07-22
track: C
depth: L2
titleZh: 分布式并行训练策略（DP / TP / PP / ZeRO·FSDP）
titleEn: "Distributed Parallel Training Strategies: DP, TP, PP, ZeRO, and FSDP"
summaryZh: 新主题(闭合day8/12/15三钩子)。单卡两大瓶颈=装不下+算不快,沿三正交轴切:DP切batch(每卡全副本→不省显存,只买吞吐,每步all-reduce梯度可重叠)、TP切层内权重矩阵(Megatron:A列切GeLU免通信+B行切→前向后向各1次all-reduce激活,卡在关键路径故必锁NVLink内、度≤8)、PP切层为stage(点对点传激活最便宜、跨节点用,代价=气泡(p-1)/(m+p-1)→micro-batch/1F1B/Zero-Bubble拆∂x∂w填气泡填)。锚点=Adam混合精度显存账16P/参数(fp16参2+梯2+fp32 master4+m4+v4=12→7B=112GB单卡爆)。ZeRO/FSDP=给DP打补丁分片冗余状态:ZeRO-1优化器态/2+梯度/3+参数(=FSDP,16P/N,7B/8卡=14GB✓,代价all-gather通信×1.5)。合龙第一性原理=把通信强度匹配带宽层级:TP最重进NVLink/PP最轻扛跨节点IB,与day8/15同一带宽金字塔世界观。3D(TP8×PP8×DP16=175B/1024卡52%峰值)→5D加EP(day12第4刀)+CP。反直觉=DP不省显存(缩模型是ZeRO)、ZeRO-3与TP都切参数但ZeRO-3先all-gather回完整层再算(通信=参数)而TP算时保持分片(通信=激活)
summaryEn: This lesson explains how data, tensor, and pipeline parallelism divide batches, intra-layer matrices, and model layers, while ZeRO and FSDP shard redundant training state. It compares their memory and communication tradeoffs and shows how hybrid parallelism maps communication intensity to high-bandwidth intra-node and lower-bandwidth inter-node links.
slug: lesson-0018
tags:
  - distributed-training
  - data-parallelism
  - tensor-parallelism
  - pipeline-parallelism
  - fsdp
sourceStatus: unreviewed
sourceStatusHash: sha256:89564ae3d5c5d0a036669776c63309fbc0e169d1c73523d52f455cdf95183038
metadataStatus: current
metadataSourceHash: sha256:89564ae3d5c5d0a036669776c63309fbc0e169d1c73523d52f455cdf95183038
featured: false
---
# 📅 2026-07-22 · 讲18 · 轨道C · 分布式并行训练策略（DP / TP / PP / ZeRO·FSDP） · 深度 L2

> **TL;DR**：单卡装不下、也算不快一个大模型，于是我们沿**三个正交的轴**把工作切开——切**数据**（DP）、切**层内的权重矩阵**（TP）、切**层（深度）**（PP）；而 **ZeRO/FSDP** 是给 DP 打的补丁，把冗余的模型状态也分片掉。选哪把刀、怎么组合，本质是**把通信压力匹配到互连带宽的层级**（NVLink > InfiniBand），这正是 day8/day12「带宽受限」世界观从单卡 kernel 到多卡集群的推广。

---

## 一、核心概念精讲（L1 精炼快过 → L2 主体展开）

### L1 · 一句话直觉：为什么需要"并行"

一张 H100 有 80GB 显存、约 1000 TFLOPS 算力。两件事会把它逼爆：

1. **装不下（memory-bound）**：一个 7B 模型用 Adam 混合精度训练，光"模型状态"就要 **112GB**（下面会算），单卡直接 OOM。
2. **算不快（throughput-bound）**：就算装得下，几万亿 token 的数据用一张卡要跑几十年。

解决办法只有一个字：**切**。但"切什么"有讲究——切错了，通信开销会把并行带来的好处全吃掉。这一讲就是讲清楚**三个可切的轴**，以及它们各自的账单。

> **先建立一个显存账（贯穿全讲的锚）**。混合精度 + Adam 训练一个 P 参数的模型，每个参数要存：
> - fp16 参数 **2P** 字节 + fp16 梯度 **2P** 字节
> - fp32 优化器状态：master 权重 **4P** + Adam 一阶矩 m **4P** + 二阶矩 v **4P** = **12P**（回填 day5 Adam 的 m/v）
> - **合计 ≈ 16P 字节/参数**（著名的"16 bytes per param"）
>
> 所以 7B → 16×7 = **112 GB**（>80GB，单卡装不下）；70B → **1120 GB**（要 14 张卡才勉强放下状态，还没算激活）。**这个 16P 就是所有并行策略要对付的敌人。** 注意：激活（activation）是**另一笔账**，正比于 batch×seq_len×层数，不在这 16P 里——这也是 day15 FlashAttention / activation checkpointing 省的东西。

---

### L2 · 三把刀 + 一个补丁

#### 刀 1 · 数据并行 Data Parallelism（DP）——切 batch

**做法**：每张卡放一份**完整**模型副本，把一个 batch 切成 N 份，各算各的前向/反向，最后一次 **all-reduce** 把 N 份梯度求平均，保证所有副本参数一致。

```
   GPU0            GPU1            GPU2
[整个模型]      [整个模型]      [整个模型]     ← 完整副本(冗余!)
 data[0:B/3]    data[B/3:2B/3]  data[2B/3:B]
    │               │               │
   反向 → 梯度      反向 → 梯度      反向 → 梯度
    └───────── all-reduce 求平均 ──────────┘
```

- **省显存？→ 完全不省。** 每卡都是整份 16P，7B 依旧 112GB/卡。DP 只买**吞吐**（更大有效 batch），不解决"装不下"。
- **通信**：每步一次梯度 all-reduce，量 = 参数量 2P（fp16），且可与反向计算**重叠**，相对便宜。
- ⚠️ **最反直觉、最常见的错**：很多人以为"多卡 DP 就能训大模型了"——不能。DP 不缩模型，装不下就是装不下。**缩模型是下面 ZeRO 干的事。**

#### 补丁 · ZeRO / FSDP——把 DP 里的冗余分片掉

ZeRO（Zero Redundancy Optimizer）的洞察：DP 里 N 张卡存了 N 份**一模一样**的 16P，纯冗余。既然 batch 已经分给各卡，何不把这份状态也**切成 N 片**、每卡只存 1/N，用到时再临时 **all-gather** 回来？三个阶段渐进：

| 阶段 | 分片什么 | 每卡显存（近似） | 7B/8卡 |
|------|----------|------------------|--------|
| ZeRO-1 | 仅优化器状态(12P) | 2P + 2P + 12P/N | 28+14+21 → **约 21GB** |
| ZeRO-2 | +梯度 | 2P + 14P/N | 14+16 → **约 16GB** |
| ZeRO-3 | +参数（全切） | **16P/N** | **14GB** ✓ |

- **ZeRO-3 = FSDP**（PyTorch 原生实现，2024 起 **FSDP2** 用 DTensor 做**逐参数分片**，可与 TP/PP/`torch.compile` 干净组合）。7B 从 112GB/卡降到 **14GB/卡**，轻松装下——这就是"补丁"的威力。
- **代价**：ZeRO-3 前向/反向前要 all-gather 参数、算完丢弃，通信量比纯 DP 约 **×1.5**。**本质是"通信换显存"**（又一次带宽换空间，呼应 day8/day12）。

#### 刀 2 · 张量并行 Tensor Parallelism（TP）——切层内的权重矩阵

**做法**：把**单层内部**的大矩阵乘法切到多卡。以 Megatron 的 MLP `Y = GeLU(XA)·B` 为例：

- A **按列**切成 [A₁|A₂]，两卡各算 `GeLU(X·Aᵢ)`——因为 GeLU 是逐元素的，列切**不需要通信**；
- B **按行**切成 [B₁;B₂]，各算 `GeLU(X·Aᵢ)·Bᵢ`，最后 **all-reduce 求和** 得完整 Y。
- 一层前向 1 次 all-reduce、反向 1 次，共 **2 次**。注意力层则**按 head 切**（天然，每卡管一部分头）。

```
        X (完整,广播)
       ╱          ╲
  X·A₁ (GPU0)   X·A₂ (GPU1)     ← 列切,无通信
  GeLU          GeLU
  ·B₁           ·B₂             ← 行切
    ╲            ╱
     all-reduce 求和 → Y         ← 通信在这里(激活,不是参数!)
```

- **省显存？→ 省**（每卡只存半个矩阵），且真正把**单层的算力**摊开。
- ⚠️ **硬约束**：通信是 all-reduce **激活**、且**卡在计算关键路径上**（算一层等一次通信）。所以 TP **必须待在 NVLink 域内**（H100 NVLink 900GB/s），跨节点走 InfiniBand 会慢 10×+，直接吃光收益。**故 TP 度数几乎不超过 8**（一个节点的卡数，也约束于 head 数）。

#### 刀 3 · 流水线并行 Pipeline Parallelism（PP）——切层（深度）

**做法**：把模型按层分成若干 **stage**，stage 0 放第 1-k 层在 GPU0，stage 1 放第 k+1-2k 层在 GPU1……前向时激活**逐 stage 往后传**，反向时梯度**往前传**，相邻 stage 间只是**点对点**发一小块激活——**通信最便宜**。

- **代价 = 流水线气泡（bubble）**：GPU1 得等 GPU0 算完第一批才能开工，首尾必有空转。朴素 PP 气泡占比 ≈ **(p−1)/(m+p−1)**（p=stage 数，m=micro-batch 数）。
- **解法**：把 batch 切成很多 **micro-batch** 灌进流水线填满气泡（GPipe）；**1F1B** 调度（一前一后交替）压低激活显存峰值；**Zero-Bubble PP**（2024）更进一步——把反向拆成"算输入梯度 ∂x"和"算权重梯度 ∂w"两半，用 ∂w 去填气泡，同步语义下气泡近乎为 0，比 1F1B 吞吐高 ~23%。

```
时间 →
GPU0: F1 F2 F3 F4 |          B4 B3 B2 B1
GPU1:    F1 F2 F3 | F4    B4 B3 B2 B1
GPU2:       F1 F2 | F3 F4 B4 B3 B2 B1
              ↑ 灌 micro-batch 填气泡;左上/右下的空白就是 bubble
```

- **省显存？→ 省**（每卡只放几层），通信便宜（点对点），适合**跨节点**用慢速互连拉开。

---

### 合龙 · 3D / 5D 并行：为什么这样组合

大模型训练**同时用三把刀**（3D parallelism / Megatron 的 PTD-P）。经典配置：**175B 模型 / 1024 卡 = TP8 × PP8 × DP16**，达到约 52% 峰值算力。

**第一性原理——把通信强度匹配到带宽层级**（这是全讲最该记住的一句）：

| 策略 | 通信内容 | 强度/位置 | 该放在哪层互连 |
|------|----------|-----------|----------------|
| **TP** | all-reduce 激活 | 最重,在关键路径,每层 2 次 | **节点内 NVLink**(度≤8) |
| **DP/ZeRO** | all-reduce/all-gather 梯度·参数 | 中等,每步 1 次,可重叠 | 节点间也可 |
| **PP** | 点对点传激活 | 最轻,只在 stage 边界 | **跨节点 InfiniBand** |

一句话：**最费带宽的 TP 塞进最快的 NVLink，最省带宽的 PP 去扛最慢的跨节点链路。** 这与 day8「decode 带宽受限」、day15「FlashAttention 省的是搬运不是算力」是**同一个世界观**——整个高效训练/推理的游戏，都是在 `SRAM < HBM < NVLink < InfiniBand` 这个带宽金字塔上，让字节尽量少跨越慢的那一层。

现代框架（Megatron-Core）已扩到 **5D**：在 TP/DP/PP 之上再加 **EP（专家并行，day12 那把刀，MoE 专用）** 和 **CP（Context/序列并行，长上下文专用）**。**day12 的钩子"EP 到底怎么和 DP/TP/PP 排布"就此闭合**：EP 是第 4 把刀，因为 MoE 省算力不省显存（day12），必须把专家切到多卡，且它的 all-to-all 通信最重，通常也锁在高带宽域。

---

### 🎯 完整走一遍：训练一个 7B，再推到 70B

**场景 A：7B，我有 8×H100(80GB)。**
1. 单卡？模型状态 16×7 = **112GB > 80GB**，OOM。❌
2. 上 **ZeRO-3(FSDP)**：112/8 = **14GB/卡** 放模型状态，剩 66GB 留给激活和 batch。✓ 一把刀搞定，**无需 TP/PP**。这就是"13B 以下基本 ZeRO-DP 一把梭"的经验来源。

**场景 B：70B，同样 8×H100。**
1. ZeRO-3：1120/8 = **140GB/卡 > 80GB**，还是爆。❌ 单靠切数据轴不够了。
2. 组合：**TP=8**（节点内 NVLink，把每层矩阵摊到 8 卡，单层显存/算力 ÷8）+ ZeRO 分片优化器状态。若有多节点，再加 **PP** 跨节点分层、**DP** 复制吞吐。
3. 这就是为什么 **7B 是"一把刀"、70B+ 是"3D 组合刀"** 的分水岭——不是模型变聪明了，是 16P 这笔账把你逼上多轴切分。

**核心结论**：并行策略的选择是一道**显存×带宽的约束满足题**，不是"越多卡越好"。先用 ZeRO 榨干数据轴，装不下再上 TP（锁 NVLink 内），层太多再上 PP（跨节点），MoE 再加 EP。

---

## 二、最新动态 / 论文速览

✅ 已联网并核对来源域名；诚实标注：该主题近 30 天无重磅新作，故取**最新综述 + 系统设计指南**，最新一条为 2026-06，其余为 2025-12 ~ 2026-02，**非严格近 30 天**，具体日期以页面为准。

1. **《Efficient training of large language models on distributed infrastructures: a survey》** · Springer / Vicinagearth · **2026-06** —— 目前最新的分布式训练系统综述，从 GPU 集群、高速网络、分布式存储到并行策略与"长时训练可靠性"一网打尽。**与今天的关系**：把本讲的"三把刀 + 通信/显存优化"放进完整系统栈里看，是继续挖 L3 的地图。
2. **《Distributed Hybrid Parallelism for LLMs: Comparative Study and System Design Guide》** · arXiv:2602.09109 · **2026-02** —— 直接对比各种混合并行组合并给出"系统设计指南"。**与今天的关系**：正是本讲 3D/5D 组合的选型手册，L3 复访首选读物。
3. **FSDP2（DTensor 逐参数分片）× Zero-Bubble PP × TP 的可组合性** · PyTorch / TorchTitan · **2026 持续** —— PyTorch 原生栈让 ZeRO-3(FSDP2)、张量并行、零气泡流水在 `torch.distributed.pipelining` 里干净组合，DTensor 是统一的分片抽象。**与今天的关系**：把本讲三把刀从"论文概念"变成"一份配置就能拼起来"的工程现实。
4. **《A Comprehensive Survey on Distributed Deep Learning Training》** · Preprints.org 202512.2207 · **2025-12** —— 按"并行策略 / 框架(DeepSpeed·Megatron·FSDP) / 通信优化 / 网络互连(NVLink·InfiniBand·RoCE)"四维梳理。**与今天的关系**：给"通信匹配带宽层级"这条主线补上网络互连的硬件细节。

---

## 三、🔁 旧知回顾（间隔重复日 · 第 18 讲）

> 换血：避开最近两次（第 12、15 讲）复习过的 CoT / KV Cache / 对比学习 / Adam / reward hacking / √d 缩放，挑最久没被复习的核心——**day14 信息论（从未复习）**、**day13 结构化输出（上次复习约在第 9 讲）**、**day10 进阶 RAG（上次约第 9 讲）**。问题在前，答案折叠。

- **Q1（07-18 信息论）**：为什么监督训练里**最小化交叉熵损失 = 最大似然估计（MLE）**？在标签是 **one-hot** 的分类/语言建模里，交叉熵会化简成什么、和困惑度 PPL 什么关系？
- **Q2（07-17 结构化输出 / 受约束解码）**：什么是"**格式税（format tax）**"？受约束解码能保证输出 JSON **结构合法**，为什么保证不了**语义正确**？
- **Q3（07-14 进阶 RAG）**：**RRF（Reciprocal Rank Fusion）** 融合多路召回时，为什么按 **rank（名次）**而不是按原始相似度分数融合？**cross-encoder rerank** 为什么只能用在 top-K 候选上、不能直接扫全库？

<details><summary>点开看答案</summary>

- **A1**：给定数据，最大似然要最大化 `Σ log q_θ(x)`（模型对真实样本给的对数概率），等价于最小化 `−Σ log q_θ(x)` 即 **NLL（负对数似然）**。而交叉熵 `H(p,q) = −Σ p(x) log q(x)`：当真实分布 p 是 **one-hot**（真类概率 1、其余 0）时，求和塌成**只剩真类那一项** `−log q(真类)`，正是该样本的 NLL。所以**逐样本平均交叉熵 = 平均 NLL = MLE 目标**——三者是同一件事的不同说法（day14 主线）。**与 PPL 关系**：困惑度 `PPL = exp(平均交叉熵)`，是"模型平均在多少个等概选项里犹豫"的直觉度量，交叉熵降 → PPL 降。（延伸：交叉熵 = 熵 H(p) + KL(p‖q)，one-hot 下 H(p)=0，故此时交叉熵**就等于** KL 散度，最小化交叉熵 = 最小化 KL(真实‖模型)。）

- **A2**：**格式税**指"为了逼模型吐出合法结构（如 JSON），在解码时用 grammar/schema 把不合法的 token 的 logit 置 −∞，这个**掩码本身会在某些位置吃掉'本该出现'的 token**，把模型推向更保守的默认值，导致答案质量/推理能力掉分"——典型实测：结构 100% 合法，但约 11% 的内容语义跑偏。**为什么保证不了语义**：受约束解码只约束**语法/结构**（括号配对、字段名、类型），它管的是"长得像不像合法 JSON"，完全不理解**值对不对**。schema 说 `age` 得是整数，它能保证不吐字符串，但拦不住模型填个错误的年龄。**格式合法 ≠ 内容正确**，这也是 day17 那一讲"合法≠正确"要单独做语义正确率评测的原因。

- **A3**：**RRF 按 rank 融合**是因为不同召回路（dense 向量的 cosine、BM25 的 TF-IDF 分）的**分数尺度根本不可比**——一个是 0~1 的余弦、一个是没有上界的词频分，直接相加等于拿苹果加橘子。RRF 只取**名次**：`score = Σ 1/(k + rank)`（k 常取 60），把"分数"这个不可比的量替换成"排第几"这个可比的量，稳健地把多路结果融成一路。**cross-encoder rerank 只用于 top-K** 是因为它把 (query, doc) **拼在一起做一次完整前向**做逐层交互（精度高，补 day11 双塔"无交互"的洞），但**每个候选都要跑一次模型**，复杂度 O(候选数)——扫千万级全库在计算上不可行；所以流程是"先用便宜的双塔召回 top-100（要 recall）→ 再用贵的 cross-encoder 精排（要 precision）"，两段分工。

</details>

---

## 四、🎯 留给明天的钩子

- **今天点到没展开的**：
  - **1F1B / interleaved 1F1B / Zero-Bubble** 调度的具体时间线排布，以及"反向拆 ∂x/∂w"在 autograd 层怎么实现；
  - **序列并行 SP / 上下文并行 CP** 怎么和 TP 配合省激活显存（长上下文训练专用，与 day15 FlashAttention 正交互补）；
  - **通信-计算重叠**（Centauri 式把集合通信切成 intra/inter-node 分组去 overlap）的工程细节。
- **建议下次深入**：
  - **C 轨把本讲升到 L3**：读 arXiv:2602.09109 系统设计指南 + Megatron-Core 5D 折叠，把"给定模型/集群怎么求最优 TP×PP×DP×EP×CP 配置"讲成一道可算的优化题（**强烈推荐，直接接今天的合龙**）；
  - 或 **C 轨长上下文技术**：RoPE 外推 × 稀疏注意力 × KV 压缩——"省看多少历史"这条轴，与今天"切模型/切数据"正交；
  - 或 **B 轨归一化与残差（LayerNorm/RMSNorm/残差连接）**：补齐 Transformer 稳定训练的最后一块基础拼图，也解释 TP 里为什么 LayerNorm 要特殊处理。

## 五、📚 延伸阅读

- **Megatron-LM: Efficient Large-Scale LM Training on GPU Clusters**（arXiv:2104.04473）—— PTD-P（3D 并行）的奠基论文，TP 的行/列切矩阵推导和 52% 峰值算力的来源就在这里，第 3-4 节值得手推一遍。
- **Zero Bubble Pipeline Parallelism**（arXiv:2401.10241，Sea AI Lab）—— "拆反向填气泡"的原始论文，配合 sail-sg 的开源实现看调度图最直观。
- **DeepSpeed ZeRO 官方文档 / FSDP2 PyTorch 教程** —— 对照本讲第一节的 16P 显存账，看 ZeRO-1/2/3 每一阶到底省下哪几项，动手能立刻验证 14GB/卡 那个数字。

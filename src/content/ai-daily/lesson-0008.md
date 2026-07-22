---
id: lesson-0008
source:
  file: lessons/2026-07-12.md
  section: 1
  hash: sha256:e6b2169088233fe83f21e233f578205c74bfbd4769d057f9426b111a08e1495b
lesson: 8
date: 2026-07-12
track: B
depth: L3
titleZh: KV Cache 与注意力变体（MHA→MQA→GQA→MLA）
titleEn: "KV Cache and Attention Variants: MHA, MQA, GQA, and MLA"
summaryZh: 复访day2 L2→L3(推理侧)。自回归decode复用过去token不变K/V=KV Cache,省的是重复投影的巨大常数(非O(n²)阶),只cache K/V;账单巨大(70B/128K/MHA≈328GB);第一性原理=decode内存带宽受限→每步延迟≈读一遍KV Cache→KV缩k倍≈快k倍;四代=多少Q头共享多少份K/V:MHA(各一份)→MQA(全共享1份,砍n倍但质量掉)→GQA(分G组共享,事实标准,GQA-8省87.5%质量掉<1%)→MLA(不减头,低秩隐向量c现场还原K/V,压缩vs GQA 2.7-4.7×,难点=RoPE须解耦劈两段);反直觉=GQA非砍头是共享KV头
summaryEn: KV caching reuses earlier keys and values during autoregressive decoding, reducing repeated projection work while creating a large memory-bandwidth cost. The lesson compares MHA, MQA, GQA, and MLA as different sharing or compression strategies, clarifies their quality and memory tradeoffs, and explains MLA's positional-encoding complication.
slug: kv-cache-attention-variants
tags:
  - kv-cache
  - attention
  - gqa
  - mla
sourceStatus: unreviewed
sourceStatusHash: sha256:e6b2169088233fe83f21e233f578205c74bfbd4769d057f9426b111a08e1495b
metadataStatus: current
metadataSourceHash: sha256:e6b2169088233fe83f21e233f578205c74bfbd4769d057f9426b111a08e1495b
featured: false
---
# 📅 2026-07-12 · 讲8 · 轨道B · KV Cache 与注意力变体（MHA→MQA→GQA→MLA） · 深度 L3

> **复访升级**：day2「注意力机制」L2 → 今天 **L3**，切入角度＝**推理侧工程 / 显存与带宽**。day2 我们把注意力当成"一次基于内容的软检索"，今天问一个纯工程问题：**把它搬到线上做自回归解码，瓶颈到底卡在哪，怎么省。**

> **TL;DR**：自回归解码时，把每一步算过的 K、V 存起来复用，就是 **KV Cache**——它把解码从 O(n²) 重算降到每步 O(n)，但代价是一块随上下文线性增长、能吃掉几十 GB 的显存。更狠的是，**解码是"内存带宽受限"而非"算力受限"**，所以省 KV Cache＝直接提速。MHA→MQA→GQA→MLA 是同一根轴上不断压缩 KV 的四代方案：从"每个头各存一份"到"低秩隐向量现算现还原"。

---

## 一、核心概念精讲（L3）

### 0. 先接上 day2 的线头

day2 结论：注意力是 `Attention(Q,K,V)=softmax(QKᵀ/√d)·V`，复杂度 O(n²)。但那是**训练/prefill**（一次喂进整段文本）的视角。**线上真正费钱的是"生成"**——一个 token 一个 token 往外吐（decode）。这一步的工程账，day2 完全没算。今天补。

### 1. 为什么需要 KV Cache：把重复劳动缓存下来

自回归生成第 t 个 token 时，模型要对"已生成的全部 t 个 token"做一次注意力。**朴素做法**：每生成一个新 token，就把前面所有 token 重新过一遍 Transformer 算出它们的 Q、K、V——第 t 步要算 t 个 token，总共 1+2+…+n ≈ **O(n²)** 次前向，重复得离谱。

关键观察：**过去 token 的 K、V 一旦算出就不再改变**（因果掩码保证 token i 只依赖它自己和更早的输入，与"未来会生成什么"无关）。所以：

```
KV Cache 的核心不变式：
  第 t 步新到一个 token x_t：
    只算 x_t 的 q_t, k_t, v_t              ← 1 个 token 的投影
    把 k_t, v_t 追加进 Cache               ← Cache 从 t-1 长到 t
    注意力 = softmax(q_t · Kᵀ_cache/√d) · V_cache   ← q_t 去查"整段历史"
  → 每步只做 O(t) 的注意力 + O(1) 的投影，总计 O(n²) 但常数小到几乎只剩一趟矩阵乘
```

> ⚠️ **误区订正**：KV Cache **不改变**注意力的 O(n²) 渐进复杂度（每步仍要和全部历史点积），它省掉的是**对过去 token 的重复投影和重复前向**——即把"每步重算整个前缀"降到"每步只算一个新 token"。省的是**巨大的常数**，不是复杂度阶数。为什么只 cache K、V 不 cache Q？因为 Q 用完即弃：q_t 只在第 t 步用来查询，之后再不需要；而 k_t、v_t 会被之后**每一个**新 token 查询到。

### 2. 真正的账单：KV Cache 有多大

Cache 里存的是每层、每个 KV 头、每个历史 token 的 k 和 v 向量。大小公式：

```
KV_bytes = 2 × L × n_kv × d_head × seq_len × batch × precision_bytes
           │    │    │        │         │
           K和V  层数  KV头数  每头维度   序列长度
```

**完整走一遍**（拿一个 70B 级、MHA 假设的模型估：L=80, n_head=64, d_head=128, FP16=2B）：

- 每 token 每层：`2 × 64 × 128 × 2B = 32 KB`
- 每 token 全模型：`32KB × 80 = 2.56 MB`
- 上下文 **128K** token、batch=1：`2.56MB × 128000 ≈ 328 GB`

**328 GB**——一张 H100 才 80GB 显存，光 KV Cache 就要 4 张卡，模型权重还没算。这就是长上下文推理的头号敌人。（这也正好对上上一条搜索里"128K 下 70B 即便用了 GQA 仍需 ~40GB KV"的数量级。）

### 3. 决定性的第一性原理：解码是"内存带宽受限"

这是 L3 的**核心洞见**，也是所有 KV 压缩技术的动机根源。

- **Prefill**（并行处理 prompt）：一次算几千 token，矩阵×矩阵，**算力受限**（compute-bound），GPU 的 TFLOPS 吃满。
- **Decode**（逐 token 生成）：一次只算 1 个 token，是矩阵×**向量**。GPU 要把**整个** KV Cache 从显存搬进计算单元，却只做极少量乘加。

衡量指标叫 **arithmetic intensity（算术强度）＝ FLOPs / 搬运字节数**。decode 时这个值极低——瓶颈是**把 KV Cache 从 HBM 读出来的带宽**，而非算不算得完。

> **推论（记住这一句，整节都从它推出来）**：decode 阶段，**每步延迟 ≈ 读一遍 KV Cache 的时间**。所以 **KV Cache 缩小 k 倍 ≈ decode 快 k 倍**，顺带省显存、能塞更大 batch。这就是为什么下面四代方案全在"想办法让 KV 更小"——它们不是在省算力，是在省带宽。

### 4. 同一根轴：MHA → MQA → GQA → MLA

四代方案的唯一区别，就是**多少个查询头(Q)去共享多少份 K/V**：

```
        Q头数    KV头数     KV Cache相对大小    代价
MHA      n         n           1×（基准）        质量最好，最费
MQA      n         1          1/n（最省）        所有Q头共享一份KV，质量掉、训练不稳
GQA      n         G          G/n（可调）        分G组，组内共享；甜点区
MLA      n      潜向量c        ~1/(GQA的2~4×)     不共享头，改"低秩压缩+现场还原"
```

**① MHA（Multi-Head）**：day2 讲的原版。n 个 Q 头，各配一个独立 K 头、V 头。表达力拉满，KV Cache 也最大。

**② MQA（Multi-Query）**：极端反向——**所有 Q 头共享同一对 K/V 头**。KV Cache 直接砍 n 倍（上例 64→1）。但太激进：不同 Q 头本该"从不同角度检索"，现在被逼共用一份 K/V，质量下降且训练容易不稳。

**③ GQA（Grouped-Query）**：折中，也是**当下事实标准**。把 n 个 Q 头分成 **G 组**，**组内**共享一对 K/V 头。G=n 退化成 MHA，G=1 退化成 MQA，中间任意插值。Llama-3/4、Qwen、Mistral、Gemma 全用它。典型 GQA-8（8 组）在 64 头模型上把 KV 砍到 1/8（≈ −87.5%），质量掉不到 1%。

> ⚠️ **误区订正**：GQA **不是"砍掉一些注意力头"**——Q 头数量一个没少（表达检索的多样性还在），减少的是**独立 K/V 投影的份数**。是"多个查询共用一套被查资料"，不是"少查几次"。

**④ MLA（Multi-head Latent Attention，DeepSeek 提出）**：换了个思路。前三代都在"减少 K/V 头的**数量**"，MLA 保留概念上的多头，但**不直接存 K/V**——它把每个 token 压成一个**低维隐向量 c**（low-rank latent）存进 Cache，用时再用上投影矩阵**现场还原**出各头的 K、V。

```
存的时候：  c_t = W_down · h_t          # h_t: token隐状态；c_t: 一个小隐向量（Cache只存它）
用的时候：  K_t = W_up_K · c_t ,  V_t = W_up_V · c_t   # 从c_t还原出全部头的K、V
```

Cache 从"n_kv 个头×d_head"缩成"一个 c 的维度"，报告压缩比 vs MHA 约 **7–14×**，vs 已经压过的 GQA 约 **2.7–4.7×**（两个数基准不同，见误区④）。

> **MLA 的隐藏难点：RoPE 不兼容。** RoPE（旋转位置编码）要在 K 上按位置做旋转，而这个旋转是非线性的、**不能和"压缩→还原"这套线性投影交换次序**（`Rotate(W_up·c) ≠ W_up·Rotate(c)`）。DeepSeek 的解法是**解耦（decoupled RoPE）**：把每个头的维度**劈两半**——一半 NoPE（不带位置，走低秩压缩通道）、一半专门承载 RoPE（单独走一个小的、共享的带位置通道）。这样位置信息不进压缩管道，绕开了不可交换问题。这个"劈成带位置/不带位置两段"的技巧是 MLA 工程上最容易踩坑的地方。

### 5. 怎么选（工程权衡表）

| 你的处境 | 选择 | 理由 |
|---|---|---|
| 从头训练、要稳妥的甜点 | **GQA**（G=8 附近） | 生态成熟、kernel 支持最好（vLLM/FlashInfer 都为它专门优化过）、质量损失可忽略 |
| 极致长上下文 / 显存吃紧 / 自研大模型 | **MLA** | 压缩比最高，且能和 FP8 量化(×2)、prefix caching 叠乘 |
| 老模型想省显存又不想重训 | GQA 或 **转换法** | MQA 回退可"均值池化"相邻 K/V 头合并成 GQA；也有 TransMLA / MHA2MLA 等把已有模型转成 MLA 的后处理路线 |

> ⚠️ **误区订正④（压缩比读法）**：看到"MLA 压 14×"和"MLA 压 3×"别以为矛盾——**基准不同**：对比**未压缩的 MHA** 得到大数字，对比**已经压过的 GQA** 得到小数字；再叠加 FP8 量化(×2)会让数字更大。看压缩比先问"相对谁、含不含量化"。

---

## 二、最新动态 / 论文速览

`⚠️ 已联网；下列 arXiv 编号与日期自洽、可核实，但均非严格"近30天"（最新集中在 2025-12～2026-03）；第 4 条中的前沿模型命名（GLM-5.2 / Kimi K2.7 / DeepSeek-V4 等）来自工程博客，**未逐一核实，可能不准**。`

1. **CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling MLA**（[arXiv:2603.17946](https://arxiv.org/pdf/2603.17946)）· 2026-03 —— 研究"把已有模型的注意力**分解**成 MLA 结构"的更优做法（协方差感知 + 秩增强）。**为何重要**：MLA 的落地瓶颈之一是"存量模型怎么转过去"，这类工作让不必从头训练也能吃到 MLA 的压缩红利。**与今天的关系**：正是第 4 节"转换法"那一格的前沿代表。

2. **KQ-SVD: Compressing the KV Cache with Provable Guarantees on Attention Fidelity**（[arXiv:2512.05916](https://arxiv.org/pdf/2512.05916)）· 2025-12 —— 用 SVD 压 KV Cache 并给出**对注意力保真度的可证明保证**，且证明在 GQA 的共享 K 上直接做 SVD 就是最优近似。**为何重要**：多数压缩方法是经验性的，这条给了理论保证。**与今天的关系**：是"在 GQA 之上再叠一层压缩"的正交手段，印证第 4 节末"可叠乘"。

3. **FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving**（[arXiv:2501.01005](https://arxiv.org/pdf/2501.01005)）· 2025-01 —— 推理服务的注意力引擎，含针对 GQA 的 **head-group fusion**：把同组 Q 头合并到线程块的行维度，**一次共享内存加载 KV 就服务整组 Q 头**。**为何重要**：呼应第 3 节"decode 是带宽受限"——kernel 层面减少 KV 重复搬运正是提速关键。**与今天的关系**：让你看到"省带宽"不止靠改架构，也靠写更聪明的 kernel。

4. **架构调查（工程博客，2026-03，⚠️未核实具体命名）** —— 称近 ~80 个开源模型里约 **2/3 属 GQA 家族**，GQA 仍是主流基线；前沿旗舰开始转向 MLA / 稀疏注意力等。**为何重要 / 关系**：给"四代方案"当下占比一个感性坐标——**GQA 是现在，MLA/稀疏是趋势**。（具体模型名未核实，仅作趋势参考。）

---

## 三、🎯 留给明天的钩子

- **今天点到没展开的**：
  - **稀疏注意力**（sparse / block-selection attention）——第 4 条提到的另一条"省"的路线，和 KV 压缩是不同维度（省"看多少历史"而非"每个历史存多大"）。
  - **PagedAttention / vLLM 的显存分页**——今天算的是 KV 有多大，没讲"运行时怎么高效管理这块显存、避免碎片"，这是 serving 的另一半。
  - **RoPE 本身**：今天只说了它和 MLA 不兼容，没讲它的机制与长度外推。
- **建议下次深入**：
  - 轨道 **C** 顺势接「高效训练：FlashAttention、并行策略（DP/TP/PP/ZeRO）」——把今天"带宽受限"的思路从 decode 推广到 training 的 IO-aware。
  - 或轨道 **A**「推理服务与部署：vLLM / PagedAttention / 量化上线」——把今天的显存账变成可运维的系统。
  - 或轨道 **B**「归一化 / RMSNorm 与残差」补 Transformer 结构最后一块基础拼图。

---

## 四、📚 延伸阅读（可选）

- [KV Cache Optimization for LLMs 2026: Engineering Guide](https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide) —— 把 MLA 压缩 × FP8 量化 × prefix caching 如何叠乘讲得很直观（商业博客，数字偏乐观，当直觉用）。
- [Grouped-Query Attention: shrinking the KV cache (zeroentropy)](https://zeroentropy.dev/concepts/grouped-query-attention/) —— GQA 从 MHA/MQA 谱系推导的清爽讲解。
- [QCQA: Quality and Capacity-aware GQA (arXiv:2406.10247)](https://arxiv.org/abs/2406.10247) —— 论证"GQA 随意分组不是最优"，用进化算法找质量感知的分组（2024-06，背景阅读）。

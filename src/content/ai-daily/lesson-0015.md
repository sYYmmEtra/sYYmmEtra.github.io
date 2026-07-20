---
id: lesson-0015
source:
  file: lessons/2026-07-19.md
  section: 1
  hash: sha256:42a207e5b2fbe1f3fe2093ae06ddc222f8159bd29a748e9ebdb0c71e02b9e184
lesson: 15
date: 2026-07-19
track: C
depth: L2
titleZh: FlashAttention（IO-aware 精确注意力：tiling + online softmax + 重计算）
titleEn: "FlashAttention: IO-Aware Exact Attention with Tiling, Online Softmax, and Recomputation"
summaryZh: 新主题(闭合day2/day8/day12钩子)。第一性原理:attention慢在搬运不在算力——标准实现把N×N分数矩阵在HBM来回读写、且softmax是低算术强度elementwise→memory-bound;是day8"decode带宽受限"的训练/prefill侧孪生(算力过剩、带宽稀缺)。三件套:①tiling把Q/K/V切块塞进快而小的SRAM、永不物化N×N,只写回O(N)的O;②online softmax(维护running max m/sum ℓ,新块来了用α=e^{m_old−m_new}回溯修正,数学上逐位等价于一次性softmax→exact非近似)解决"softmax需整行"与"分块流式"的矛盾;③backward用O/m/ℓ重计算S/P(用过剩算力换稀缺带宽/显存)。收账:HBM读写O(N²)→近线性、额外显存O(N²)→O(N)、FLOPs不变(计算复杂度仍O(N²))、加速2-4×。三误区:非降计算复杂度(那是Mamba/线性注意力且近似)、非稀疏丢token、省的不是FLOPs而是IO。谱系FA1(三件套)→FA2(并行划分)→FA3(Hopper异步/FP8)→FA4(2026-03 Blackwell:TensorCore翻倍而exp单元没提速→softmax成头号瓶颈,多项式逼近exp+条件式rescaling砍10×rescale)
summaryEn: FlashAttention accelerates exact attention by tiling Q, K, and V in fast on-chip memory, applying online softmax without materializing the full attention matrix, and recomputing intermediates during backpropagation. It reduces memory traffic and storage while retaining quadratic computational complexity and equivalent attention results.
slug: lesson-0015
tags:
  - flash-attention
  - online-softmax
  - gpu-memory
  - attention-kernels
  - recomputation
sourceStatus: unreviewed
sourceStatusHash: sha256:42a207e5b2fbe1f3fe2093ae06ddc222f8159bd29a748e9ebdb0c71e02b9e184
metadataStatus: current
metadataSourceHash: sha256:42a207e5b2fbe1f3fe2093ae06ddc222f8159bd29a748e9ebdb0c71e02b9e184
featured: false
---
# 📅 2026-07-19 · 讲15 · 轨道C · FlashAttention（IO-aware 精确注意力：tiling + online softmax + 重计算）· 深度 L2

> **TL;DR**：Attention 慢，不是因为算得多，而是因为**搬得多**——标准实现把 N×N 的注意力矩阵在慢速显存（HBM）里来回读写，成了 **memory-bound（内存带宽受限）**。FlashAttention 用**分块（tiling）+ 流式 softmax（online softmax）**在快速片上内存（SRAM）里一次算完、**永不落地 N×N 矩阵**，backward 靠**重计算**省显存。结果：HBM 读写从 O(N²) 降到近线性、显存从 O(N²) 降到 O(N)、速度数倍提升——而且是**数值精确**（exact），不是近似、不是稀疏。这是 day8「decode 带宽受限」的**训练/prefill 侧孪生兄弟**：算力早已过剩，**带宽才是真瓶颈**。

## 一、核心概念精讲（L2）

### 0. 承接：day2 埋下的 O(n²) 炸弹，今天来拆
day2 我们知道自注意力是 `softmax(QKᵀ/√d)·V`，瓶颈是 O(n²)。但"O(n²)"到底卡在哪？很多人以为卡在**算力（FLOPs）**——错。今天的第一性原理是：**它主要卡在内存搬运**。理解这一点，是理解过去三年几乎所有高效 attention 工作的钥匙。

### 1. 第一性原理：GPU 的"内存墙"与算术强度
GPU 有两层内存（先给直觉，术语随后）：
- **HBM（High Bandwidth Memory，高带宽显存）**：大（80GB）但"慢"（A100 ≈ 2 TB/s）。就是我们平时说的"显存"。
- **SRAM（on-chip 片上内存 / shared memory）**：极快（A100 ≈ 19 TB/s，约 HBM 的 10 倍）但**极小**（每个 SM 才 ~192KB，全卡量级几十 MB）。

关键不等式：**GPU 的算力增速 ≫ 内存带宽增速**。现代 GPU 每秒能做的浮点运算，远多于每秒能从 HBM 搬进来的数据。于是一个 kernel 到底是"算力受限（compute-bound）"还是"带宽受限（memory-bound）"，取决于它的**算术强度（arithmetic intensity）= 每搬 1 字节数据能顺带做多少次浮点运算**。

- 强度高（如大矩阵乘 matmul）→ 数据搬进来后能反复算很多次 → 算力受限，GPU 跑满。
- 强度低（如 softmax、dropout、masking 这些 **elementwise 操作**：每个元素读一次、算一两下、写回）→ 时间全耗在搬运上 → **带宽受限，算力单元大量闲置**。

> 这与 day8 是**同一条定律**：day8 说 decode 阶段"每步只算长度 1 的 query 却要读整个 KV Cache"，算术强度极低 → decode 是带宽受限，KV 缩 k 倍则快 k 倍。今天说的是**训练 / prefill（一次性处理整条长序列）**里的 attention kernel，同样栽在带宽上。**"算力过剩、带宽稀缺"是这两讲共享的世界观。**

### 2. 病根：标准 attention 把 N×N 矩阵在 HBM 里来回搬
标准（naive）attention 分三步，每步都要完整读/写 HBM：

```
输入 Q,K,V ∈ R^{N×d}   （N=序列长, d=head维度）
① S = QKᵀ            → 把 N×N 的分数矩阵 S 写回 HBM       （写 O(N²)）
② P = softmax(S)     → 从 HBM 读 S，算 softmax，再写回 P   （读+写 O(N²)）
③ O = P·V            → 从 HBM 读 P，乘 V，得到 O           （读 O(N²)）
```

问题一目了然：中间那个 **N×N 矩阵被反复读写**。N=8192 时，一个 head 的 S 就是 8192²≈6700 万个数（fp16 约 128MB），要在 HBM 里进出好几趟。而 ② 的 softmax 本身是**低算术强度的 elementwise 操作**——搬运多、计算少。所以标准 attention 的墙**不在两次 matmul（那反而算得挺满），而在夹在中间的 softmax 那趟 O(N²) 的 HBM 往返**。

- **显存代价**：materialize（物化，指真把矩阵写进显存）N×N，显存占用 O(N²)，这是长上下文最先爆的地方。
- **速度代价**：HBM 带宽被 O(N²) 的读写占满，GPU 算力单元干等。

### 3. 解法（一）：Tiling —— 别落地整张矩阵，切块塞进 SRAM
既然病根是"往 HBM 写 N×N"，那就**根本不写**：把 Q、K、V 沿序列维切成小块（tile），每次只把**一小块** Q 和一小块 K/V 搬进 SRAM，在 SRAM 里算出这一小块的局部注意力贡献，**累加到输出 O**，只把最终的 O（N×d，线性大小）写回 HBM。N×N 的 S、P **从头到尾只活在 SRAM 里、从不整体存在**。

但这里有个**硬骨头**：softmax 需要**整行**才能算。分母 `Σⱼ e^{sⱼ}` 要等看完这一行**所有** K 才知道；而且为数值稳定要减去**整行最大值** max（否则 e 指数爆炸/溢出）。可我们是一块一块看 K 的，看第一块时根本不知道后面的 max 和分母——怎么办？

### 4. 解法（二）：Online Softmax —— 边走边修正，且数学上完全等价
**核心技巧**，也是 FlashAttention 的灵魂。维护两个"跑动统计量"：running max `m`（目前见过的最大分数）和 running sum `ℓ`（当前 max 基准下的指数和），外加未归一化的输出累积 `O`。每来一个新的 K/V 块：

1. 算这块的局部 max，更新 running max：`m_new = max(m_old, 块内max)`；
2. **回溯修正**：既然基准 max 变大了，之前累积的 `ℓ` 和 `O` 都要乘一个修正因子 `α = e^{m_old − m_new}`（把旧的指数值从旧基准平移到新基准）；
3. 加上本块的贡献，更新 `ℓ` 和 `O`。

看完所有块后，`O / ℓ` 就是精确答案。**关键点：这不是近似！** 它在数学上**逐字节等于**一次性 softmax（只差正常浮点误差）。所以 FlashAttention 是 **exact attention（精确注意力）**——这是它区别于"稀疏/线性注意力（那些是近似）"的根本身份。

**⭐ 完整走一遍（手算验证 online softmax = 标准 softmax）**
设一行 4 个分数 `s=[1,3,2,5]`，对应 value（简化为标量）`v=[10,20,30,40]`。分两块处理：块1=[s1,s2]，块2=[s3,s4]。

*标准做法（一次性）*：全局 max=5，权重 `e^{s−5}=[0.0183, 0.1353, 0.0498, 1]`，和 ℓ=1.2034，
输出 O = (0.0183·10+0.1353·20+0.0498·30+1·40)/1.2034 = 44.383/1.2034 = **36.88**。

*online 做法（流式）*：
- **块1** [1,3]：m₁=3；ℓ₁=e^{1−3}+e^{3−3}=0.1353+1=1.1353；O₁=0.1353·10+1·20=21.353。
- **块2** [2,5]：m₂=max(3,5)=5；修正因子 α=e^{3−5}=0.1353。
  - ℓ₂ = α·ℓ₁ + e^{2−5}+e^{5−5} = 0.1353·1.1353 + 0.0498 + 1 = **1.2034** ✅（与标准的分母一致）
  - O₂ = α·O₁ + e^{2−5}·30 + e^{5−5}·40 = 0.1353·21.353 + 1.494 + 40 = 44.383
- **归一化**：O = O₂/ℓ₂ = 44.383/1.2034 = **36.88** ✅

完全吻合。直觉上，`α=e^{m_old−m_new}` 就是"把之前用旧基准算的所有指数项，统一平移到新基准"的一次性回溯乘法——**看似要改一堆历史，其实只需一个标量修正**，这就是 online softmax 便宜又精确的原因。

### 5. 解法（三）：Backward 用重计算（recomputation）省显存
反向传播需要注意力矩阵 P 来算梯度。但我们前向**故意没存** N×N 的 S/P（那正是要省的显存）。怎么办？**backward 时用前向存下的极少量统计量（O、m、ℓ，都是 O(N) 大小）现场重算 S、P**。

这看起来是"多算一遍、浪费算力"，但记住第一性原理：**瓶颈是 IO 不是 FLOPs**。重算发生在 SRAM 里、不碰 HBM，比"把 N×N 矩阵存进 HBM 再读回来"**更快也更省显存**——用过剩的算力换稀缺的带宽/显存，稳赚。（这是 activation checkpointing 思想在 attention kernel 里的专用化。）

### 6. 收账：到底省了什么
| 指标 | 标准 attention | FlashAttention |
|---|---|---|
| **HBM 读写量** | O(N²) | ≈ O(N²·d²/M)，M=SRAM大小 → **实测近线性** |
| **额外显存（注意力矩阵）** | O(N²)（物化 S/P） | **O(N)**（只存 O、m、ℓ） |
| **计算复杂度 FLOPs** | O(N²·d) | O(N²·d)（**没变**，重算还略增） |
| **数值** | 精确 | **精确**（不是近似） |
| **实测加速（FA2, A100）** | 1× | **2–4×**，长序列更明显 |

### 7. 常见误区（务必点清）
- **❌ "FlashAttention 把复杂度从 O(N²) 降到 O(N)"** —— 降的是 **HBM 访问量和显存占用**，**计算复杂度仍是 O(N²)**（还是要算 N² 个点积）。把 attention 的**计算**降到线性的是 Mamba / 线性注意力那一类（且它们是**近似/换机制**）。
- **❌ "它靠稀疏/丢弃部分注意力来加速"** —— 不，它是 **exact**，一个 token 都不丢，结果与标准实现逐位等价。
- **❌ "它省的是 FLOPs"** —— 恰恰相反，重计算让 FLOPs 略增；省的是**内存搬运**。理解"attention 是 memory-bound 而非 compute-bound"是整个理解的地基。
- **❌ "FlashAttention 只对推理有用"** —— 它对**训练/prefill（长序列一次性并行）**收益最大；decode 单步的带宽问题主要靠 day8 的 KV Cache 变体（MQA/GQA/MLA）和 PagedAttention 解决，二者互补。

### 8. 版本谱系（为下一层铺路）
- **FA1（2022）**：确立 tiling + online softmax + recomputation 三件套。
- **FA2（2023）**：优化并行划分与 work partitioning（减少非 matmul 运算、更好地跨线程块并行），A100 上再快约 2×。
- **FA3（2024）**：吃满 Hopper（H100）的**异步**特性——用 warp-specialization 让 Tensor Core 的 matmul 与 softmax 的搬运/计算**重叠流水**，并支持 FP8。
- **FA4（2026-03）**：为 Blackwell（B200）重构。见下节——硬件"非对称扩展"催生的新瓶颈。

---

## 二、最新动态 / 论文速览

`✅ 已联网并逐条核实（arXiv 编号 YYMM 与日期自洽；FA4 为 2026-03 发布，虽超 30 天但为当前 SOTA 基石，特此标注）`

1. **FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling** · arXiv:2603.05451 / Together AI · **2026-03-05** —— *为什么重要*：Blackwell 上把 attention 推到 **1613 TFLOPs/s、71% 利用率**（FA3 在 Hopper 约 740）。*与今天的关系*：完美印证本讲世界观——Blackwell 的 Tensor Core 翻倍、但**指数单元（MUFU.EX2）几乎没提速**，于是 **softmax 从"两次 matmul 中间的配角"变成头号瓶颈**。FA4 的对策正是本讲主题的延伸：①用 FMA 单元**多项式逼近 exp**、绕开变慢的 SFU；②**条件式 rescaling**——只有 running max 变动大到威胁数值稳定时才回溯修正，把 online softmax 的 rescale 次数砍约 10×（直接优化第 4 节那个 α 修正步）。

2. **TileMaxSim: IO-Aware GPU MaxSim Scoring with Dimension Tiling and Fused Product Quantization** · arXiv:2606.26439 · **2026-06（约 3 周前）** —— *为什么重要*：把 FlashAttention 的 **IO-aware + tiling** 范式迁移到**多向量检索打分（ColBERT 的 MaxSim）**，roofline 分析显示 naive 实现只跑到峰值带宽的 **17.6%**（又一个 memory-bound 铁证）。*与今天的关系*：说明"别物化中间大矩阵、切块在 SRAM 里流式算"是**跨任务的通用招式**，不止用于 attention——还直接呼应 day10 进阶 RAG 里 ColBERT 的 late-interaction。

3. **PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel** · arXiv:2511.22333（ASPLOS '26）· **2025-11** —— *为什么重要*：pack–forward–**merge** 三段式加速 decode，最后一步靠 **online softmax** 把各 tile 的部分结果无损合并。*与今天的关系*：展示 online softmax 不止用于前向一整行，还能把"**按共享前缀分组算出的局部结果**"精确拼起来——是第 4 节技巧在推理侧（含 day8 KV 复用思路）的活用。

---

## 三、🔁 旧知回顾（间隔重复日 · 第 15 讲）

> 换血：避开最近两次（第 9、12 讲）复习过的 Function Calling / RAG切分 / MoE路由塌方 / CoT / KV Cache / 对比学习，挑最久没被复习的核心——day5 优化器、day9 验证器、day2 注意力缩放。问题在前，答案折叠。

- **Q1（07-09 优化器）**：Adam 的**二阶矩 v̂** 起什么作用？为什么实践中几乎都用 **AdamW** 而不是原始 Adam？
- **Q2（07-13 验证器 / 奖励模型）**：什么是 **reward hacking**？为什么 **RLVR（可验证奖励）**天然更抗 hack？它赖以成立的 **generation-verification gap** 是什么，一旦这个 gap 消失会怎样？
- **Q3（07-06 注意力）**：自注意力打分 `QKᵀ/√d` 里为什么要**除以 √d**？不除会发生什么？

<details><summary>点开看答案</summary>

- **A1**：Adam 同时维护一阶矩 m̂（梯度的 EMA，给"方向"和动量）和**二阶矩 v̂（梯度平方的 EMA，估计每个参数梯度的量级/方差）**。更新量 ∝ m̂/√v̂：**v̂ 的作用是给每个参数做"自适应步长归一化"**——梯度一直很大的参数除以大 √v̂、步子收小，梯度很小的参数除以小 √v̂、步子放大，于是所有参数被拉到大致 ±1 的更新量级，对不同尺度的参数和稀疏梯度都稳。**为什么用 AdamW**：原始 Adam 把 L2 正则当成梯度的一部分，于是权重衰减也被 √v̂ 归一化了——梯度大的参数反而被"少衰减"，正则强度和参数尺度耦合、失真。**AdamW 把权重衰减从梯度里"解耦"出来**，直接在参数上乘 (1−λη)，让"自适应步长"和"权重衰减"各管各的，泛化更好、也更好调，遂成事实标准。

- **A2**：**reward hacking** 是优化器钻奖励函数的空子——拿到高 reward 却没真正完成任务。典型如 day9 的 PRM 沦为"流畅度探测器"：RL 把 reward 刷到 >0.9，正确率却 <4%，靠的是套话风格而非真推理（Goodhart 定律：一个**学出来、有洞**的奖励函数 × 一个**不知疲倦**的优化器 = 洞必被找到）。**RLVR（Reinforcement Learning with Verifiable Rewards）**用**规则化、可验证**的奖励替代学出来的模型：数学答案精确匹配、代码跑单测、Lean 证明校验——**二值、无参数、无洞可钻**，故天然抗 hack，是 o1/R1/Kimi 这代推理模型的基石。它成立的前提是 **generation-verification gap：验证一个答案对不对，比生成它容易得多**（做出证明难，检查证明易）。一旦某个领域这个 gap 消失（验证和生成一样难，如开放式写作的"好坏"没有廉价判据），RLVR 范式就失效——只能退回学出来的奖励模型，又暴露在 hacking 风险下。

- **A3**：Q、K 是 d 维向量，点积 `q·k = Σ qᵢkᵢ` 是 d 项之和；若各维近似独立、均值 0 方差 1，**点积的方差正比于 d**，即 d 越大、分数的数值波动越大（尺度 ≈ √d）。把这种大尺度分数喂进 softmax，会让它**落到饱和区**——最大项的 e 指数把其余项彻底压没，softmax 输出趋近 one-hot、几乎不可微，梯度消失、训练不稳。**除以 √d 把点积方差重新拉回 ~1**，让 softmax 待在梯度健康的区间。这正是今天 online softmax 里"减 max 防溢出"之外，attention 数值稳定的**另一道保险**。

</details>

---

## 四、🎯 留给明天的钩子
- **今天点到没展开的**：FA3 的 **warp-specialization 与异步流水重叠**（怎么让 matmul 和 softmax 在时间上交叠）；重计算与 **activation checkpointing** 的关系；tile 大小如何受 SRAM 容量约束（第 6 节那个 M）。
- **建议下次深入（C 轨接力）**：
  - **并行策略 DP / TP / PP / ZeRO** —— day8、day12 反复埋的钩子，把"带宽/显存受限"的世界观从**单卡 kernel** 推广到**多卡训练**；与今天的 FlashAttention 合起来就是"高效训练"完整地图。**（强烈推荐，闭合三条历史钩子）**
  - 或 **FlashAttention → FA3/FA4 深挖到 L3**：异步流水、FP8、Blackwell 非对称扩展下的 kernel co-design。
  - 或转 C 轨 **长上下文技术**：稀疏注意力（省"看多少历史"）× RoPE 外推，与今天的"省搬运"正交互补。

## 五、📚 延伸阅读
- Tri Dao & Tri Dao 团队原始论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*（arXiv:2205.14135, NeurIPS 2022）——三件套的源头，第 2、3 页的 IO 复杂度分析值得细读。
- *FlashAttention-4* 官方博客（Together AI / Lambda）——图解 Blackwell 上 softmax 为何成为新瓶颈。
- Zhuo's Blog《Flash Attention 1 & 2》——带公式推导的 online softmax 逐步讲解，可对照本讲第 4 节手算例子。

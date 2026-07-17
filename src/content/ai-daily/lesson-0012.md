---
id: lesson-0012
source:
  file: lessons/2026-07-16.md
  section: 1
  hash: sha256:898bd1ca0adf885bf8ea6ea112a01de9b86ce267b2955ae3814720363e9601cb
lesson: 12
date: 2026-07-16
track: C
depth: L3
titleZh: MoE 专家并行与分布式训练
titleEn: MoE Expert Parallelism and Distributed Training
summaryZh: 复访升级day6 MoE L2→L3(分布式侧)。第一性原理:MoE省算力不省显存(671B权重全须驻显存,激活37B只省FLOPs)→显存逼出EP=把N个专家切到多卡。EP在四刀(DP切数据/TP切矩阵/PP切层/EP切专家)里最特殊:token走哪个专家运行时才知→通信是最重的all-to-all,总与DP/TP/PP组3D并行。一层MoE六步:gate→⚡dispatch(all-to-all#1按目标专家分拣寄出)→expert compute→⚡combine(all-to-all#2寄回原位,因残差x+FFN要原位相加)→加权和。核心账单:all-to-all吃40%+训练时,14GB/层走NVLink 16ms vs IB 280ms(17×)→PCIe跑不动frontier MoE,拓扑是可行性门槛非优化项。本质=用带宽压力换单卡显存压力(与day8 KV Cache带宽受限同源:算力过剩带宽才是真瓶颈)。第二旋钮:all-to-all要定长buffer→capacity=CF×(tokens×topk/专家数),超额token被dropping走残差pass-through;CF小省算力但掉质量/CF大浪费/dropless要动态形状撞显存墙;须逐层监控drop率(路由塌方体温计),配day6无损失偏置法压均衡。通信三主线:拓扑感知层级化(NVLink vs RDMA分层)/计算通信overlap/路由感知放置(Occult)
summaryEn: Expert parallelism shards MoE experts across devices because sparse activation saves computation but leaves all weights resident somewhere. The lesson follows routing, dispatch and combine all-to-all exchanges, capacity factors, token dropping, topology constraints, and communication optimization, showing how distributed MoE trades local memory pressure for severe bandwidth demands.
slug: moe-expert-parallelism-distributed-training
tags:
  - mixture-of-experts
  - expert-parallelism
  - distributed-training
  - all-to-all
sourceStatus: unreviewed
sourceStatusHash: sha256:898bd1ca0adf885bf8ea6ea112a01de9b86ce267b2955ae3814720363e9601cb
metadataStatus: current
metadataSourceHash: sha256:898bd1ca0adf885bf8ea6ea112a01de9b86ce267b2955ae3814720363e9601cb
featured: false
---
# 📅 2026-07-16 · 讲12 · 轨道C · MoE 专家并行与分布式训练 · 深度 L3

> **复访升级**：day6 MoE 上次 **L2 → 今天 L3**。切入角度：day6 讲清了 MoE「是什么」（稀疏激活、路由、负载均衡三代）；今天回答一个更硬的问题——**当 671B 总参塞不进一张卡，专家到底怎么切到多卡上跑？账单花在哪里？** 这是把 MoE 从"架构图"落到"分布式系统"的一层。

> **TL;DR**：MoE 省的是**算力（FLOPs）不是显存**，所以 671B 权重必须切开——这就是 **Expert Parallelism (EP)**：把 N 个专家分到不同卡上。代价是每一层多出**两次 all-to-all**（把 token 寄到专家所在卡、再把结果寄回），它在大规模训练里能吃掉 40%+ 的运行时间，**互联拓扑（NVLink vs RDMA）直接决定 MoE 能不能跑**。为了让通信形状可预测，每个专家的 buffer 定长 → 超额 token 被 **dropping**（丢弃走残差），于是多出一个 **capacity factor** 旋钮在"省算力"与"掉质量"之间调。一句话：**EP 把显存压力换成了带宽压力。**

---

## 一、核心概念精讲（L3）

### 0. 先接住 day6 埋的钩子：MoE「省算力不省显存」

day6 反复强调过一句容易被忽略的话：**MoE 解耦的是"总参数（容量）"和"激活参数（算力）"，但不解耦显存。** 今天从这句话往下推，你会自然得到 EP 的必要性。

第一性原理走一遍：DeepSeek-V3 是 671B 总参、每 token 只激活 37B。

- **激活 37B** → 每个 token 的前向 **FLOPs** 约等于一个 37B 稠密模型（这是 MoE 便宜的地方）。
- **总参 671B** → 但这 671B 的**权重仍然要老老实实存在显存里**，因为下一个 token 可能路由到任何一个专家。671B × 2 字节（BF16）≈ **1.34 TB**，一张 H100 才 80GB。

结论：**你可以用少算力跑 MoE，但你没法用少显存存 MoE。** 于是唯一出路是把专家**切开摊到很多张卡上**——这就是 Expert Parallelism。

> ⚠️ **头号误区**：很多人以为"EP = 省显存"。恰恰相反。EP 是被显存**逼出来**的：总显存一点没少，只是从"一张卡放不下"变成"128 张卡分着放"，而且还**额外**引入了通信 buffer、激活显存和 all-to-all 开销。EP 是**分摊**显存，不是**节省**显存。

### 1. EP 在四种并行里的独特位置

分布式训练里切模型有四把刀，先摆清楚它们各切什么（中英对照）：

| 并行方式 | 切什么 | 每卡有什么 | 主要通信 |
|---|---|---|---|
| **DP** 数据并行 | 切数据（batch） | **整份**模型副本 | 梯度 all-reduce |
| **TP** 张量并行 | 切单个大矩阵 | 每个权重矩阵的一片 | 每层 all-reduce/all-gather |
| **PP** 流水线并行 | 按层切 | 连续的若干层 | 相邻 stage 点对点传激活 |
| **EP** 专家并行 | 切**专家** | N 个专家里的一部分 | **all-to-all**（dispatch + combine） |

EP 的独特之处：DP/TP/PP 里"数据流"是规整的、静态的——你在编译期就知道谁跟谁通信、传多大。**但 EP 里，token 走哪个专家是运行时由路由器决定的**，每个 token 的目的地都不一样，且随 batch 内容变化。这种"人人可能发给人人、且量不固定"的模式，恰恰就是通信里最重的那一种：**all-to-all**。现实中 EP 从不单飞，总是和 DP/TP/PP 组成 **3D/4D 并行**一起用（比如 attention 部分用 TP、FFN 部分用 EP）。

### 2. 完整走一遍：一层 MoE 在 4 张卡上怎么跑

设定：**4 张卡**，共 **8 个专家**，每卡放 2 个专家（E0,E1 在卡0；E2,E3 在卡1；……），**top-2** 路由。每张卡本地手里有一批 token（比如各 1000 个）。看一层 MoE FFN 的完整生命周期，**6 步**：

```
① gate routing   每张卡对自己的 token 跑路由器 softmax(x·Wg)，
                 算出每个 token 的 top-2 目标专家编号。
                 例：卡0 的 token#7 → 要去 E3(卡1) 和 E5(卡2)。

② dispatch       ⚡ ALL-TO-ALL #1 ⚡
 （分拣+寄出）    每张卡把本地 token 按"目标专家在哪张卡"分拣成 4 堆，
                 然后所有卡同时互发：卡i 的第 j 堆 → 卡j。
                 像邮局：先按收件城市分拣，再全国对发。

③ expert compute 现在每张卡收到的都是"该来我这的" token，
                 本地跑 E0..E7 对应的 FFN（SwiGLU），各算各的。

④ combine        ⚡ ALL-TO-ALL #2 ⚡
 （寄回原位）     把每个 token 的专家输出，按它"原来来自哪张卡"寄回去。
                 必须寄回！因为残差连接 x+FFN(x) 要在原位置相加，
                 下一层 attention 也要 token 回到它原本的序列位置。

⑤ combine weights 原卡上把 top-2 专家的输出按路由权重 g_i 加权求和。

⑥ 进入下一层
```

**关键观察**：一层 MoE = **两次 all-to-all**（dispatch 寄出 + combine 寄回）。一个几十层的 MoE 模型，每步前向就是几十对 all-to-all，反向传播再来一遍。**通信不是偶尔发生，是每层的主旋律。**

### 3. all-to-all 才是真正的账单（L3 的核心）

**为什么 all-to-all 这么贵？** 因为它是通信模式里最"全连接"的一种：N 张卡，每张都要跟其余 N−1 张都通信，且大规模下走的是慢速跨机网络。研究里它能吃掉 **40%+ 的训练运行时**。

**用一组真实数量级把痛感建立起来**（来自 2026 生产栈实测）：
> 32k batch tokens、d_model=7168、BF16、top-8 路由、16 个 EP rank
> → **每层每步约 14 GB** 的 all-to-all 流量。
> - 走 **NVLink 4（900 GB/s）**：≈ **16 ms**
> - 走 **InfiniBand NDR（400 Gb/s ≈ 50 GB/s）**：≈ **280 ms**

**17 倍的差距！** 这就是为什么结论是硬的：**PCIe-only 集群根本跑不动 >32 专家的 frontier MoE**，互联拓扑不是优化项、是可行性门槛。

**第一性原理小结**：
> EP 做了一笔交换——**用"网络带宽压力"换"单卡显存压力"**。
> 这和 day8 的 KV Cache 是同一个幽灵的两副面孔：day8 说 decode 是**内存带宽**受限（每步要读一遍 KV Cache），今天说分布式 MoE 是**网络带宽**受限（每层要 all-to-all 一遍）。**算力早就过剩，带宽（无论片上还是片间）才是现代大模型的真瓶颈。**

**2026 的三条优化主线**（对应今天的联网材料）：

1. **拓扑感知 / 层级化**：naive 做法用一个"全局 all-to-all"把所有 EP 卡一视同仁，但**同机 NVLink（快）和跨机 RDMA（慢）差一个数量级**，一刀切等于按最慢的走。修法是分层：先跨机粗聚合、本地过滤、再机内细分发（TeleChat3 的三步法、NVIDIA Hybrid-EP）。
2. **计算-通信重叠（overlap）**：把 all-to-all 藏在专家计算背后，理想上限是"通信被计算完全掩盖"。
3. **路由感知放置**：让经常被同一 token 共同激活的专家尽量待在同一张卡（Occult 的 intra-collaboration），减少跨卡流量。

### 4. capacity factor 与 token dropping（另一半工程权衡）

EP 还逼出第二个旋钮。问题根源：**all-to-all 和专家计算都喜欢"定长形状"**——每个专家的输入 buffer 大小要在编译期定死，这样显存可预测、CUDA kernel 好调度、CUDA Graph 能用。**但路由器不保证均匀**，某个热门专家这一 batch 可能分到远超平均的 token。

于是引入 **capacity（容量）**：

```
capacity = capacity_factor × (本批 token 数 × top_k / 专家数)
```

`capacity_factor`（容量系数）就是给每个专家 buffer 留的余量：
- **CF = 1.0**：只给"绝对平均份额"，一点不宽裕。
- **CF = 1.25**：留 25% 缓冲，容忍轻度不均衡。

**超出容量的 token 怎么办？→ token dropping（丢弃）**：多出来的 token **直接跳过这一层专家**，走残差连接 pass-through 出去。等于这个 token 在这一层"没被 MoE 处理过"。

**权衡三角（这就是 L3 的味道）**：

| 选择 | 省什么 | 代价 |
|---|---|---|
| **CF 调小**（如 1.0） | 省显存 & 算力，通信量小 | 丢 token 多 → **质量掉** |
| **CF 调大**（如 2.0） | 丢得少，质量稳 | buffer 撑大 → **浪费算力和显存** |
| **Dropless**（一个不丢） | 质量最好 | 要**动态形状**（grouped GEMM），实现复杂、易撞**显存墙**（2026 AMD 实测在纯 JAX 下不可行需专用 kernel） |

工程实践：**训练时必须逐层监控 drop 率**（每层丢了百分之几），它是负载不均衡的直接体温计——drop 率飙高往往意味着 day6 讲的"路由塌方"正在发生。现代大厂（DeepSeek 系）配合 day6 讲的**无辅助损失偏置法**把负载压均衡，从而可以用较小的 CF 甚至趋近 dropless，两个旋钮是配合使用的。

> ⚠️ **易错点**：
> - "token dropping 是 bug" ❌ ——它是**设计上的性能-质量阀门**，不是故障。
> - "CF 越大越安全" ❌ ——CF=∞ 等于放弃了 MoE 的效率优势。
> - "把 EP 数量开大就能扩展" ❌ ——EP 越大，all-to-all 的参与方越多、跨机比例越高，通信可能反成瓶颈；实践中常用"增大 expert-TP、减小 expert-EP"来在均衡与通信间折中。

---

## 二、最新动态 / 论文速览

> ✅ 已联网检索（2026-07-16）。⚠️ 说明：arXiv 编号取自检索结果、其 `YYMM` 与日期自洽（2607=2026-07、2606=2026-06），但我未逐一打开 PDF 逐字核验，标注 `（检索结果，未逐字核实）`。

1. **UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods** · arXiv 2607.06202 / SIGCOMM '26（2026-07，会议 8 月）（检索结果，未逐字核实）—— **为什么重要**：把 EP 通信库当作"生产级超节点"的一等公民重新架构，是本轨"all-to-all 是账单"命题的最新工业回应。**与今天的关系**：正是第 3 节"通信优化三主线"里"专用通信库"路线的最前沿。

2. **UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing** · arXiv 2606.04101（2026-06）（检索结果，未逐字核实）—— **为什么重要**：直击 EP 的负载不均衡痛点，宣称接近最优均衡。**与今天的关系**：把 day6 的"路由塌方/负载均衡"和今天的"capacity/dropping"合流到 rack 级系统设计上。

3. **NVIDIA Hybrid-EP（Optimizing Communication for MoE Training with Hybrid Expert Parallel）** · NVIDIA 技术博客（2026-02）—— **为什么重要**：给出拓扑感知 EP 的工程范式，区分 intra-node NVLink 与 inter-node RDMA、warp 组分离 dispatch/combine、原生 FP8/BF16 与通信-计算全重叠，逼近硬件带宽上限。**与今天的关系**：第 3 节"层级化 + overlap"两条主线的落地实现。

4. **Dropless MoE Training in JAX with Primus-Turbo** · AMD ROCm 博客（2026-06）—— **为什么重要**：具体展示"丢弃 vs 无丢弃"的实测取舍——定长丢弃换速度、纯 JAX 无丢弃撞显存墙需专用 grouped-GEMM kernel。**与今天的关系**：第 4 节 capacity/dropping 权衡三角的真实工程注脚。

来源汇总：[NVIDIA Hybrid-EP](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/) · [UBEP (arXiv 2607.06202)](https://arxiv.org/pdf/2607.06202) · [UltraEP (arXiv 2606.04101)](https://arxiv.org/pdf/2606.04101) · [AMD Dropless MoE](https://rocm.blogs.amd.com/software-tools-optimization/maxtext-dropless-moe/README.html) · [All-to-All in MoE Training (APXML)](https://apxml.com/courses/mixture-of-experts/chapter-4-scaling-moe-distributed-training/all-to-all-communication-moe) · [Handling Dropped Tokens (APXML)](https://apxml.com/courses/mixture-of-experts/chapter-3-moe-training-dynamics-optimization/handling-dropped-tokens) · [LMSYS 大规模 EP 部署 DeepSeek](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)

---

## 三、🔁 旧知回顾（间隔重复日 · 第 12 讲）

> 换血：避开最近两次（第 6、9 讲）复习过的注意力 / 测试时计算 / Adam / Function Calling / RAG切分 / MoE路由塌方，挑最久没被复习的核心——day1 提示工程、day8 KV Cache、day11 对比学习。问题在前，答案折叠。

- **Q1（07-06 提示工程）**：从第一性原理看，Chain-of-Thought（CoT）为什么能提升推理表现？为什么说它的本质是"用 token 换算力"？
- **Q2（07-12 KV Cache）**：为什么说自回归 **decode** 阶段是"**内存带宽**受限"？为什么把 KV Cache 缩小 k 倍，decode 大约就能快 k 倍？（这也是 GQA/MLA 的动机）
- **Q3（07-15 对比学习）**：InfoNCE 里的 **in-batch negatives（批内负样本）** 是什么？为什么它直接导致"embedding 模型偏爱超大 batch size"？

<details><summary>点开看答案</summary>

- **A1**：Transformer 每生成一个 token 的计算量是**固定**的（约正比于参数量），它没法对一道难题"多想一会儿"。CoT 让模型**把中间推理显式写成一串 token**，于是原本要在一次前向里"心算"完成的多步推理，被摊开成很多步、每一步都获得一次完整的前向计算——**更多的输出 token = 更多次前向 = 更多算力**投在这道题上。本质上 CoT 是在**用序列长度（token 预算）换取有效计算深度**，把"参数固定、单步算力固定"的模型变得能按题目难度追加算力。这也是它和 day3「测试时计算」序贯家族一脉相承的原因。

- **A2**：decode 是**一次生成一个 token**，每步只处理长度为 1 的新 query，但要拿它去和**过去所有 token 的 K/V**做注意力。这一步的**算术强度极低**——搬运的数据量（整个 KV Cache）远大于要做的浮点运算量，于是瓶颈不在算力（GPU 的 FLOPs 用不满），而在**把 KV Cache 从显存搬进计算单元的带宽**。所以每步延迟 ≈ 读一遍 KV Cache 的时间 ≈ 正比于 KV Cache 大小。把 KV Cache 压小 k 倍（MQA/GQA 共享 K/V 头、MLA 低秩压缩），要搬的数据就少 k 倍，decode 延迟大约就降到 1/k。**核心：decode 快慢由带宽决定，而带宽消耗由 KV Cache 体积决定。**

- **A3**：训练 embedding 用 InfoNCE，是"从 1 个正对 + 一堆负样本里做 (N+1) 选 1 的 softmax"。**in-batch negatives** 的技巧是：**不额外去采负样本，直接把同一个 batch 里"别人的正样本"当作我的负样本**——batch 里有 B 个 (query, doc) 正对，对某个 query 来说，其余 B−1 个 doc 全是免费的负样本。于是 **batch 越大 → 每个样本的负样本数越多 → InfoNCE 分母里对比的项越多 → 对比信号越强、学出的 embedding 越好**。负样本数量几乎直接进了损失函数的分母，这就是 embedding 训练"batch size 越大越香"、动辄上万甚至十万级 batch 的根本原因。（注意 day11 也提醒过：**堆数量 ≠ 有效**，几个精挑的 hard negative 常胜过几千个随机负样本。）

</details>

---

## 四、🎯 留给明天的钩子

- **今天点到没展开的**：
  - **3D/4D 并行**里 EP 到底怎么和 DP/TP/PP 组合排布（比如 attention 用 TP、FFN 用 EP 的混合切法）——今天只提了一句。
  - **DeepEP** 的两种 dispatch 模式（prefill 用 Normal 追吞吐、decode 用 Low-Latency 追延迟）、以及 FP8 grouped GEMM（DeepGEMM）× NVSHMEM 的推理栈。
  - **推理侧的大规模 EP**（LMSYS 在 96×H100 上做 PD 分离 + 大规模 EP 部署 DeepSeek），和训练侧的取舍不同。
- **建议下次深入（轨道 C）**：
  - 顺势开新题 **「高效训练：FlashAttention + 并行策略（DP/TP/PP/ZeRO）」**——今天已经把 EP 讲透，正好把另外四把"并行的刀"补齐，形成完整的分布式训练地图（多条旧钩子都指向这里）。
  - 或 MoE **L3→L4**：细粒度专家 + 共享专家的最新结构演化、专家专业化（specialization）的可解释性研究、EP 推理部署前沿。

---

## 五、📚 延伸阅读

- [NVIDIA · Optimizing Communication for MoE Training with Hybrid Expert Parallel](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)（拓扑感知 EP 的工程范式）
- [APXML · All-to-All Communication in MoE Training](https://apxml.com/courses/mixture-of-experts/chapter-4-scaling-moe-distributed-training/all-to-all-communication-moe)（dispatch/combine 六步流水线讲得很细）
- [LMSYS · Deploying DeepSeek with PD Disaggregation and Large-Scale EP on 96 H100](https://www.lmsys.org/blog/2025-05-05-large-scale-ep/)（推理侧大规模 EP 的真实系统设计）

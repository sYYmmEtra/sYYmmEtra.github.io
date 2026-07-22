---
id: lesson-0011
source:
  file: lessons/2026-07-15.md
  section: 1
  hash: sha256:f5abf138d03f4633fb4bf282e59e08921be0b8fcb9a277cfb44b1080279f3d5f
lesson: 11
date: 2026-07-15
track: B
depth: L2
titleZh: 嵌入与表示学习：对比学习（Contrastive Learning）
titleEn: "Embeddings and Representation Learning: Contrastive Learning"
summaryZh: 回填day7/10钩子(bi-encoder为何work)。Embedding=离散词句→R^d、几何距离≈语义相似度;核心难题=语义相似度无标签,监督信号从哪来?对比学习答:相似度是相对的,只需"A比C更像B"排序信号,正对可免费自造(dropout=SimCSE/翻译对/query↔doc)。主损失InfoNCE=(N+1)选1的softmax=抬正对压负对;sim=cosine,τ温度小则重罚hard neg。两工程点:①in-batch negatives(batch越大负样本越多→偏爱超大batch根因)②hard negatives(几个好的胜几千随机,易误伤false negative)。双塔无交互缺陷正是day10需rerank的原因(day7→10→11合龙);现代=decoder LLM对比微调成embedding(E5-Mistral)
summaryEn: Contrastive learning builds embedding spaces from relative similarity signals instead of hand-designed features. This lesson explains InfoNCE, cosine similarity, temperature, in-batch and hard negatives, false-negative risk, pooling, and instruction-aware encoders, connecting efficient dual-tower retrieval to the interaction limitations that motivate later cross-encoder reranking.
slug: embeddings-contrastive-learning
tags:
  - embeddings
  - contrastive-learning
  - infonce
  - representation-learning
sourceStatus: unreviewed
sourceStatusHash: sha256:f5abf138d03f4633fb4bf282e59e08921be0b8fcb9a277cfb44b1080279f3d5f
metadataStatus: current
metadataSourceHash: sha256:f5abf138d03f4633fb4bf282e59e08921be0b8fcb9a277cfb44b1080279f3d5f
featured: false
---
# 📅 2026-07-15 · 讲11 · 轨道B · 嵌入与表示学习：对比学习（Contrastive Learning） · 深度 L2

> **TL;DR**：Embedding 是把离散的词/句"翻译"成稠密向量、让**几何距离 = 语义相似度**的地图；而对比学习是画这张地图的**主流训练法**——它不教模型"这句话是什么类别"，只教"哪两句该靠近、哪两句该推远"。核心一招 InfoNCE：把 1 个正样本从一堆负样本里"认出来"，等价于最大化正对相似度、压低负对相似度。这就是 day7/day10 里 bi-encoder 双塔"为什么 work"的地基。

---

## 一、核心概念精讲（L1 快过 → L2 主体）

### L1 · 直觉：地图与坐标

先建立一个类比。你要给全世界的**词/句子**发"经纬度坐标"，要求是：**含义相近的东西，坐标也相近**。这张坐标表就是 **embedding（嵌入）**——一个把离散符号映射到连续向量空间 $\mathbb{R}^d$（常见 $d=768/1024/4096$）的函数。

- "国王" → `[0.21, -0.8, ..., 0.05]`
- "女王" → `[0.19, -0.7, ..., 0.11]`（离"国王"很近）
- "香蕉" → `[-0.6, 0.3, ..., 0.9]`（离得很远）

**表示学习（Representation Learning）**：与其人工设计特征（如 TF-IDF 数词频，day10 讲过的 BM25 就是这一派），不如让神经网络**自己学出**这套坐标。学得好，下游任务（检索、聚类、分类）都只需在这张地图上量距离，简单得多。这就是"representation"一词的含义——把原始输入**重新表示**成一组好用的坐标。

**关键问题来了**：坐标要"语义近则靠近"，可"语义相似度"本身没有标签——没人给你标"这两句话相似度 0.83"。**监督信号从哪来？** 这正是对比学习登场的地方。

### L2 · 机制：对比学习怎么造监督信号

**第一性原理**：相似度是**相对**的。你不需要绝对分数，只需要"**A 和 B 比 A 和 C 更像**"这种排序信号。而这种信号可以**几乎免费地自动构造**：

- **正样本对（positive pair）**：本就该相近的一对。来源可以是——同一句话的两次不同 dropout/裁剪（自监督，SimCSE 的招）、翻译对、问题↔答案、query↔相关文档。
- **负样本（negatives）**：随便抓的、大概率不相关的其它样本。

训练目标一句话：**把正对拉到一起，把负对推开**（pull together / push apart）。

```
        正对要近                负对要远
   anchor ●───● positive     anchor ●        ● negative
          (拉近)                       ＼＿＿＿＿／
                                        (推远)
```

#### InfoNCE：把"推拉"写成一个损失函数

主力损失叫 **InfoNCE**（Noise-Contrastive Estimation 的一种）。把它理解成一道**多选题**：给定 anchor $q$，在"1 个正样本 $k^+$ + N 个负样本 $k^-_i$"里，让模型**认出哪个是正的**——本质是一个 (N+1) 选 1 的 softmax 分类。

$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(q, k^+)/\tau)}{\exp(\text{sim}(q, k^+)/\tau) + \sum_{i=1}^{N}\exp(\text{sim}(q, k^-_i)/\tau)}
$$

逐项拆开——**别只念公式，看每个符号在干嘛**：

- $\text{sim}(a,b)$：相似度，通常是 **cosine 相似度**（先把向量归一化到单位球面，再点积）。呼应 day7：语义近 = 向量夹角小 = cosine 大。
- 分子 = 正对的相似度（指数化）。分母 = 正对 + 所有负对之和。**这就是个 softmax**：分子占分母的比例越大，loss 越小。
- 要让 loss 变小，模型被逼着：**抬高分子（正对更像）＋压低分母里的负项（负对更不像）**——推拉两件事一个式子搞定。
- $\tau$（temperature，温度）：除在指数里的缩放。**易错点**：$\tau$ 小 → 分布更尖锐 → 对"最难那个负样本"惩罚极重（更 aggressive）；$\tau$ 大 → 更平滑。典型 0.05–0.07，是**必调超参**，调不好训练直接崩或学不动。

#### 两个工程关键：in-batch negatives 与 hard negatives

**(1) In-batch negatives（批内负样本）——几乎白嫖的负样本**
不用专门去采负样本。一个 batch 有 B 个正对 $(q_i, k^+_i)$，对 $q_i$ 而言，**同批里其他所有 $k^+_j (j\neq i)$ 天然就是负样本**。于是 batch 越大，每个 anchor 见到的负样本越多，对比信号越强。**这就是"embedding 模型偏爱超大 batch（几千甚至上万）"的根本原因**——不是玄学，是负样本数量直接进了 InfoNCE 分母。

**(2) Hard negatives（难负样本）——质量决定天花板**
随机负样本大多"太好认"（香蕉 vs 国王），模型学不到细粒度差别。**难负样本**是"看起来很像但其实不相关"的负样本（如"苹果公司" vs "苹果水果"）。它们把决策边界逼到更精细的地方。**常见误区**：以为堆数量就行；实际上**几个高质量 hard negative 比几千个随机负样本更能提分**——但挖 hard negative 又容易误伤（把真正的正样本当成负样本，即"假负样本 false negative"，会毒化训练）。这是该领域最吃工程手艺的一环。

#### 双塔（bi-encoder）：回填 day7/day10 的地基

现在能闭环了。day7/day10 里的 **bi-encoder 双塔**——query 塔和 doc 塔**各自独立**把文本编码成一个向量，然后算 cosine——**它凭什么让"语义相近的向量靠近"？** 答案就是：**它是用对比学习（InfoNCE）训出来的**。训练时 $(query, 相关doc)$ 是正对、批内其它 doc 是负对，模型被 InfoNCE 逼着把相关的 query/doc 编到向量空间的邻近位置。

也顺带解释了 day10 说的双塔"**无交互**"缺陷：因为两塔独立编码、相关性只在最后一步用一个点积近似，所以它省算力（doc 可预先编码入库）但表达力受限——于是才需要 day10 的 cross-encoder rerank 来补。**day7 → day10 → 今天，三课在这里合龙。**

#### 现代做法：拿解码器 LLM 改造成 embedding 模型

2023 年后的主流不再从零训小编码器，而是**拿一个预训练好的 decoder-only LLM（Mistral/Qwen/LLaMA 等），用对比学习微调成 embedding 模型**（代表作 E5-Mistral、后续 Qwen3-Embedding、Gemini/NV-Embed 一系）。要点两个：
- **怎么从"逐 token 输出"的 LLM 拿到一个句向量？** 常见 mean-pooling（平均所有 token 隐向量）或取最后一个 token（如 `[EOS]`）的隐向量。
- **instruction-aware**：输入前拼一句任务指令（"Represent this sentence for retrieval:"），**同一段文本按不同任务能编出不同向量**——这是近两年 embedding 模型的重要能力升级。

### ✅ 完整走一遍（一个 batch 的前向）

设 batch=3，三个正对，`sim`=cosine，$\tau=0.05$。只追 anchor $q_1$="猫喜欢睡觉"：

1. **编码**：双塔把 6 段文本各编成单位向量。$q_1$ 的正样本 $k^+_1$="猫咪整天在打盹"，负样本是同批的 $k^+_2$="欧元汇率上涨"、$k^+_3$="如何安装显卡"。
2. **算相似度**（假想值）：$\text{sim}(q_1,k^+_1)=0.82$，$\text{sim}(q_1,k^+_2)=0.10$，$\text{sim}(q_1,k^+_3)=0.15$。
3. **除温度**：0.82/0.05=16.4；0.10/0.05=2.0；0.15/0.05=3.0。（注意温度把差距**放大**了 20 倍——这就是 $\tau$ 的威力）
4. **softmax**：$e^{16.4}=1.3\text{e}7$；$e^{2.0}=7.4$；$e^{3.0}=20$。正类概率 = $1.3\text{e}7/(1.3\text{e}7+7.4+20)\approx 0.9999$。
5. **loss** $= -\log(0.9999)\approx 0.0001$——极小，说明这个 anchor 已经学得很好，梯度几乎不推它。
6. 若换成 hard negative "狗喜欢散步" sim=0.75：0.75/0.05=15，$e^{15}=3.3\text{e}6$，正类概率降到 ≈0.80，loss≈0.22——**梯度显著增大，模型被迫去区分猫/狗这种细微差别**。这一步直观展示了 hard negative 为什么值钱。

---

## 二、最新动态 / 论文速览

`⚠️ 已联网但未逐篇精读`：WebSearch 已成功返回，以下 arXiv 条目**编号与日期自洽、可自行核验**，但我**仅据检索摘要转述其主张、未点开读全文**——引用前请核对原文。营销博客与无法核实身份的匿名投稿（如 OpenReview 上的 ReCo）已剔除不列。

1. **"Cropping outperforms dropout as an augmentation strategy for self-supervised training of text embeddings"**（arXiv 2508.03453，2025-08）—— 为什么重要：直接挑战 L2 里讲的 **SimCSE"同句两次 dropout 造正对"**，实测**裁剪(cropping)增强比 dropout 更好**。正对怎么造是对比学习的命门，这条给了"SimCSE 那招未必最优"的新证据。与今天的关系：补在"正样本对来源"那一段。
2. **"Training Sparse Mixture of Experts Text Embedding Models"**（arXiv 2502.07972，2025-02）—— 为什么重要：把 **MoE（day6 学过）架构塞进 embedding 模型**、仍用 InfoNCE 对比目标训，用稀疏激活省 embedding 推理成本。与今天的关系：day6(MoE) × day11(对比学习) 两条线在这里交汇。
3. **"GRACE: Generative Representation Learning via Contrastive Policy Optimization"**（arXiv 2510.04506，2025-10）—— 为什么重要：把对比学习和**策略优化(RL)**缝起来，代表"直接拿 LLM 当 embedding 模型"的调优派新方向。与今天的关系：呼应 L2 末尾"拿 decoder LLM 改造成 embedding"，并串到 day9 的 RL/验证器线。

> 一条趋势印证（多方资料，非单篇）：**Matryoshka 表示学习**（一个向量可从 2048 截到 256 维做速度/精度权衡）已是 2026 embedding 模型的标配——正是「明天的钩子」里点的支线，现有外部佐证。
>
> 稳固地基（长期标准引用，非近期动态）：SimCSE（arXiv 2104.08821）造正对的起点、E5-Mistral（arXiv 2401.00368）开"合成数据+对比微调 decoder LLM"范式、MTEB（arXiv 2210.07316）事实标准评测榜——想看当前榜首去 HuggingFace MTEB leaderboard。

---

## 三、🎯 留给明天的钩子

- **今天点到没展开的**：
  - **对比学习的表示塌缩（representation collapse）与各向异性（anisotropy）**——为什么没有负样本、光拉正对会让所有向量挤成一团？非对比方法（BYOL/Barlow Twins/VICReg）怎么不用负样本也不塌缩？
  - **Matryoshka 表示学习**（一个向量前 256 维就能当低维版用）——day10 提过向量库成本，这直接关系到检索省显存。
  - **cross-encoder（day10 的 rerank）与 bi-encoder 的训练差异**：为什么 cross-encoder 不用 InfoNCE 也能训。
- **建议下次深入**（下一次 B 轨，约讲14）：把本主题升到 **L3**——对比学习的**工程与失败模式**：大 batch 的显存/梯度技巧、hard negative 挖掘 pipeline、false negative 治理、温度与 batch size 的相互作用；或按台账另一条线转 **B 轨「信息论基础：熵/交叉熵/KL」**，正好回填 InfoNCE 里 softmax 与交叉熵的信息论来历。

## 四、📚 延伸阅读（可选）

- Lilian Weng, *"Contrastive Representation Learning"*（lilianweng.github.io 博客）—— 把 InfoNCE/SimCLR/MoCo 等一网打尽的高质量综述（视觉为主但原理通用）。
- SentenceTransformers 官方文档的 *Training Overview* —— 想动手：`MultipleNegativesRankingLoss` 就是 in-batch-negative 版 InfoNCE，几十行能跑通一个双塔。

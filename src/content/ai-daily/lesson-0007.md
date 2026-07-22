---
id: lesson-0007
source:
  file: lessons/2026-07-11.md
  section: 1
  hash: sha256:8a1fa6e1bc5fafd113107498acefca2ca053589940497d526f8327f29d8223b8
lesson: 7
date: 2026-07-11
track: A
depth: L2
titleZh: RAG 基础（切分 / embedding / 向量库 / 召回）
titleEn: "RAG Foundations: Chunking, Embeddings, Vector Stores, and Retrieval"
summaryZh: RAG=开卷考,知识从参数记忆搬到非参数记忆(可增删/引用/实时更新);两段式=离线(切分→embedding→入库)+在线(query编码→ANN检索TopK→拼prompt→生成);成败70%在检索,第一张骨牌是切分(大块稀释信号,必挂metadata);embedding=bi-encoder双塔+对比学习,语义近则夹角小(cosine),query与doc须同模型编码;向量库存向量+原文+metadata,千万级用ANN(HNSW,O(N)→O(logN));反直觉=语义≠关键词(需hybrid+BM25)、Lost in the Middle需rerank
summaryEn: Retrieval-augmented generation gives a model an editable external memory through offline indexing and online retrieval. This lesson traces chunking, embeddings, vector storage, approximate nearest-neighbor search, prompt assembly, and generation, while showing why metadata, hybrid retrieval, reranking, and careful top-k selection often matter more than the generator itself.
slug: rag-foundations-retrieval-pipeline
tags:
  - rag
  - embeddings
  - vector-search
  - retrieval
sourceStatus: unreviewed
sourceStatusHash: sha256:8a1fa6e1bc5fafd113107498acefca2ca053589940497d526f8327f29d8223b8
metadataStatus: current
metadataSourceHash: sha256:8a1fa6e1bc5fafd113107498acefca2ca053589940497d526f8327f29d8223b8
featured: false
---
# 📅 2026-07-11 · 轨道A · RAG 基础（切分 / embedding / 向量库 / 召回）· 深度 L2

> **TL;DR**：07-08 我们学的 Tool Use，是把模型接进"确定性的软件世界"；今天的 **RAG** 是它的姊妹篇——把模型接进"外部知识"。做法是给模型开一场**开卷考试**：把知识切成块、编码成向量、存进可近似检索的向量库；提问时先按语义捞回最相关的几块塞进 prompt，让模型"照着资料答"。它把知识从**参数记忆**（贵、易过时、会幻觉、不可溯源）搬到**非参数记忆**（可增删、可引用、可实时更新）。今天的核心洞见：**RAG 的成败大头不在生成，而在检索；而检索的第一张多米诺骨牌是「切分」——它一旦切坏，后面全链路静默劣化。**

---

## 一、核心概念精讲（L2）

### 0. 先把它放回地图上：微调 / Tool Use / RAG 三条"外挂知识"的路

模型的知识默认**焊死在权重里**（parametric knowledge，参数化知识），这带来四个硬伤：**会过时**（训练截止后的事不知道）、**会幻觉**（不知道也会流畅地编，因为它建模的是"像不像人话"而非"真不真"）、**不可溯源**（给不出"出自哪份文档第几页"）、**私有知识进不去**（公司内网 wiki 不在训练集里）。

给模型补充新知识，工程上有三条路，正好对应我们的学习脉络：
- **微调 / LoRA**（我们最熟的老本行）：把知识**再压进权重**。代价高、更新慢、仍不可溯源，适合教"新技能/新风格"而非"新事实"。
- **Tool Use**（07-08）：让模型**实时调函数**去外部世界取数（查数据库、调 API）。
- **RAG（Retrieval-Augmented Generation，检索增强生成）**：今天的主角。知识不塞脑子，塞进"书架"；考试时（推理时）现翻书，把相关段落抄到答题纸（prompt）上再作答。

> **最小心智模型**：闭卷考 → 开卷考。模型的推理能力不变，变的是它手边有没有"正确的参考资料"。所以 RAG 的工程重心，全压在一句话上——**"怎么在浩如烟海的资料里，快速翻到那关键的两三页"**。

### 1. L2 机制：完整管线拆两段

RAG 系统在时间上分成**离线建索引**和**在线查询**两个阶段。先看全景，再逐环节讲透：

```
【离线 · Indexing，做一次】
 原始文档 ──①切分──▶ chunks ──②embedding──▶ 向量 ──③入库──▶ 向量数据库
                                                          (向量 + 原文 + 元数据)

【在线 · Retrieval + Generation，每次提问】
 用户 query ──②同一个 embedding 模型──▶ query 向量
        │
        ▼
   ④向量库里做 ANN 近似最近邻搜索 ──▶ Top-K 相似 chunks
        │
        ▼
   ⑤拼进 prompt 模板：「根据以下资料回答：{chunks}\n\n问题：{query}」
        │
        ▼
   ⑥LLM 生成答案（可带 [来源] 引用）
```

**注意两个"同一"**：离线和在线用的必须是**同一个 embedding 模型**（否则 query 和 doc 落在不同语义空间，没法比）；④检索命中的是**向量**，但⑤拼进 prompt 的是**原文**，引用靠的是**元数据**——向量只是"用来快速定位的索引指纹"。

---

#### 环节①：切分 Chunking —— 最被低估、最容易翻车的一步

**为什么必须切？** 三个约束：(a) embedding 模型有输入长度上限（常见 512 token）；(b) 检索的"粒度"就是 chunk——你召回的是整块，块太大则夹带大量无关内容稀释信号，块太小则语义不完整、答案被拦腰截断；(c) LLM 上下文窗口和成本有限，不能把整库塞进去。

**主流三档策略（从笨到巧）：**

- **固定长度切分（fixed-size）**：每 N 个 token 一刀，配 **overlap 重叠**（如每块 512、相邻重叠 50）。重叠是为了缝合被切断的句子——否则一个跨边界的关键句会被劈成两半，两块都语义残缺。**最简单，也是最强的 baseline**，但会硬生生切断段落。
- **结构 / 递归切分（recursive）**：按文档天然分隔符的优先级递归切——先按 `\n\n`（段落），装不下再按 `\n`（行），再按 `。`（句），最后才按字符。LangChain 的 `RecursiveCharacterTextSplitter` 就是这个。**性价比最高的默认选择**，尊重了文档结构。
- **语义切分（semantic）**：不看字符数，看**意思**。逐句 embedding，当相邻句子的向量相似度**骤降**（出现"话题拐点"）时就下刀。好处是每块语义自洽；代价是要先跑一遍 embedding、更慢，阈值难调——今天最新动态里有论文专门质疑"这份额外开销值不值"。

> **常见误区①（大块 ≠ 信息全）**：以为 chunk 越大越保险。恰恰相反——**大块会稀释相似度信号**。检索靠"query 向量和 chunk 向量的相似度"，一个 2000 字的块里只有 1 句相关，它的平均语义会被另外 1999 字带偏，反而排不进 Top-K。这是"检索在切分处静默失败"的头号原因。
> **常见误区②（忘挂元数据）**：没有 metadata（来源文档、标题、页码、父块 ID）就没法做引用、没法按来源过滤、没法层级检索。工业界铁律：**每个 chunk 必带 metadata**。

---

#### 环节②：Embedding —— 把文本压成"语义坐标"

**是什么**：embedding 模型（一个 **bi-encoder 双塔编码器**）把一段文本映射成一个定长稠密向量（如 768 或 1536 维）。训练目标（对比学习 contrastive learning）让它满足：**语义相近的文本，向量在空间里也靠近**——把"同义句"拉近、"无关句"推远。

**为什么能"按意思"检索**：文本一旦变成向量，"语义相似"就退化成一个纯几何问题：**两个向量夹角有多小**。用 **余弦相似度（cosine similarity）** 度量：

$$\text{sim}(a,b)=\cos\theta=\frac{a\cdot b}{\|a\|\,\|b\|}$$

值域 [−1, 1]，越接近 1 越相似。它只看**方向**不看长度——正合我们要的，因为文本长短不该影响"讲的是不是一回事"。（实践中向量常先归一化到单位长度，此时余弦相似度就等于点积，算得更快。）

> **常见误区③（双塔的"不对称"陷阱）**：query 和 document 形态天差地别——query 是"苹果第一款手机哪年出的？"，答案 chunk 是"iPhone 于 2007 年发布……"，两者字面几乎不重叠。好的 embedding 模型是**专门训练**来跨越这道鸿沟的（有些还要求 query 加前缀 `query:`、doc 加 `passage:`，**用错前缀检索质量会明显掉**）。这也解释了为什么 **query 和 chunk 必须用同一个模型**编码——要落在同一个语义空间里才能比较。
> **常见误区④（语义 ≠ 关键词）**：dense 向量擅长"意思相近"，却可能**漏掉精确匹配**——产品型号 `RTX-4090`、错误码 `E1234`、人名这类"必须一字不差"的词，语义检索常翻车。所以工业界几乎都上**混合检索（hybrid）= 稠密向量 + 稀疏关键词（BM25）**。这是明天"进阶 RAG"的入口，今天先埋个钩子。

---

#### 环节③④：向量库与 ANN —— 为什么不能"逐个比对"

假设库里有 1000 万个 chunk 向量。来一个 query，最朴素的做法是拿 query 向量和**每一个**算相似度取 Top-K——这叫**精确最近邻（exact kNN）**，复杂度 O(N·d)，千万级每次查询几百毫秒到数秒，**上不了生产**。

于是用 **ANN（Approximate Nearest Neighbor，近似最近邻）**：牺牲一点点召回精度，换回数量级的提速。最主流的是 **HNSW（Hierarchical Navigable Small World，分层可导航小世界图）**：把向量组织成一张**多层图**，上层稀疏（长跳、快速逼近大方向），下层稠密（短跳、精细定位）。查询时像"先坐高铁到城市、再打车到街区、最后步行到门牌"，把搜索复杂度从 O(N) 压到近似 **O(log N)**。Faiss、Milvus、Qdrant、pgvector 等向量库底层多是它。

> **一句话记住**：向量库存的**不只是向量**，还有**原文 + 元数据**。这是上面"两个同一"里第二条的落点。

---

#### 环节⑤⑥：召回与生成 —— 把"证据"喂给模型

检索返回 Top-K（K 常取 3~10）个 chunk 后，按模板拼成 prompt：

```
你是问答助手。只依据下面提供的资料回答，若资料中没有答案就说"未找到"。
【资料】
[1] {chunk_1 原文}（来源：doc_A p.3）
[2] {chunk_2 原文}（来源：doc_B p.7）
...
【问题】{用户 query}
```

模型据此生成，并可标注 `[1][2]` 引用。

> **常见误区⑤（Lost in the Middle 中间迷失）**：塞进去的 chunk **不是越多越好**。研究反复发现，模型对上下文**首尾**利用最好，**正中间**的容易被忽略。所以 K 不宜过大，且值得把"最相关的"排在开头/结尾——这引出下一课的 **rerank 重排**（今天最新动态第 4 篇正是干这个）。

---

### 2. 🎬 完整走一遍：一个问题跑通全管线

**场景**：公司知识库里有一份《员工手册》。用户问："**入职多久之后可以开始休年假？**"

1. **【离线，早已完成】** 手册被递归切分成 200 块，其中 chunk #87 内容是：
   > "试用期为 3 个月。员工在**转正后**方可申请带薪年假，年假额度按司龄计算……"

   它被 `bge-large` 编码成一个 1024 维向量 `v87`，连同原文和元数据 `{doc:手册, section:假期, page:12}` 存入 Qdrant。

2. **【query 编码】** 用户问题用**同一个** `bge-large`（加 query 前缀）编码成 `q`。注意 query 里根本没有"转正""试用期"这些词。

3. **【ANN 检索】** Qdrant 用 HNSW 在 20 万向量里毫秒级找出与 `q` 余弦相似度最高的 5 块。尽管字面不匹配，`bge` 知道"入职多久能休假"与"试用期 / 转正后可申请年假"**语义强相关**，于是 chunk #87 以 cos=0.82 排到第 1（误区④场景下若问的是错误码这类精确串，这一步就可能翻车）。

4. **【拼装 + 生成】** Top-5 原文塞进模板。模型读到 #87，生成：
   > "根据《员工手册》，试用期为 3 个月，**转正后**即可申请带薪年假，即入职满 3 个月后。[来源：手册 p.12]"

**机制在这个例子里做对了两件闭卷模型做不到的事**：(a) 跨越了 query 与 doc 的字面鸿沟（②的语义匹配）；(b) 给出了可核查的页码引用（③的元数据）。而如果切分时把这句话从"试用期 3 个月"和"转正后可申请年假"中间切断（误区①），检索就会残缺——**这就是为什么说切分是第一张骨牌**。

---

## 二、最新动态 / 论文速览

> ✅ 本次 WebSearch 联网成功；下列 4 条的标题 / arXiv 编号 / 日期均已用 WebSearch + WebFetch 二次核实（CAR 一篇连作者与摘要都已核对）。主线：**2026 年，切分/检索正从"离线拍脑袋定死"走向"随 query 自适应"，同时出现一股"这些花招值不值"的冷静评估潮。**

1. **Query-Adaptive Semantic Chunking (QASC)** · arXiv:2605.22834 · 2026-05
   - 为什么重要：切分长期是"离线定死、与 query 无关"的一步。QASC 首次把 query **引入切分阶段**——按 query-句子余弦选"种子句"，再向两侧做上下文窗口扩展动态构块；报告 F1≈0.85，较固定切分相对提升 18–27%。
   - 与今日主题的关系：直接挑战本课环节①"切分是纯离线步骤"的默认假设，是"切分决定成败"这条主线的最新延伸。

2. **Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG** · arXiv:2504.19754 · 2025-04
   - 为什么重要：严格对比两大"保上下文"技术——**late chunking**（先整篇长上下文编码、再切，块向量携带全局语境）与 Anthropic 的 **contextual retrieval**（给每块补一句 LLM 生成的上下文摘要再编码）。结论：contextual retrieval 语义更完整但更贵；late chunking 更省但会牺牲相关性。
   - 与今日主题的关系：正是对环节①"固定切分会切断语境"这一痛点的两种前沿解法，是从 L2 迈向 L3 的直接读物。

3. **Chunking Methods on RAG — Effectiveness Evaluation Against Computational Cost** · arXiv:2606.00881 · 2026-05-30（近 30 天，Wrocław 理工）
   - 为什么重要：一篇"泼冷水"评估。系统横评各类切分法后指出：**很多语义/花式切分带来的提升，撑不起它们高昂的计算成本**；并主张"chunking 被当成 trivial 预处理，其实埋着一堆被忽视的坑"。
   - 与今日主题的关系：给本课"语义切分更好？"补上工程理性——**别为边际收益上重方案**，呼应误区①之外的成本视角。

4. **CAR: Query-Guided Confidence-Aware Reranking for RAG** · arXiv:2605.04495 · 2026-05
   - 为什么重要：传统 rerank 只优化"query-文档相关性"，但相关 ≠ 对生成有用。CAR（training-free、即插即用）改用**生成器置信度变化**当"文档有用性"信号：能显著提升生成器确定性的文档才上位，且排序增益与下游生成 F1 强相关。
   - 与今日主题的关系：正对本课误区⑤的"Lost in the Middle"与⑥留的钩子，是明天"进阶 RAG（rerank）"的主角。

> 溯源补充：late chunking 原始论文见 arXiv:2409.04701（Jina AI）；"中间迷失"实证见下方延伸阅读。

---

## 三、🎯 留给明天的钩子

- **今天点到但没展开的**：混合检索（dense + BM25 稀疏）、rerank 重排（cross-encoder vs bi-encoder，以及 CAR 的置信度信号）、Lost in the Middle 的缓解、contextual retrieval / late chunking 的具体实现、query-adaptive 切分（QASC）。
- **建议下次深入**：
  - 轨道 A 顺势推进 **进阶 RAG（L2，可与今日无缝衔接）**——混合检索、rerank、查询改写（query rewriting / HyDE）、Contextual Retrieval、GraphRAG；
  - 或转 B 轨补 **嵌入与表示学习（对比学习如何训出 embedding 空间）**，正好回填今天环节②里"为什么语义相近向量就相近"的原理欠账。

---

## 四、📚 延伸阅读

- Anthropic，**《Introducing Contextual Retrieval》**（官方技术博客）——工业界"给每块补上下文"最实用的一招，含 BM25 混合与 rerank 的组合拳。
- **《Lost in the Middle: How Language Models Use Long Contexts》**（Liu et al., 2023, arXiv:2307.03172）——理解误区⑤"中间迷失"的原始实证。
- LangChain 文档 · `RecursiveCharacterTextSplitter` 与向量库集成——用 30 行代码把今天的管线亲手跑通一遍（今天的"可动手验证点"）。

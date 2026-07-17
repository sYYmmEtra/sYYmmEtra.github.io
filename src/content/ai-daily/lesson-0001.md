---
id: lesson-0001
source:
  file: lessons/2026-07-06.md
  section: 1
  hash: sha256:e74046d660a7fedec3671cd8ae668da7fbd7f5a0ee92f5a281a0ca02610ff9bb
lesson: 1
date: 2026-07-06
track: A
depth: L1
titleZh: 提示工程基础与常用模式
titleEn: Prompt Engineering Foundations and Common Patterns
summaryZh: prompt 是在条件化自回归分布；四招本质=定位任务(few-shot)/用token换算力(CoT)/结构拆解(分解)/推理集成(自一致性)
summaryEn: "Prompting conditions an autoregressive model rather than programming a deterministic system. The lesson connects few-shot examples, chain-of-thought, task decomposition, and self-consistency to four practical goals: locating the task, spending tokens on reasoning, breaking structure apart, and aggregating multiple reasoning paths."
slug: prompt-engineering-foundations
tags:
  - prompt-engineering
  - chain-of-thought
sourceStatus: unreviewed
sourceStatusHash: sha256:e74046d660a7fedec3671cd8ae668da7fbd7f5a0ee92f5a281a0ca02610ff9bb
metadataStatus: current
metadataSourceHash: sha256:e74046d660a7fedec3671cd8ae668da7fbd7f5a0ee92f5a281a0ca02610ff9bb
featured: false
---
# 📅 2026-07-06 · 轨道A · 提示工程基础与常用模式 · 深度 L1

> **TL;DR**：提示工程的本质是**用输入前缀去"条件化"一个自回归概率分布**，而不是给模型"下命令"。few-shot、CoT、分解、自一致性这四招之所以有效，底层其实是同一件事的不同侧面——**要么帮模型定位任务，要么给模型更多"串行计算"的草稿纸**。理解了这层，你就不会把提示工程当成玄学咒语。

---

## 一、核心概念精讲（L1）

### 0. 先建立唯一重要的心智模型：LLM 是个条件概率机

一切提示技巧都站在这一个第一性原理上。自回归 LLM 做的事只有一件：

$$p(x_1, x_2, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

- **x_{<t}**（context，上下文）：当前已有的所有 token。
- 模型在每一步吐出的是"下一个 token 的概率分布"，然后采样。

**提示词（prompt）不是指令，而是这个乘积里的"条件"。** 你写 prompt，本质是在挑选一个前缀，把整个后续分布 `p(答案 | 前缀)` 往你想要的方向掰。

> 🔑 **直觉类比**：LLM 像一条湍急的河流（预训练学到的分布），你的 prompt 是在上游放下的几块石头——你不能凭空造水，但能改变水流最终淌向哪个出口。所谓"提示工程"就是**摆石头的艺术**，而不是对河流喊话。

这个视角立刻解释了很多"为什么"：为什么示例的**格式**比正确性还重要、为什么让模型"先想再答"能变聪明——都是在改变条件分布。

---

### 1. Few-shot / 上下文学习（In-Context Learning, ICL）

**是什么**：在 prompt 里塞几对"输入→输出"示例（few-shot demonstrations），模型不更新任何权重，就能模仿这个任务。零示例叫 zero-shot，给 k 对叫 k-shot。

```
情感分类：
评论：这餐厅太难吃了     → 负面
评论：服务很热情           → 正面
评论：等位两小时气死我了   →        ← 模型在这里续写"负面"
```

**为什么有效（机制）**：注意这里**没有梯度下降、没有微调**。模型是在推理时（inference time）从上下文里"临时读懂"任务。主流有两种解释，学界仍在争论：

1. **任务定位说**：预训练时模型已经见过海量分类任务，示例的作用是**在庞大的能力空间里"定位/激活"出你要的那一个技能**，并锁定输出格式。
2. **隐式元学习说**：Transformer 的前向传播在数学上可以模拟出类似"在示例上做梯度下降"的效果（隐式地拟合了一个小模型）。

> ⚠️ **反直觉但极其重要的坑**：Min et al. (2022) 发现，即使把示例的**标签打乱成随机的**（"太难吃→正面"），性能也只掉一点点。真正起作用的是**示例的格式、标签空间的分布、输入的分布**，而不是"输入-标签配对是否正确"。
> **启示**：花时间纠结示例答案对不对，常常不如花时间把**格式和覆盖面**设计好。

---

### 2. 思维链（Chain-of-Thought, CoT）——本课的核心

**是什么**：不让模型直接蹦出答案，而是先让它**写出中间推理步骤**再给结论。

- **Few-shot CoT**（Wei et al. 2022）：示例里就把解题步骤写出来。
- **Zero-shot CoT**（Kojima et al. 2022）：一句魔法咒语 `Let's think step by step` 就能触发，几乎零成本。

**为什么有效（这是今天最该带走的第一性原理）**：

一个 Transformer **每生成一个 token 所做的计算量是固定且有限的**（深度固定、大致处于 TC⁰ 这个"浅并行"复杂度类）。这意味着——**有些问题的答案，靠"一步前向传播"根本算不完**。就像要求你心算 `17 × 483` 必须张嘴就报，不许打草稿。

CoT 的本质是：**把输出序列本身当成"外置草稿纸/工作记忆"，把需要串行的计算，摊开成一串 token 逐步做。**

- 理论支撑（Liu et al. 2024、Feng et al.）：给足够多的中间步骤，CoT 能把模型的**表达能力从 TC⁰ 扩展到多项式时间可解的问题**。换句话说，CoT 不是"提示技巧"，而是**用 token 换算力**。

> 🔑 **一句话记住**：**CoT = 把"并行的一次前向"改写成"串行的多步计算"，token 就是计算的时间。**

---

### 3. 分解（Decomposition / Least-to-Most）

**是什么**：把一个大问题**显式拆成有序的子问题**，逐个解决，把前一步的结果喂给下一步。

**与 CoT 的区别（易混点）**：
- CoT 是**一条连续的思维流**，通常一次调用生成。
- 分解是**结构化的多步**，常常是**多次 LLM 调用**、每步聚焦一个子任务，可控性更强、更适合工程化（还能在中间插入工具调用、校验）。

> 直觉：CoT 像"一口气把心算过程念出来"；分解像"列一张待办清单，一项项打勾，每项单独做"。任务越复杂、越需要中途纠错，越该往分解靠。

---

### 4. 自一致性（Self-Consistency）

**是什么**：对同一个问题，用**较高温度采样出 N 条不同的 CoT 推理路径**，然后对最终答案**投票取多数**。

```
问题 Q
 ├─ 采样路径1 → 答案 A
 ├─ 采样路径2 → 答案 B
 ├─ 采样路径3 → 答案 A
 ├─ 采样路径4 → 答案 A
 └─ 采样路径5 → 答案 B
        多数投票 ⇒ A
```

**为什么有效（机制）**：把中间推理路径记作隐变量 z，最终答案 a。单条 CoT 是从 `p(a, z | Q)` 里采一个样本——**方差很大，一步走错满盘皆输**。自一致性做的是**对推理路径求边缘化**：

$$a^* = \arg\max_a \sum_{z} p(a, z \mid Q)$$

用采样近似这个求和。这本质上是**在"推理空间"里做集成（ensembling）**：多条独立思路若都指向同一答案，可信度就高。代价是 **N 倍的推理开销**——典型的"花钱买准确率"。

> ⚠️ **坑**：自一致性只适用于**答案可比较/可归一化**的场景（数学题、选择题）。开放式生成没法"投票"，要换成 reward model 打分或其他聚合方式。

---

### 🧭 四招的统一视角（把知识钉在一起）

| 技巧 | 一句话本质 | 它在"掰"什么 |
|------|-----------|-------------|
| Few-shot | 用示例定位任务 & 锁定格式 | 帮模型**定位**到正确的条件分布 |
| CoT | 用 token 换串行算力 | 给模型更多**计算步数** |
| 分解 | 把大问题拆成可控子步 | 用**结构**降低单步难度 |
| 自一致性 | 多路径采样 + 投票 | 用**集成**压低方差 |

**常见误区总清单**：
1. **CoT 不一定"忠实"**：模型写出来的推理，可能是**事后合理化（post-hoc rationalization）**，不是它真正的因果计算过程（Turpin et al. 2023）。别把 CoT 文本当成模型"内心的真实想法"。
2. **甚至无意义的 filler token 也能提升性能**（Pfau et al. 2024）——暗示 CoT 的部分收益单纯来自"多给了计算预算"，而非文字语义。
3. **推理模型（o 系列/reasoning models）改变了游戏规则**：它们已经通过 RL 内化了推理，你再手动堆 few-shot 或强行"think step by step"，可能**收益递减甚至变负**（见下方动态）。
4. **示例不是越多越好**：存在顺序偏置（近因效应）、上下文变长带来的干扰；选例和排序本身是门学问。

---

## 二、最新动态 / 论文速览

1. **Context Engineering: From Prompts to Corporate Multi-Agent Architecture** · arXiv:2603.09619 · 2026-03
   - **为什么重要**：2026 年最明显的范式迁移信号——从"提示工程（抠措辞）"走向"**上下文工程（Context Engineering）**"：管理"检索→组织→隔离→经济→溯源"这整条上下文生命周期，并把 context 视作 agent 的"操作系统"。
   - **与今日主题的关系**：今天学的四招是"上下文工程"的**地基与最小单元**。你得先懂怎么摆好一个 prompt，才谈得上管理成千上万条动态上下文。这是本主题通往轨道 A 后续（RAG、Agent）的桥。

2. **Context-CoT: Enhancing Context Learning via High-Quality Reasoning Synthesis** · arXiv:2605.25354 · 2026-05
   - **为什么重要**：直指传统 ICL 的软肋——few-shot/CoT 主要是在**静态的预训练知识**上做推理；而真实应用越来越需要模型对**新的、任务特定的信息**（RAG、agentic pipeline 提供）做推理。该文用"高质量推理合成"来强化这一点。
   - **与今日主题的关系**：帮你看清 few-shot/CoT 的**能力边界**——它们擅长"唤起已有能力"，不擅长"注入新知识"。这正是明天为何要学 RAG 的动机。

3. **关于"提示技巧在新模型上可能负收益"的基准发现**（多篇 2026 基准/综述反复提及，如 CL4SE, arXiv:2602.23047）
   - **为什么重要**：随着 o1 类推理模型出现，研究者发现**早期提示技巧（复杂 few-shot、显式 CoT 咒语）在这些模型上可能不再显著、甚至带来负收益**。经典结论要绑定"它所属的模型时代"来看，不是普适定律。
   - **与今日主题的关系**：直接支撑上面"常见误区 3"。提醒你：**任何提示技巧都要在你实际用的那代模型上 A/B 验证**，别照搬 2022 年的论文结论。

> ✅ 本次已联网，以上为近 30 天内（2026-02 至 05）相关进展。

---

## 三、🎯 留给明天的钩子

- **今天点到但没展开的**：
  - ICL "为什么有效"的两派解释（任务定位 vs 隐式梯度下降），背后是一整块**机制可解释性**研究——留待轨道 C。
  - CoT 的**忠实性（faithfulness）** 与安全的关系（能不能信任模型自陈的推理）——留待轨道 A"安全与护栏"或轨道 C"可解释性"。
  - 自一致性、Tree/Graph-of-Thoughts、test-time compute 是一条完整的"**测试时计算**"主线——留待轨道 C"推理模型：test-time compute"。
- **建议下次深入**：明天是**轨道 B（基础原理）**。建议开讲 **注意力机制与 Transformer 架构（L1）**——正好回答今天反复用到的"一次前向传播到底算了什么、为什么算力是固定的"，与 CoT 的"用 token 换算力"形成闭环。

---

## 四、📚 延伸阅读（可选）

- 📄 Wei et al. (2022) *Chain-of-Thought Prompting Elicits Reasoning in LLMs*（经典必读，CoT 开山作）
- 📄 Wang et al. (2023) *Self-Consistency Improves Chain of Thought Reasoning*（自一致性原文）
- 📄 Min et al. (2022) *Rethinking the Role of Demonstrations*（随机标签也有效，颠覆直觉）
- 🔗 [Prompt Engineering Guide](https://www.promptingguide.ai/)（体系化的技巧速查手册）
- 🔗 [A Survey of Context Engineering for LLMs](https://github.com/Meirtz/Awesome-Context-Engineering)（2026 上下文工程综述，看趋势）

---
---


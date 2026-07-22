---
id: lesson-0003
source:
  file: lessons/2026-07-07.md
  section: 1
  hash: sha256:1902f94e2045fbe5fc5aaa22fa0394bffe000cb60c20809b1a71bf91e1c41782
lesson: 3
date: 2026-07-07
track: C
depth: L2
titleZh: 测试时计算与推理模型（Test-Time Compute & Reasoning Models）
titleEn: Test-Time Compute and Reasoning Models
summaryZh: 额外算力砸在推理而非参数；两家族=序贯(长CoT/RL,用token换深度)与并行(采N条+验证器选优);coverage随N对数线性涨但坏验证器致收益反转;compute-optimal按难度自适应分配;本质=对固定模型后处理,逼不出未知知识
summaryEn: Test-time compute spends additional inference resources without changing model parameters. The lesson contrasts sequential long reasoning with parallel sampling and verification, explains why weak verifiers can reverse scaling gains, and motivates difficulty-aware allocation while emphasizing that extra inference cannot create knowledge absent from the model.
slug: test-time-compute-reasoning-models
tags:
  - test-time-compute
  - reasoning-models
  - verifiers
sourceStatus: unreviewed
sourceStatusHash: sha256:1902f94e2045fbe5fc5aaa22fa0394bffe000cb60c20809b1a71bf91e1c41782
metadataStatus: current
metadataSourceHash: sha256:1902f94e2045fbe5fc5aaa22fa0394bffe000cb60c20809b1a71bf91e1c41782
featured: false
---
# 📅 2026-07-07 · 轨道C · 测试时计算与推理模型（Test-Time Compute & Reasoning Models）· 深度 L2

> **TL;DR**：传统 Scaling Law 把算力砸在**训练**（更大参数、更多数据），而"测试时计算"把额外算力砸在**推理**——让模型在给出答案前"多想一会儿"。Snell 等人（2024）证明：给定固定算力预算，把它花在推理上，有时比放大参数更划算。但这不是免费午餐——它只能**逼近模型已有能力的上限**，逼不出模型不知道的知识。

---

## 一、核心概念精讲（L2）

### 1. 直觉层（L1，快速过一遍）

先建立最小心智模型。**训练 = 平时学习**，把知识压进参数里，训完就固定了；**测试时计算 = 考场上的草稿纸和时间**，让模型在同一套固定参数下，通过"演算更多步、生成更多候选、反复检查"来提升答对的概率。

一个精准类比：两个学生考同一份数学卷。
- 学生甲：天赋高（参数多），但草率，看一眼就写答案。
- 学生乙：天赋中等（参数少），但肯打草稿、反复验算、换几种方法交叉检验。

Snell 等人的核心洞见就是：**在很多推理任务上，"让乙多算"比"把乙换成甲"更省成本**。这在 2024 年之前是反直觉的——那之前所有人都盯着"把模型做大"。o1、DeepSeek-R1、Claude 的 extended thinking 全是这条路线的产物。

> ⚠️ **常见误区①**："测试时计算能替代训练。" 错。下面 L2 会看到，它本质是对一个**固定模型**做后处理，天花板由训练决定。

### 2. 机制层（L2，本课主体）

测试时计算分两大家族，区别在于**额外的 token 花在"深度"还是"宽度"**。

#### 家族 A：序贯扩展（Sequential / 深度）
让模型自己生成一条**很长的思维链（CoT）**，中途自我反思、回溯、纠错，最后收敛到答案。这就是 o1 / R1 "长思考"的路子。

- **机制**：把"思考"外化成更长的自回归 token 序列。每多生成一个 token，模型就多做一次前向计算——相当于**用 token 数量换取计算深度**（这正好接上我们第 1 天讲的 CoT："用 token 换算力"）。
- **怎么训出来的**：主流做法是**强化学习（RL）**，对最终答案的正确性给 reward（R1 用的 GRPO、o 系列用的类似 RL 目标），模型自己学会"什么时候该多想、该回溯"。

#### 家族 B：并行扩展（Parallel / 宽度）
对同一个问题**采样 N 个独立的解**，再从中选一个。选法有三档，精度递增：
1. **多数投票（self-consistency）**：采 N 条 CoT，答案取众数。第 1 天讲过——本质是蒙特卡洛边缘化掉推理路径的噪声。
2. **Best-of-N + 验证器（Verifier）**：训一个 reward model 给每个候选打分，取最高分。
   - **ORM（结果奖励模型）**：只看最终答案对不对。
   - **PRM（过程奖励模型）**：给推理的**每一步**打分，能定位"从哪一步开始错"，信号更细。

#### 关键公式一：覆盖率随采样数增长
并行采样为什么有效？设单次采样答对概率为 $p$，采 $N$ 次里**至少有一条对**的概率（称 coverage / pass@N）：

$$\text{coverage}(N) = 1 - (1-p)^N$$

Brown 等人（2024）实测发现：coverage 随 $N$ 呈**对数线性**增长——采样越多，"解空间里被覆盖到正确答案"的概率稳步上升。这解释了为什么"多采几次总能蒙对一条"。

**但注意这里的陷阱**：coverage 衡量的是"存在一条对的"，可你**得能把它挑出来**。这就引出下一个公式。

#### 关键公式二：不完美验证器的收益递减
理想情况下你有个完美验证器，能从 N 条里精准挑出那条对的，那 coverage 涨多少、准确率就涨多少。现实中验证器有**假阳性率**（把错的判成对的）。Stroebl 等人（2024）从理论上证明：当采样数 $N$ 很大时，**假阳性会主导结果**——你采得越多，越可能采到一条"看起来对、验证器也点头、实则错"的解。于是准确率先升后**降**，出现收益递减甚至反转。

一句话总结这对张力：**采样提升的是"上界"（coverage），验证器决定你能兑现多少（precision）；验证器不够好时，加算力反而有害。**

#### 关键机制三：Compute-Optimal（算力最优分配）
Snell 等人的真正贡献不是"多想有用"，而是**"给定预算 C，怎么分配最优"**，且答案**取决于题目难度**：
- **简单题**：模型第一版答案大概率八九不离十 → 把预算花在**少量并行采样 + 轻微修正**上，性价比最高。
- **困难题**：需要**更深的序贯修正**（长 CoT、多轮回溯），并行乱采意义不大。

所以"compute-optimal scaling"是一个**自适应策略**：先估计 prompt 难度，再决定把算力投向"深度"还是"宽度"。他们的实测结论很强：在算力受限时，这套自适应分配可以让一个小模型的表现**追平甚至超过一个大 14 倍的模型**（在特定推理任务上）。

> ⚠️ **常见误区②**："想得越久越准，单调递增。" 错。见下方最新动态 arXiv:2509.06861——过度推理会诱发 **confirmation bias（确认偏误）**，让模型对错误答案越想越自信，幻觉不降反升。

### 3. 走一遍完整例子

**题目**：一个水池，甲管单独注满需 6 小时，乙管单独注满需 4 小时，两管齐开几小时注满？（正确答案：$1/(1/6+1/4)=2.4$ 小时）

设模型单条 CoT 的答对率 $p=0.6$，我们采 **N=5** 条，看到如下结果：

| 采样 | 推理要点 | 答案 |
|------|---------|------|
| #1 | 1/6+1/4=5/12，取倒数 | **2.4h** ✅ |
| #2 | 错把速率相加算成 6+4 | 10h ❌ |
| #3 | 1/6+1/4 算成 2/10 | 5h ❌ |
| #4 | 正确通分 | **2.4h** ✅ |
| #5 | 正确通分 | **2.4h** ✅ |

现在对比三种"选择"策略怎么兑现这批样本：
- **Coverage（上界）**：5 条里存在正确解 → coverage 命中 = 1。理论天花板 100%。
- **多数投票**：2.4h 出现 3 次，10h 和 5h 各 1 次 → 众数 = **2.4h ✅**。免费、无需训验证器，就把 $p=0.6$ 提到了正确。
- **Best-of-N + PRM**：若 PRM 靠谱，会给 #2 的"6+4"步打低分、给 #1/#4/#5 的通分步打高分 → 选出 **2.4h ✅**，且能告诉你 #2 **错在第一步**（多数投票做不到这点）。
- **如果验证器很烂**（假设它偏爱"整数答案"）：可能把 #2 的 10h 选出来 → **错**。这就是 Stroebl 警告的场景：加了采样，却被坏验证器带沟里。

这一遍你能看到：**同一批候选，兑现多少完全取决于选择机制**。序贯路线则是另一条：不采 5 条，而是让模型生成 1 条长 CoT，写到 #2 那种错误时自己反思"速率不能直接相加"，回溯改正——用深度换宽度。

### 4. 工程权衡（点到，主体仍在 L2）

- **序贯 vs 并行的硬约束**：并行可以多卡同时跑、延迟低，但吃显存/吞吐；序贯延迟高（token 必须一个个出，且 KV cache 随长度线性涨），但对难题更有效。生产系统常**混合**：先并行采几条，对最难的再序贯深挖。
- **天花板问题（重要）**：Setlur 等人（ICML 2025）证明"没有验证器或 RL 的纯测试时扩展是次优的"——光靠采样+启发式选择，逼不出模型的真实上限，必须有高质量验证信号。

---

## 二、最新动态 / 论文速览

1. **Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters**（Snell, Lee, Xu, Kumar）· arXiv:2408.03314 / ICLR 2025 ·（**经典奠基，非近期**）
   - 为什么重要：本领域的"开山之作"，第一次系统论证测试时计算的 compute-optimal 分配，并给出"小模型+多想 ≥ 大模型"的量化结论。
   - 与今日主题的关系：本课的理论骨架就出自这篇。

2. **Test-Time Scaling in Reasoning Models Is Not Effective for Knowledge-Intensive Tasks Yet**（Zhao, Hooi, Ng · 新加坡国立）· arXiv:2509.06861，v2 修订于 2026-01-31 ·（近期）
   - 为什么重要：一篇关键的"泼冷水"研究。在 14 个推理模型 + 2 个知识密集基准上发现：加测试时计算**不稳定提升准确率，反而常增加幻觉**；并给出信息论论证——纯测试时扩展是对固定模型的后处理，**无法增加模型里本就不存在的关于答案的信息**。
   - 与今日主题的关系：直接界定了本方法的**适用边界**（推理/逻辑密集 ✅，知识密集 ❌），对应我们讲的"误区②"。

3. **Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph**（Wang 等）· arXiv:2511.00086，提交于 2025-10-29 ·（近期）
   - 为什么重要：把"如何组合并行/序贯/多模型协作"抽象成一张**可优化的图**（节点=角色与模型分配，边=信息流），并用 Agent-REINFORCE 搜索最优结构；实证再次确认"**超过最优预算后算力收益转负**"。
   - 与今日主题的关系：是 compute-optimal 从"单模型难度自适应"到"多模型协作架构自适应"的自然推广。

4. **Parallel Test-Time Scaling for Latent Reasoning Models**（arXiv:2510.07745，2025-10）·（近期）
   - 为什么重要：把并行扩展从"文本 token 空间"推进到**连续隐空间**——用 Monte Carlo Dropout / 高斯噪声做采样，用 LatentRM（隐空间奖励模型）做聚合，探索"不吐字也能并行想"。
   - 与今日主题的关系：并行家族的前沿变体，提示测试时计算未必绑定在离散 token 上。

---

## 三、🔁 旧知回顾（间隔重复日 · 第 3 天）

> 台账目前仅 2 个历史主题，故从中出 3 题，覆盖两天要点。问题在前，答案折叠。

- **Q1**：自注意力里为什么要除以 $\sqrt{d_k}$？不除会怎样？
- **Q2**：self-consistency（自一致性）从概率角度看，本质在做什么运算？它和今天讲的"多数投票"是什么关系？
- **Q3**：因果掩码（causal mask）在注意力矩阵上具体是怎么实现的？它保证了什么？

<details><summary>点击看答案</summary>

- **A1**：$Q\cdot K^\top$ 的点积方差随维度 $d_k$ 线性增长，数值会很大，导致 softmax 落进饱和区、梯度趋近 0。除以 $\sqrt{d_k}$ 把方差拉回 ~1，防止 softmax 过于尖锐、保持梯度健康。
- **A2**：本质是**对推理路径这个隐变量做蒙特卡洛边缘化**——采样多条 CoT，把不同路径带来的噪声平均掉，只保留稳定收敛的答案。它就是今天"并行扩展家族"里最基础的那一档（多数投票），无需训练额外验证器。
- **A3**：在 $QK^\top/\sqrt{d}$ 得到的注意力分数矩阵上，把每个 query 位置 $i$ 对所有 $j>i$（未来位置）的分数置为 $-\infty$，再过 softmax 使这些位置权重变 0。它保证每个 token **只能看到自己及左侧的前文**，从而支持自回归生成、不泄露未来信息。

</details>

---

## 四、🎯 留给明天的钩子

- **今天点到但没展开的**：
  - PRM（过程奖励模型）到底怎么训、怎么标注"每一步对错"？MCTS 式的树搜索（Tree-of-Thoughts）如何与验证器结合？
  - 序贯路线背后的 RL 细节（GRPO / PPO 如何用"答案正确性"这个稀疏 reward 训出长思考）——这正好是**轨道 B「对齐算法 RLHF/PPO/DPO」**的入口。
- **建议下次深入**：
  - 轨道 B · L2：**RL 对齐算法（PPO → GRPO → DPO）**，把今天"怎么训出会思考的模型"补齐。
  - 或轨道 C 复访升级本主题至 **L3**：切入"验证器工程"——PRM 数据构造、reward hacking、验证器与生成器的对抗。

---

## 五、📚 延伸阅读

- Snell et al., *Scaling LLM Test-Time Compute Optimally…*（arXiv:2408.03314）——先读 §3 问题设定与 §5 compute-optimal 策略。
- Brown et al., 2024, *Large Language Monkeys*（repeated sampling / coverage 的 log-linear 规律）——理解并行扩展为何有效的实证基础。
- DeepSeek-R1 技术报告——序贯路线（纯 RL 训出长 CoT）的最佳工业级案例。

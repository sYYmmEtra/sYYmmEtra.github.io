---
id: lesson-0016
source:
  file: lessons/2026-07-20.md
  section: 1
  hash: sha256:dd205a259a1367433a15ccca8fa9fbb5fd1b1302f1376049fd98ff8f26a97191
lesson: 16
date: 2026-07-20
track: A
depth: L2
titleZh: Agent 基础（ReAct · 规划 · 工具编排）
titleEn: "Agent Fundamentals: ReAct, Planning, and Tool Orchestration"
summaryZh: 新主题(闭合day4/day13两段式钩子)。Agent=无状态LLM大脑+外层while循环+世界;ReAct=给day13填表员→host执行→解说员两段式加一条显式Thought再套进循环:Thought(为何,是day3花在规划上的test-time compute,用token换深度)→Action(做什么,=day4/13的tool_call,受约束解码保schema合法)→Observation(host执行后填回)。第一性原理:Agent无隐藏状态,全部"记忆"=不断增长的context文本(呼应day1条件化自回归),故智能一半在模型一半在外层loop怎么拼context/何时停/给啥工具。loop骨架必须有MAX_STEPS预算防死循环烧钱;停止由模型自己Finish决定。两方言:prompt版ReAct(正则解析Thought/Action脆)vs native function-calling版(day13受约束解码保合法,现代生产用这个)。编排两骨架:ReAct交替式(走一步看一步,适应强但每步一次推理贵)vs Plan-and-Execute(先规划后执行,省调用可并行但计划错则整盘输),生产常混用。误区:①Agent≠更聪明的模型(能力一半在harness,换强模型≠换架构)②Thought只是文本scratchpad,想得漂亮≠做得对(合法≠正确,承day13)③无停止条件→死循环④全历史无脑塞context→溢出+context rot⑤Observation啰嗦稀释信号 [近期动态已核实:Less Context Better Agents 2606.10209裁剪Observation反更准/SmoothAgent 2607.00151改context需重算KV cache呼应day8;OpenAI实践指南日期未核验]
summaryEn: An introduction to agents as stateless LLMs embedded in controlled loops that manage tools, context, and termination. It explains the Thought-Action-Observation cycle, contrasts prompt-based ReAct with native function calling, compares ReAct and plan-and-execute orchestration, and highlights context growth, verbose observations, unreliable reasoning, and missing execution limits.
slug: lesson-0016
tags:
  - llm-agents
  - react
  - tool-orchestration
  - function-calling
  - context-engineering
sourceStatus: unreviewed
sourceStatusHash: sha256:dd205a259a1367433a15ccca8fa9fbb5fd1b1302f1376049fd98ff8f26a97191
metadataStatus: current
metadataSourceHash: sha256:dd205a259a1367433a15ccca8fa9fbb5fd1b1302f1376049fd98ff8f26a97191
featured: false
---
# 📅 2026-07-20 · 讲16 · 轨道A · Agent 基础（ReAct · 规划 · 工具编排） · 深度 L2

> **TL;DR**：Agent 不是"更聪明的模型"，而是把**无状态的 LLM 放进一个由外层代码控制的 while 循环**里——ReAct 的本质就是给 day13 的「填表员→host执行→解说员」两段式加上一条显式 `Thought`、再套进循环：`想 → 做 → 观察 → 再想`，直到自己喊停。模型的"记忆"就是那段不断变长的 context，这也埋下了它最大的工程病根。

## 一、核心概念精讲（L1→L2）

### L1 · 一句直觉：Agent = 大脑 + 循环 + 世界

把 LLM 当成一个**只会"读一段文字→写下一段文字"的大脑**（day1：prompt 是在条件化自回归分布）。它本身不能上网、不能算数、不能记住上一秒——一次前向传播完就"失忆"。**Agent（智能体）** 就是在这个大脑外面裹一层代码（harness/外壳），让它能：调用工具、把工具结果读回来、循环多轮，直到解决问题。

ReAct（**Rea**soning + **Act**ing，"边想边做"，Yao et al. 2022）解决的是它之前两条路各自的残疾：

- **只推理（CoT，day1）**：模型闷头空想，但事实全靠脑补——问它"某 CEO 哪年出生"，它会一本正经编一个年份（hallucination）。**有脑无手。**
- **只行动（早期 tool-use / WebGPT）**：模型盲目调工具，但不"想"为什么调、下一步该干嘛，遇到意外不会调整。**有手无脑。**

> **类比**：ReAct 像个侦探办案。不是坐在椅子上空想推理（会脑补线索），也不是无头苍蝇乱翻档案（没策略），而是**"想一步（这案子得先查不在场证明）→ 做一步（调监控）→ 看结果（他不在现场）→ 再想（那得查同伙）"**。推理指导行动，行动的观察又反哺推理，两者咬合前进。

### L2 · 机制：Thought–Action–Observation 循环

ReAct 把 Agent 的每一轮拆成三段文本，循环拼接进 context：

| 段 | 内容 | 谁生成 | 呼应旧课 |
|---|---|---|---|
| **Thought** | "为什么"——规划/拆解/自查/纠错的思考 | LLM 生成 | 是花在**规划上的 test-time compute（day3）**，用 token 换深度 |
| **Action** | "做什么"——调哪个工具、传什么参数（结构化） | LLM 生成 | 就是 day4/13 的 `tool_call`，受约束解码保证 schema 合法 |
| **Observation** | 工具跑完的返回结果 | **host 代码**执行后填回 | day13"host 才真正执行函数"的那一步 |

**第一性原理：整个 Agent 没有隐藏状态，它的全部"记忆"就是这段不断增长的 context 文本。** 模型每一步看到的，是「系统提示（角色 + 工具 schema + 格式说明 + 几条 ReAct 示范轨迹）+ 到目前为止的 Thought/Action/Observation 全历史」，然后自回归续写下一段。所以——**Agent 的智能一半在模型，一半在外层这个 loop 怎么拼 context、何时停、给什么工具。**

**外层 loop 的伪代码（这就是"Agent"的全部骨架）：**

```
context = system_prompt + tools_schema + few_shot_react_examples + user_task
for step in range(MAX_STEPS):                 # ← 必须有预算上限，否则死循环烧钱
    out = LLM(context)                        # 模型续写 Thought + Action
    thought, action = parse(out)              # 或用 native function-calling 直接拿结构化 tool_call
    if action.name == "Finish":               # 模型自己决定"够了，交答案"
        return action.answer
    observation = TOOLS[action.name](**action.args)   # ← host 真正执行（day13）
    context += out + format(observation)      # 把这一轮追加回 context，进入下一圈
```

**注意这个循环里 LLM 被调用了 N 次**——每一步 Thought 都是一次完整前向传播。这既是 ReAct 强的原因（每步都能重新规划、处理意外），也是它贵、慢的根源。

**两种工程方言（重要）：**
1. **Prompt 版 ReAct**：让模型在纯文本里吐 `Thought: ... Action: search[...] Observation: ...`，host 用正则解析。原始论文的做法，**脆**——模型格式跑偏就崩。
2. **Native function-calling 版**：Action 走 day13 的 `tool_call` 结构化通道，受约束解码（CFG→PDA）**保证 schema 合法**。现代 Agent（OpenAI/Anthropic tool use、LangGraph 等）都用这种。**day13 学的受约束解码，正是让 ReAct 从"实验室 prompt trick"变成"生产可靠组件"的地基。**

### 🔬 完整走一遍：一个真实的 ReAct 轨迹

**任务**：「Anthropic 现任 CEO 是哪年出生的？他到 2026 年多大？」（模型参数里既可能没有、也不该硬记这种事实——正好逼它用工具）

```
User: Anthropic 现任 CEO 哪年出生？到 2026 年多大？

Thought: 我不确定 CEO 是谁、更不知道生年，硬答会编。先查 CEO 是谁。   ← 显式承认无知，规划第一步
Action: web_search{"q": "Anthropic CEO 2026"}
Observation: "Anthropic 的 CEO 是 Dario Amodei。"                      ← host 执行搜索填回

Thought: 拿到人了，但没生年。再查 Dario Amodei 的出生年份。            ← 用上一步观察，规划第二步
Action: web_search{"q": "Dario Amodei date of birth"}
Observation: "Dario Amodei 出生于 1983 年。"

Thought: 有了生年 1983，算年龄是纯数学、别让我自己心算(易错)，交给计算器。 ← 自查+把易错子任务外包
Action: calculator{"expr": "2026 - 1983"}
Observation: 43

Thought: 信息齐了，可以作答。                                          ← 决定停止
Action: Finish{"answer": "Anthropic 的 CEO 是 Dario Amodei，1983 年生，到 2026 年 43 岁。"}
```

看清三件事：① **推理和行动真的在交替**，每个 Thought 都基于上一个 Observation 改变计划；② 模型**主动把易错子任务（算数）外包给可靠工具**——这正是 Agent 比裸 LLM 强的核心；③ 停止是**模型自己判断**的（`Finish`），不是外层写死步数。

### 工程权衡（L2 收尾，点到即止，留 L3 深挖）

- **ReAct（交替式）vs Plan-and-Execute（先规划后执行）**：ReAct 走一步看一步，适应性强但每步都要一次推理、贵；Plan-and-Execute 先让模型列好全程计划再逐条执行，省调用、可并行，但计划一旦基于错误假设就整盘皆输、中途难纠偏。生产里常**混用**：先粗规划，执行中用 ReAct 局部纠错。这就是"工具编排（orchestration）"的两种基本骨架。
- **context 会爆**：每个 Observation 都往 context 里追加，多跑几十步就撑爆窗口，或触发"context rot"（越长越噪、性能下滑）——这正是今天最新动态的主题，也是 2026 年 Agent 工程的头号战场。

### ⚠️ 常见误区 / 易错点

1. **"Agent = 一个更聪明的模型"** ——错。模型是无状态的；Agent 的能力一大半来自**外层 harness**（怎么管 context、给哪些工具、何时停）。换个更强模型 ≠ 换个 Agent 架构。
2. **把 Thought 当成"真规划"** —— Thought 只是文本 scratchpad，模型完全可能"想得漂亮、做得错"（呼应 day13：**合法/流畅 ≠ 正确**）。别因为轨迹看着有条理就信它对。
3. **没有停止条件 / 预算** —— 忘了 `MAX_STEPS` 或没设 token 预算 → 模型陷入"再查一次…再查一次…"的死循环，账单爆炸。
4. **把全部历史无脑塞回 context** —— 天真做法，直接导致 context 溢出与 rot（见下）。
5. **Observation 太啰嗦稀释信号** —— 企业工具（如 MCP）常返回整页状态快照，几步就把窗口填满、把关键信息淹没。

## 二、最新动态 / 论文速览

`✅ 已联网核实；第 3 条（OpenAI 实践指南）已联网但未逐字核验其发布日期，谨慎参考`

1. **《Less Context, Better Agents: Efficient Context Engineering for Long-Horizon Tool-Using LLM Agents》** · arXiv:2606.10209 · 2026-06 —— **为什么重要**：直击今天讲的 ReAct 病根——每步 Observation 追加导致 context 爆炸。它在企业场景（D365 MCP 工具返回"整页表单状态快照+元数据+导航面包屑"）实测，证明**主动裁剪/结构化 Observation 反而让 Agent 更准更省**。**与今天的关系**：把误区 4/5（context 无脑增长、Observation 啰嗦）从"注意事项"升级成一个有数据的研究方向。
2. **《SmoothAgent: Efficient Long-Horizon LLM-Based Agent Serving with Lookahead Context Engineering》** · arXiv:2607.00151 · 2026-07 —— **为什么重要**：点明每次改写 context（压缩/摘要/外置记忆）都要**重算 KV cache（呼应 day8！）**，落在推理关键路径上、造成 TTFT 尖峰。**与今天的关系**：把 ReAct 循环的"每步都拼 context"和 day8 的 KV Cache 世界观直接接上——Agent 的延迟瓶颈本质仍是显存带宽。
3. **OpenAI《A Practical Guide to Building Agents》（生产模式指南）** · OpenAI 官方 · 2026-04（日期未核验）—— **为什么重要**：把单 Agent vs 多 Agent 编排（manager 集中式 vs decentralized handoff 去中心化交接）、工具设计、分层护栏（输入校验/输出过滤/工具风险分级/人工介入）系统化。**与今天的关系**：今天的单循环 ReAct 是最小骨架，这份指南是它往"多 Agent 工具编排"扩展的下一张地图。
4. **《Agentic AI: Architectures, Taxonomies, and Evaluation of LLM Agents》** · arXiv:2601.12560 · 2026-01 —— **为什么重要**：给 Agent 家族做谱系分类（从 ReAct 的 think-act 环 → Reflexion 加自我批判与记忆 → 工具增强框架），是建"Agent 知识地图"的骨架文献。**与今天的关系**：帮你把今天的 ReAct 定位成"最早、最基础的一环"，看清后续升级路线。

## 三、🎯 留给明天的钩子

- **今天点到没展开的**：① Plan-and-Execute vs ReAct 的完整取舍与"何时该规划、何时该交替"；② **Reflexion / 自我批判 + 记忆**（在 loop 里加一条"复盘上次失败"）——它和 day9 的验证器/奖励是近亲：Agent 怎么给自己的轨迹打分并改进；③ 多 Agent 编排（manager / handoff）与"何时**不**该上框架"。
- **建议下次深入（A 轨）**：把本讲复访升级到 **L3**——「Agent 工程：context engineering / 记忆系统 / 错误恢复」，直接吃掉今天最新动态 1/2 两篇；或开新题 **「多 Agent 与编排框架（LangGraph）以及何时不该用框架」**。
- **跨轨联动**：ReAct 里"检索 = 一种工具"，天然通向 **Agentic RAG**（把 day7/10 的检索塞进 loop）；"给轨迹打分并改进"通向 **B 轨对齐 / C 轨验证器**。

## 四、📚 延伸阅读

- ReAct 原论文：Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*（ICLR 2023）——读它的 Figure 1，四种范式（Standard / CoT-only / Act-only / ReAct）并排对比，一图看懂"边想边做"为何赢。
- Anthropic《Building Effective Agents》——工程视角讲"workflow（写死的编排）vs agent（模型自己决策循环）"何时用哪个，以及别过度设计。
- LangGraph 文档中的 ReAct / state-graph 教程——把今天的伪代码 loop 落成可跑代码，直观看到"外层控制器管 state、每步注入指令"。

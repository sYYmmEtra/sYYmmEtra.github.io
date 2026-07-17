---
id: lesson-0004
source:
  file: lessons/2026-07-08.md
  section: 1
  hash: sha256:d4bd7b6d3d33fdfbbb7a2dfb415654190e70a3dfdbe85c6bac145d51be1cddab
lesson: 4
date: 2026-07-08
track: A
depth: L2
titleZh: 结构化输出与 Function Calling / Tool Use
titleEn: Structured Outputs, Function Calling, and Tool Use
summaryZh: 模型不"执行"函数，只生成符合 schema 的结构化文本(JSON)，host 代码才执行；可靠性靠受约束解码=每步用 schema编译的CFG语法把非法token的logit置-∞，保证结构合法(≠语义正确)；两段式：先当填表员发tool_call，host执行后再当解说员；核心张力=格式税(过早锁JSON挤压推理)
summaryEn: Models do not execute functions; they emit structured text that host code validates and runs. This lesson explains schema-constrained decoding, the difference between syntactic validity and semantic correctness, the two-stage tool-call workflow, and the format tax created when rigid output constraints suppress useful reasoning.
slug: structured-outputs-function-calling
tags:
  - structured-output
  - function-calling
  - tool-use
sourceStatus: unreviewed
sourceStatusHash: sha256:d4bd7b6d3d33fdfbbb7a2dfb415654190e70a3dfdbe85c6bac145d51be1cddab
metadataStatus: current
metadataSourceHash: sha256:d4bd7b6d3d33fdfbbb7a2dfb415654190e70a3dfdbe85c6bac145d51be1cddab
featured: false
---
# 📅 2026-07-08 · 轨道A · 结构化输出与 Function Calling / Tool Use · 深度 L2

> **TL;DR**：Function Calling 的本质**不是**模型真的"调用"了函数——模型只负责**生成一段符合你给定 schema 的结构化文本（通常是 JSON）**，真正执行函数的是你的代码。它把"自由文本的语言模型"接进了"确定性的软件世界"，而让这个接口可靠的关键工程支柱是**受约束解码（constrained decoding）**：在每一步 token 采样时，用 schema 编译出的语法把非法 token 的概率直接置零，从而在**机制层保证**输出一定能被 `json.loads` 解析。

---

## 一、核心概念精讲（L2）

### 1. 先破一个误区：模型并不"调用"函数

很多人第一次接触 Function Calling（函数调用），脑子里的图景是"LLM 联网/执行代码"。**错。** 让我们从第一性原理拆开。

一个 LLM 本质上还是我们前两课讲的那台机器：给定上文，输出下一个 token 的概率分布（回顾 07-06「prompt 是在条件化自回归分布」）。它**不能**做任何 I/O，不能读数据库、不能查天气。

那 Function Calling 到底做了什么？它做的是一件纯粹的"文本"事情：

> 你把一组**工具签名（tool schema）**塞进上下文（"有一个函数叫 `get_weather`，参数是 `{city: string, unit: "C"|"F"}`"），模型在合适的时候，不再生成给人看的自然语言，而是生成一段**机器可解析的结构化文本**，比如 `{"name": "get_weather", "arguments": {"city": "Shanghai", "unit": "C"}}`。

**执行这段文本的是你的宿主代码（host / runtime），不是模型。** 模型只是"填表"。这就是为什么它也被叫 **Tool Use（工具使用）**——模型表达"我想用这个工具、参数是这些"的**意图**，控制权交还给你。

一个精准类比：Function Calling 就像**餐厅点菜的下单流程**。顾客（模型）不进厨房，他只是在**菜单（schema）**限定的选项里，填一张**格式固定的点菜单（JSON）**；后厨（你的代码）拿到单子去做菜，再把菜（函数返回值）端回来。菜单越清晰、点菜单格式越严格，出错越少。

所以整条链路是这样一个循环（ASCII 图）：

```
        ┌─────────────────────────────────────────────────┐
        │  你的宿主代码 (host / orchestrator)              │
        │                                                 │
   (1) 把 tools schema + 用户问题 拼进 prompt              │
        │                          │                      │
        ▼                          │                      │
   ┌─────────┐   (2) 生成 tool_call JSON                   │
   │   LLM   │ ─────────────────────►  (3) host 解析 JSON  │
   └─────────┘                          并真正执行函数      │
        ▲                                   │              │
        │   (5) 把结果拼回上下文，再问一次    ▼              │
        │◄───────────────────  (4) 得到函数返回值           │
        │        LLM 基于结果生成最终自然语言答案            │
        └─────────────────────────────────────────────────┘
```

注意 **(2)→(3) 之间有一道断裂**：模型停在"我要调 get_weather"，你的代码接手执行，再把结果喂回去（step 5）。模型一次前向传播**只走到 (2)**。所谓"Agent"（07-07 台账里出现过的词、也是 roadmap A 后续主题），本质就是把这个循环**自动地多跑几轮**。

### 2. 机制层：结构化输出为什么"能保证"合法？——受约束解码

上面是"是什么"。现在进入 L2 的核心："**怎么保证**模型吐出来的一定是合法 JSON、而不是缺个括号的半成品？"

这里要区分**两种实现层次**，这是今天最重要的知识点：

**(a) 软约束（prompt 层）**：你在 prompt 里写"请只输出 JSON、符合以下 schema"。模型**大概率**照做，但没有任何保证——它可能多写一句"好的，这是你要的 JSON："，或者在长输出里漏个引号。本质上你只是在**祈祷**采样分布落在合法区域。

**(b) 硬约束 / 受约束解码（constrained decoding，也叫 grammar-guided / structured generation）**：这是工程上真正可靠的做法，也是 OpenAI `strict: true`、Anthropic、以及 vLLM/SGLang 等推理引擎底层在做的事。

它的第一性原理其实很简单，一句话：**在每一步解码时，把"此刻不可能合法"的 token 的概率强行置为 0，再归一化后采样。**

展开讲清机制（这一步是理解全部结构化输出技术的钥匙）：

1. 你的 JSON Schema 先被**编译成一个形式语法**——通常是**上下文无关文法（CFG, Context-Free Grammar）**，或等价的一台下推自动机 / 有限状态机。"上下文无关文法"这里指一套产生式规则，能描述"JSON 必须括号配对、key 后面跟冒号"这类嵌套结构（正则做不到嵌套配对，所以需要 CFG）。
2. 解码时维护一个"当前解析状态"。在生成第 t 个 token 前，语法引擎回答一个问题：**从当前状态出发，词表里哪些 token 是"能让字符串继续保持合法前缀"的？** 这批 token 构成一个**掩码（mask）**。
3. 把模型算出的 logits 中，掩码外的 token 全部设为 −∞（即概率 0），只在合法集合里做 softmax 采样。

用公式写就是，标准采样是 $p(x_t) = \mathrm{softmax}(z_t)$；受约束解码变成：

$$
p(x_t \mid \text{state}) = \mathrm{softmax}\big(z_t + m_t\big),\quad
m_t[i] = \begin{cases} 0 & \text{token } i \text{ 合法}\\ -\infty & \text{否则}\end{cases}
$$

其中 $z_t$ 是模型原始 logits，$m_t$ 是语法引擎当步算出的**布尔掩码**。**因为非法路径概率恒为 0，最终字符串在数学上不可能违反语法**——这就是"保证"二字的来源，它是结构上的保证，不是概率上的侥幸。

> **常见误区 / 易错点：**
> - **误区①："结构合法 = 内容正确"。** 大错。受约束解码只保证 `{"age": <数字>}` 的**形状**对，不保证那个数字是对的——它照样能填 `999`。**结构有效性 ≠ 语义正确性，所以拿到结果后你仍然要做业务校验。**
> - **误区②：约束是"免费"的。** 不是。见下节"格式税"。
> - **误区③：掩码计算很慢所以不实用。** 早期确实慢（每步遍历几万词表）。但 2024–2025 的 **XGrammar** 等引擎用"词表分区 + 自适应 token 掩码缓存"把这步做到近乎零开销，如今已是 vLLM / SGLang / TensorRT-LLM 的默认后端。

### 3. 工程权衡（L2 里点出、留 L3 深挖）

- **"格式税"（Format Tax / 结构 vs. 推理的张力）**：这是当前最活跃的研究争论点。把模型过早锁进 JSON 的紧身衣，会**挤压它自由推理的空间**——本来它想先 Chain-of-Thought（07-06 学过：用 token 换算力）想一想，但语法逼它第一个 token 就得吐 `{`，推理链被掐断，某些任务准确率因此下降。这不是玄学，下面「最新动态」第 1 篇论文正是冲着这个问题去的。**实践缓解**：在 schema 里显式留一个 `"reasoning": string` 字段放在业务字段**前面**，让模型先在里面自由想、再填结论（这也呼应了"Schema 字段命名本身是一条隐式指令通道"的研究发现）。
- **原生 Function Calling vs. 自己 prompt 拼 JSON**：优先用厂商原生的 tool-calling API（它在底层挂了 constrained decoding，且模型专门微调过 tool 格式），比你手写"请输出 JSON"稳得多。
- **单工具 → 多工具编排**：给 1 个工具容易，给 20 个工具时模型的**工具选择**开始出错（选错工具、该调不调、不该调乱调）——这正是今天要点评的 ToolFailBench 揭示的失败谱系，也是 roadmap A「Agent 基础」的引子。

### 4. 完整走一遍：一次 tool call 的全过程

设定：用户问"上海现在多少度？"，我给模型一个工具 `get_weather`。

**Step 0 — 我在请求里声明工具（schema）：**
```json
{
  "name": "get_weather",
  "description": "查询某城市当前天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "unit": {"type": "string", "enum": ["C", "F"]}
    },
    "required": ["city"]
  }
}
```

**Step 1 — 模型第一次前向传播的输出**（注意：它没答天气，而是发起工具调用）：
```json
{"tool_calls": [{"name": "get_weather", "arguments": {"city": "上海", "unit": "C"}}]}
```
这里 `enum: ["C","F"]` 在受约束解码下意味着：当模型生成到 `unit` 的值时，**掩码只允许 `"C"` 或 `"F"` 对应的 token**，它**不可能**吐出 `"摄氏度"`。这就是 schema 直接压进解码过程的地方。

**Step 2 — 我的代码接手执行**（模型看不到这一步）：
```python
result = get_weather(city="上海", unit="C")   # → {"temp": 31, "unit": "C", "desc": "多云"}
```

**Step 3 — 我把函数结果拼回上下文，再请求一次**。模型这次基于真实数据，生成给人看的自然语言：
> "上海现在 31°C，多云。"

**全过程的关键洞察**：模型前后被调用了**两次**，中间那次真正"干活"的是我的 Python。模型的两次角色完全不同——第一次当"填表员"（结构化输出），第二次当"解说员"（自然语言）。看懂这个两段式，你就看懂了所有 Agent 框架的最小内核。

---

## 二、最新动态 / 论文速览

> （已用 WebSearch 交叉核实标题与 arXiv 编号，均为真实可点开来源。）

1. **Thinking Before Constraining: A Unified Decoding Framework for Large Language Models** · arXiv:2601.07525 · 2026-01
   - 为什么重要：正面回击本课讲的"格式税"。提出 **In-Writing**：让模型**先自由推理**，直到生成一个"触发 token"才切换进受约束解码，从而"既保留 CoT 的推理表达力、又保留硬约束的格式保证"，在分类与推理任务上超过 SOTA。
   - 与今日主题的关系：这就是第三节「结构 vs. 推理张力」的最新解法，等于把我建议的"schema 里先留 reasoning 字段"上升成了一个解码框架。
   - 链接：https://arxiv.org/abs/2601.07525

2. **ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents** · arXiv:2607.04686 · 2026-07-06（约 2 天前）
   - 为什么重要：指出"最终答案对不对"这种聚合指标会**掩盖工具用错在哪一步**。它用配对设计把失败拆成四类：**Tool-Skip（该调不调）/ Result-Ignore（调了却无视返回）/ Output-Fabrication（编造结果）/ Unnecessary-Tool-Use（不该调乱调）**。19 个模型里最好也才 86.33% 干净工具使用率——远未饱和。
   - 与今日主题的关系：正是第三节"单工具→多工具"里失败谱系的实证版，直接接上 roadmap A 的「Agent 基础」与「LLM 评测」。
   - 链接：https://arxiv.org/html/2607.04686v1

3. **JSONSchemaBench + XGrammar（结构化生成的工程基准与引擎）** · 社区基准 / 引擎 · 2024–2025（经典，非近期）
   - 为什么重要：JSONSchemaBench 用约 1 万个真实世界 schema 横评六个约束框架；XGrammar 通过"词表分区 + 自适应掩码缓存"把受约束解码做到近乎零开销，现已是 vLLM/SGLang/TensorRT-LLM 的默认后端。
   - 与今日主题的关系：是第二节"掩码计算慢不慢"的工程答案，也是你上线结构化输出时会实际接触到的底层。

---

## 三、🔁 旧知回顾（间隔重复日 · 第 3 个学习日）

- **Q1**（07-06 提示工程）：Chain-of-Thought 为什么能提升表现？请用"token 换算力"这个视角一句话解释。
- **Q2**（07-06 Transformer）：自注意力公式里那个 $\sqrt{d}$ 是干什么用的？去掉会怎样？
- **Q3**（07-07 测试时计算）：并行式测试时计算（采 N 条再用验证器选优）中，为什么"坏验证器 + 大 N"反而会让效果变差？

<details><summary>点开看答案</summary>

- **A1**：自回归模型每个 token 的计算量是固定的，答案的"思考"被压在有限步里。CoT 让模型把中间推理**显式写成更多 token**，相当于把难题拆进更长的计算路径——**用生成更多 token 来换取更多有效计算量（test-time compute）**，于是难题的正确率上升。
- **A2**：$Q\cdot K^\top$ 的点积方差随维度 $d$ 线性增长，数值过大会把 softmax 推到饱和区（梯度接近 0、注意力退化成 one-hot）。除以 $\sqrt{d}$ 把点积方差**归一到约 1**，让 softmax 停在梯度健康的区间。去掉它，高维下训练不稳、注意力过早尖锐化。
- **A3**：并行式方案的上限受验证器质量制约。N 越大，候选里"看着对但实则错"的样本越多，坏验证器会**系统性地把这些错样本选出来**（reward hacking 的雏形），于是采样越多、被误选的概率越高——coverage 上升但 selection 变差，净效果反转。

</details>

---

## 四、🎯 留给明天的钩子

- **今天点到但没展开的**：
  - 受约束解码里"JSON Schema → CFG → 下推自动机/FSM"的**编译细节**，以及为什么正则不够、必须上 CFG；
  - "格式税"到底掉多少分、在哪类任务上最严重（需要看具体 benchmark 数字）；
  - 多工具场景下的**工具选择/路由**机制（该不该调、调哪个）。
- **建议下次深入**：轨道 B「预训练目标与 Scaling Laws」或「梯度下降/反向传播」补地基（下一轮 A 再回到 roadmap A 的「结构化输出→Function Calling→Tool Use」深挖到 **L3**：原生 API vs. 手搓、多工具编排、以及把 ToolFailBench 的四类失败当作评测维度）。

---

## 五、📚 延伸阅读（可选）

- OpenAI 官方文档：Structured Outputs / `strict` mode 与 Function Calling（看它如何把 JSON Schema 落到 constrained decoding）。
- XGrammar 论文 / 博客：受约束解码如何做到近零开销（词表分区、掩码缓存）。
- 《Thinking Before Constraining》(arXiv:2601.07525)：读方法的"触发 token"策略如何消除过早触发。

---
id: lesson-0013
source:
  file: lessons/2026-07-17.md
  section: 1
  hash: sha256:48409f105f1e545dc5aab14b7eefc53e2c9df97d4bdbf49cb086f6dc34536f0c
lesson: 13
date: 2026-07-17
track: A
depth: L3
titleZh: 结构化输出与 Function Calling / Tool Use
titleEn: Constrained Decoding, Format Tax, and Tool Routing
summaryZh: 复访升级day4 L2→L3(编译细节+格式税实测+路由)。JSON嵌套非正则语言,须CFG/下推自动机(PDA)才能"记住"括号深度,而非纯FSM;编译管线=schema→一次性编译(20-50ms)出grammar/PDA→逐token按当前state把非法token logit置-∞;xgrammar靠vocab partitioning+adaptive mask caching把per-token开销压到近0,缓存命中后约束解码有时反而比无约束更快。格式税机制说清:grammar mask会在某些解码位置直接吃掉"该调用工具"的token,模型被迫塌陷成安全默认值——真实案例schema合法率99.4%但11%语义已跑偏,schema只保证格式不保证质量。MCP规模工具膨胀(50个工具=2-4万token固定prefill+选错率上升)把"塞哪些工具schema"变成day7/10的检索问题翻版,history-aware路由器(依赖图+多轮轨迹)是新解法
summaryEn: Constrained decoding compiles nested schemas into grammar-aware state machines that mask invalid tokens at each step. This lesson examines efficient mask caching, the format tax that can preserve valid JSON while distorting tool semantics, and history-aware tool routing as a retrieval problem when large tool catalogs overwhelm context and selection quality.
slug: constrained-decoding-format-tax-tool-routing
tags:
  - constrained-decoding
  - structured-output
  - tool-routing
  - mcp
sourceStatus: unreviewed
sourceStatusHash: sha256:48409f105f1e545dc5aab14b7eefc53e2c9df97d4bdbf49cb086f6dc34536f0c
metadataStatus: current
metadataSourceHash: sha256:48409f105f1e545dc5aab14b7eefc53e2c9df97d4bdbf49cb086f6dc34536f0c
featured: true
---
# 📅 2026-07-17 · 讲13 · 轨道A · 结构化输出与 Function Calling / Tool Use · 深度 L3

> **复访升级**：上次 L2（2026-07-06，day4）→ 今天 L3。切入角度：JSON Schema → CFG → FSM/PDA 的编译细节、"格式税"（constraint tax）的最新实证、以及工具目录膨胀到 MCP 规模后的路由/评测问题。

> **TL;DR**：受约束解码用"逐 token 硬掩码"把输出合法性焊死到 100%，但这层保证只覆盖"格式对不对"，不覆盖"内容对不对"——2026 年最新实证发现，schema 约束会在某些解码位置直接吃掉本该触发"调用工具"意图的 token，让模型悄悄塌陷成一个"安全默认值"；当工具目录从几个膨胀到 MCP 生态的成百上千个时，这个问题又和"该往 context 里塞哪些工具 schema"的检索问题（RAG 的老朋友）绞在一起。

## 一、核心概念精讲（L3）

### 1. 先回忆 L2 的骨架

Day4 讲过：模型不"执行"函数，只生成符合 schema 的结构化文本（JSON），真正执行的是 host 代码；可靠性靠**受约束解码**——每一步用 schema 编译出的语法把非法 token 的 logit 置 `-∞`，保证结构合法（≠语义正确）；调用链是**两段式**：模型先当"填表员"吐出 `tool_call`，host 执行后把结果塞回 context，模型再当"解说员"生成自然语言回答。今天要打开"编译"这一步的黑箱，并看清"合法≠正确"这道口子到底有多大。

### 2. JSON Schema → CFG → FSM/PDA：编译管线到底做了什么

先问一个第一性原理问题：为什么不能像 few-shot 那样"提示模型输出 JSON"就完事？因为提示只是**软约束**（soft constraint）——模型仍可能吐出多一个逗号、少一个引号，靠后处理重试代价高且不保真。受约束解码把它变成**硬约束**：数学上保证输出落在 schema 定义的语言里。

关键的技术细节：JSON 的嵌套结构（`{}` 里套 `[]` 再套 `{}`）**不是正则语言**（regular language），因为正则/FSM 只有有限个状态、**没有"记住嵌套了多深"的记忆**——一台纯 FSM 无法数清"现在欠着几个 `}` 没闭合"。真正描述嵌套结构需要**上下文无关文法**（CFG, Context-Free Grammar），工程实现上则用**下推自动机**（PDA, Pushdown Automaton）落地：PDA = FSM + 一个**栈**，栈就是那份"记忆"。

具体到 JSON，栈机制极简单——**见开括号压栈、见闭括号弹栈**：

```
输入串       {   "a"  :  [   1  ,  2   ]      }
栈动作      push{      push[         pop[   pop{
栈内容      [{]  [{]  [{]  [{,[] [{,[]...    [{]    []（空）
                                          └ 栈非空时，"}" 被掩码禁止（还欠一个 ]）
                                            栈顶是 { 时，才允许 "}"
```

这条"栈空才允许收尾"的规则，正是纯正则做不到、必须上 PDA 的根本原因：**合法 token 集合不只取决于当前 state，还取决于栈顶是什么**。理解这一点，下面"格式税"为什么会掐掉某些 token 才讲得通——掩码是 state × 栈 共同决定的，模型的"意图"在这套硬规则面前没有否决权。

编译管线大致是：

```
JSON Schema
  {"type":"object",
   "properties":{"city":{"type":"string"},
                 "unit":{"enum":["c","f"]}}}
        │  一次性编译（每个 schema 只需 20-50ms，之后走缓存）
        ▼
  Grammar / PDA（能记住"我在第几层花括号里"）
        │
        ▼
  逐 token 状态游走：
   state0 "{"           → 合法 token 集合 = {"\"city\"", "\"unit\"", ...}
   state1 已见 "city":   → 合法集合 = {字符串开头 token, ...}
   state2 已见 "unit":   → 合法集合 = {"\"c\"", "\"f\""}（enum 收紧到 2 个 token！）
        │  每一步：把当前 state 下"非法"的 token 的 logit 置 -∞
        ▼
   softmax 之后非法 token 概率恒为 0 → 100% 语法合法
```

工程上真正的挑战不是"要不要做掩码"，而是"每一步扫一遍几万到几十万的词表算掩码"太贵。xgrammar（vLLM 0.4+ / SGLang 默认后端）的核心加速技巧是**vocab partitioning + adaptive token-mask caching**：提前按"哪些 token 在哪些 state 下合法"做好分区索引，同一 schema 复用时直接查表，把 per-token 掩码开销压到接近 0；只有第一次遇到新 schema 才要付编译的一次性成本。这也是为什么 2026 年的一手数据显示"约束解码常常比无约束生成更快"——它省掉的是"生成非法 JSON 后重试"的隐藏代价，反直觉但符合工程实际。

### 3. 格式税实测：合法 ≠ 正确

Day4 已经点出"格式税"这个张力——过早锁定 JSON 结构会挤压模型的推理空间。今天要把"为什么会掐掉工具调用"这句话**推给你看**，而不是转述结论。

**最小推导：一个 token 是怎么被"结构性抢占"的。** 假设 host 要求模型输出的 schema 是"一个 JSON 数组，每个元素是 `{"tool": <enum>, "args": {...}}`，`tool` 字段的 enum 是 `["get_weather", "answer_directly"]`"。现在模型内部其实拿不定主意——它的原始 logits 分布（softmax 前）大致是：

```
状态：已生成 [{"tool": ，PDA 栈 = [ 数组[ , 对象{ ]，当前 state 期待一个 enum 字符串值
                              ┌─ 原始 logit（模型的"真实意图"，未受约束）
  token  "get_weather"        │   2.1   ← 模型其实略偏向"先别急着调，我想想"
  token  "answer_directly"    │   2.4   ← 最高
  token  "\n让我先分析一下"    │   3.8   ← 模型最想生成的：一段自由推理（CoT）
  token  "{"                  │   1.0
  ...
```

模型**最想**生成的是那句"让我先分析一下"（logit 3.8）——这正是 day4/CoT 说的"用 token 换算力"。但当前 PDA state 期待的是一个 **enum 字符串值**，合法集合只有 `{"get_weather", "answer_directly"}`。掩码一施加：

```
  m["\n让我先分析一下"] = -∞   → softmax 后概率 = 0    ← 推理链被物理掐断
  m["{"]              = -∞   → 0
  只剩 "get_weather"(2.1) 与 "answer_directly"(2.4) 参与归一化
  → P(answer_directly) = e^2.4 / (e^2.1 + e^2.4) ≈ 0.57
  → P(get_weather)     ≈ 0.43
```

看清楚发生了什么：模型**并非"不想"调用工具**，而是它本想先写一段推理再决定，可这个念头对应的 token 被 state×栈 的硬掩码判了死刑；被迫在"不经思考的两个选项"里二选一时，它滑向了那个更省事、更"安全"的 `answer_directly`。**格式税的机制真身，就是这一步 `-∞` 掩码把"先思考"的 token 抹零，逼模型在未完成推理的状态下做结构化决策**——schema 越是要求"第一个 token 就得进 JSON 结构"，这个绞杀就越致命。这也是为什么"在 schema 里前置一个 `reasoning: string` 字段"能救场：它等于在 PDA 里合法地开一段自由文本状态，把那句 logit 3.8 的推理放回牌桌。

一个业界案例（来自 2026 一篇工程评测博客，非同行评审、**数字未独立核实，仅作直觉锚点**）：某抽取流水线上线时用 OpenAI strict mode，schema 合法率报称 99.4%——每条 JSON 都能解析、每个字段约束都满足；三周后团队才发现，模型在约 11% 的工单上悄悄塌陷成了一个"安全默认值"来满足语法，schema 校验完全没能抓住这个质量问题。数字真假不论，它指向的**定性结论是硬的、且与上面的掩码推导同源**：结构化输出模式保证的是 schema，不是质量。这是 L3 工程视角必须刻进直觉的一条准则：**把"JSON valid rate"当唯一评测指标是危险的，必须叠加语义正确率的抽检。**

### 4. 工具膨胀到 MCP 规模：路由问题 = RAG 问题的翻版

Day4 时工具目录可能只有几个函数，直接把全部 schema 塞进 prompt 没问题。但 MCP（Model Context Protocol）生态把"工具"变成了可插拔的成百上千个服务，这时两个成本同时爆炸：

- **Prefill 成本**：每个工具的完整 schema 大约占 400-800 token，塞 50 个工具就是 2-4 万 token 的固定开销，且每次请求都要重新算一遍（除非有 schema 级 KV cache）。
- **选错率上升**：候选工具越多，模型在正确工具和"看起来也差不多"的工具之间选错的概率越高——这本质上就是 day7/day10 讲过的**检索问题**：与其把所有工具都塞进 context 让模型"过目所有选项"，不如先做一次**工具检索**（tool retrieval），只把 Top-K 相关工具的 schema 送进 prompt，日后可以直接复用 day10 的 hybrid + rerank 思路。这条线正是当前研究方向之一：history-aware 的工具路由器（如 ToolACE-MCP）用依赖图 + 多轮轨迹训练，专门解决"MCP 规模下该检索哪些工具"这个新版 RAG 问题。

### 5. 常见误区 / 易错点

- **误区一**：schema 校验通过 = 输出正确。实际只保证格式合法，语义可能已经塌陷成"安全默认值"（见上面案例）。
- **误区二**：约束解码一定拖慢生成。现代 xgrammar 在 schema 缓存命中后开销趋近于 0，某些场景甚至比无约束生成更快（省去了格式错误重试的代价）。
- **误区三**：工具越多、直接全塞进 context 越"周全"。这会同时推高 prefill 成本和选错率，MCP 规模下必须前置一层工具检索/路由。

### 6. 完整走一遍：一次真实的多工具调用链

用户问："帮我查一下北京今天的天气，然后把结果发到我的 Slack #weather 频道。" 工具目录里有 `get_weather(city)` 和 `send_slack_message(channel, text)`，其中 `channel` 字段的 schema 是 `{"enum": ["#weather", "#general", "#random"]}`（频道白名单）。

1. **Prompt 构造**：host 把两个工具的 JSON Schema 编入 tool 字段，一起送进模型；此时两个 schema 已各自编译成对应的 PDA（若之前调用过，直接命中缓存，省掉 20-50ms 编译）。
2. **第一次解码（get_weather）**：解码器逐 token 应用 `get_weather` schema 的掩码。走到 `{"name":"get_weather","arguments":{"city":` 之后，PDA 栈 = `[数组?, 对象{, 对象{]`，当前 state 期待一个字符串值 → 合法集合被收紧成"字符串起始 token"（`"` 及其后的中文字 token），`}`、`,`、数字 token 全被置 `-∞`。模型据此吐出 `{"name":"get_weather","arguments":{"city":"北京"}}`。此处也埋着格式税：若模型本想先写"这问题得先查天气"，那个自由文本 token 在起始 state 就被掩掉了。
3. **Host 执行**：拿到合法 JSON，调用真实天气 API，得到"北京今天 32°C，晴"。
4. **结果回填**：host 把执行结果作为一条 tool 消息塞回 context。
5. **第二次解码（send_slack_message，看 channel 字段的掩码）**：模型判断任务没完，发起第二个 tool_call。到 `"channel":` 的值位置时，enum 掩码只放行 3 个候选。**关键演示**——假设模型对频道拿不准，原始 logits 是：

   ```
   token  "#weather"    logit 1.9
   token  "#general"    logit 2.0   ← 模型原始最高（它其实想瞎猜个常用频道）
   token  "#random"     logit 0.3
   token  "#dev"        logit 2.6   ← 用户根本没这个频道 / 不在 enum 里
   ```

   `"#dev"` 虽然原始 logit 最高，但不在 enum 合法集合里 → 掩码置 `-∞`、概率归零。enum 掩码在这里是**正面作用**：它物理上杜绝了"发到不存在的频道"这种幻觉。但它救不了 `#weather` vs `#general` 的语义抉择——那要靠上文"发到 #weather"的信息把 logit 拉对（这里 `#weather` 的 1.9 得益于上文提示，最终归一化后 P(#weather)≈0.47 > P(#general)≈0.44，险胜）。**这一步同时演示了掩码的两面：结构性地灭幻觉（好），但语义正确与否仍取决于模型自身概率（掩码管不着）。**
6. **Host 执行 + 最终解说**：host 发送成功后回填结果，模型此时脱离 tool schema 约束（自由文本模式），生成："已经帮你查了北京天气（32°C，晴）并发到 #weather 频道啦。"

这条链路里，**两次进入"填表员"模式都各自触发一次独立的语法编译/掩码过程**，而"格式税"风险在每次进入 JSON 模式的第一个 token（决定要不要调用、调用哪个工具）上最集中——这也是为什么 2026 年的研究开始专门测"工具调用抑制率"，而不只是测 schema 合法率。

## 二、最新动态 / 论文速览

⚠️ 已联网并交叉核对来源标题/arXiv 编号自洽性；因当前网络环境的沙盒策略阻止直接二次访问 arXiv 原文页面做逐字核验，以下条目基于搜索引擎返回的标题、摘要片段与多源交叉印证，未做最终原文抓取核实，请视为"大概率属实，细节以原文为准"。

1. **Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints** · arXiv 2606.25605（编号前缀 2606 对应 2026-06，与"近30天"窗口自洽） —— 这篇直接实证了本讲的核心机制：JSON Schema 被编译成 grammar token mask 后，会让部分工具调用 token 在解码时变得不可达，从"实现层"解释了为什么开权重模型在强 schema 约束下会抑制工具调用。与今天"格式税"小节几乎是一一对应的最新证据。
2. **PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems** · arXiv 2606.22388（前缀 2606 对应 2026-06） —— 针对"工具目录膨胀到大规模生态后，模型长程规划/多工具编排能力"的新基准，呼应今天"工具路由=新版RAG"这一小节；说明学界已经把"选对工具"和"长程规划"当成两个要分别评测的能力维度。
3. **Berkeley Function Calling Leaderboard v4（BFCL v4）** · 2026年4月发布（略超30天窗口，但作为该领域最主要的评测框架版本更新一并列出） —— 评测重心从"单次调用对不对"转向 Agentic（多步规划+执行，占40%权重）与 Multi-Turn（跨多轮对话保持调用上下文，占30%权重），说明行业共识正从"格式合法率"转向"端到端任务完成率"，与今天"合法≠正确"的结论方向一致。

## 三、🎯 留给明天的钩子

- 今天点到没展开的：ToolACE-MCP 的 history-aware 路由器具体怎么用依赖图训练、MCP 规模下的工具 schema KV cache 复用方案。
- 建议下次深入：A 轨新主题「Agent 基础：ReAct、规划、工具编排」——今天讲的"填表员→host执行→解说员"两段式，正是 ReAct 循环里 Action/Observation 的雏形，顺理成章地把 Function Calling 升级成完整的 Agent 循环；或者 A 轨「LLM 评测：LLM-as-judge」，用今天"合法≠正确"的教训去设计"语义正确率"这类更难量化的评测维度。

## 四、📚 延伸阅读

- BFCL v4 官方排行榜与评测维度说明（Agentic / Multi-Turn 权重拆分）
- xgrammar 项目文档：vocab partitioning 与 adaptive token-mask caching 的实现细节
- day4 讲义（2026-07-06）：受约束解码的 L2 基础与两段式调用链

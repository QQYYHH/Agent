# OpenCode

> https://github.com/anomalyco/opencode
>
> 源码分析基于 commit hash: 9fc182baf23536619601dcf43be7d5539b8ad92e

**OpenCode** 是一个开源的 AI Coding Agent 项目，它通过 CLI 和 Web UI 为用户提供代码编辑、调试和软件工程任务的自动化帮助。核心基于 TypeScript + Bun 运行时构建。

## 整体架构

我只关注 Agent 部分的执行逻辑，整体架构如下图所示。我认为 opencode 的核心是将各种上层逻辑（如Plan、创建SubAgent、使用Skills等）封装成工具，显式调用，或让 LLM 决定调用时机。Agent 的工作模式沿用常见的 `ReAct` 范式。

Opencode 允许 Agent 在运行的过程中向用户提问，以获取人类反馈，让工作结果更加准确。

同时会额外生成一些提示词，以便于让 LLM 在某些情况下能继续执行下去。比如在压缩完上下文后，opencode 会额外添加 `Continue if you have next steps` ，以使得 LLM 在理解完压缩信息之后继续完成用户指定的任务。

<img width="3498" height="2082" alt="opencode_framework" src="https://github.com/user-attachments/assets/13c5e5ef-1adf-4bf7-a3fa-2e083e00494d" />




---

Opencode 依赖的核心生态如下：

```yaml
LLM 框架：
- ai: Vercel AI SDK (统一接口)
- @ai-sdk/*: 25+ 模型提供商

协议标准：
- @modelcontextprotocol/sdk: MCP 1.25.2
- @agentclientprotocol/sdk: ACP 0.12.0

工具库：
- zod: 类型验证
- remeda: 函数式编程
- ripgrep: 代码搜索
- @parcel/watcher: 文件监听

Web框架：
- hono: 轻量级 HTTP 框架
- @octokit/*: GitHub API

CLI框架：
- yargs: 命令行参数解析
- @clack/prompts: 交互式提示

其他：
- @opencode-ai/*: 自有工具库
- solid.js: Web TUI UI框架
```



## Agent系统

```yaml
// src/agent/agent.ts
- name: "build" | "plan" | "general" | "explore" | "compaction" | "summary" | "title"
- mode: "subagent" | "primary" | "all"  // 运行模式
- model: {modelID, providerID}          // LLM配置
- permission: PermissionNext.Ruleset    // 权限规则
- temperature/topP: 参数调优
- prompt: 自定义系统提示词
```

各类 Agent 的功能总结如下：

| Agent 名称     | 模式 (mode) | 描述/功能                                                    | 隐藏 | 原生 |
| -------------- | ----------- | ------------------------------------------------------------ | ---- | ---- |
| **build**      | `primary`   | 主要构建代理，允许提问和进入计划模式                         | 否   | 是   |
| **plan**       | `primary`   | 计划代理，允许提问和退出计划模式，只能编辑 `.opencode/plans/*.md` 文件 | 否   | 是   |
| **general**    | `subagent`  | 通用代理，用于研究复杂问题和执行多步骤任务，可并行执行多个工作单元 | 否   | 是   |
| **explore**    | `subagent`  | 快速探索代码库的代理，专门用于按模式查找文件、搜索代码关键字、回答代码库相关问题。支持三种彻底程度："quick"（快速搜索）、"medium"（中等探索）、"very thorough"（全面分析） | 否   | 是   |
| **compaction** | `primary`   | 压缩代理，用于压缩/精简内容，禁用所有工具权限                | 是   | 是   |
| **title**      | `primary`   | 标题生成代理，temperature=0.5，禁用所有工具权限              | 是   | 是   |
| **summary**    | `primary`   | 摘要生成代理，禁用所有工具权限                               | 是   | 是   |

Opencode 支持的后端 LLM 模型如下：

```yaml
// src/provider/provider.ts
BUNDLED_PROVIDERS:
- @ai-sdk/anthropic (Claude系列)
- @ai-sdk/openai (GPT系列)
- @ai-sdk/google (Gemini)
- @ai-sdk/azure (Azure OpenAI)
- @ai-sdk/groq, @ai-sdk/mistral, @ai-sdk/cohere
- @ai-sdk/amazon-bedrock, @ai-sdk/google-vertex
- ... 及15+其他提供商

// 模型能力：
- tool_call: 工具调用能力
- reasoning: 推理能力
- attachment: 文件附件支持
- knowledge: 知识库日期
```



### ReAct 架构实现

Opencode 主要采用简单有效的 ReAct 范式实现 Agent. 核心实现位于 `src/session/ptompt.ts` 中的 `loop` 函数。

#### Loop 函数总体结构

Loop 函数总体结构，核心就是下面代码中的无限循环。

```typescript
// src/session/prompt.ts (lines 258-637)
export const loop = fn(Identifier.schema("session"), async (sessionID) => {
  // 初始化
  const abort = start(sessionID)
  if (!abort) {
    // 如果已有相同 sessionID 的 loop 正在运行，加入队列
    return new Promise<MessageV2.WithParts>((resolve, reject) => {
      const callbacks = state()[sessionID].callbacks
      callbacks.push({ resolve, reject })
    })
  }

  using _ = defer(() => cancel(sessionID))  // 清理资源

  let step = 0
  const session = await Session.get(sessionID)
  
  // ========== 主循环开始 ==========
  while (true) {
    // A. 消息历史分析与状态检查
    // B. 任务处理 (subtask/compaction)
    // C. 正常处理 (LLM 推理)
    // D. 循环继续条件判断
  }
  
  // ========== 返回最后一条消息 ==========
  return final message
})
```

#### 消息历史分析：

- 首先过滤掉已经被压缩的消息，`MessageV2.filterCompacted` 实现，思路很简单，从后往前搜索历史消息，直到第一次碰到被标记为 `compaction` 的消息。(`part.type == "compaction"`)

- 找到最新的用户消息`lastUser`、最新的AI回复消息`lastAssistant`。这些消息主要用于判断任务是否完成。如果最新的AI消息被标记为完成（类似于`stop`）且没有更新的用户需求，则表示当前任务已完成，可以退出循环。

  ```typescript
  if (
    lastAssistant?.finish &&
    !["tool-calls", "unknown"].includes(lastAssistant.finish) &&
    lastUser.id < lastAssistant.id
  ) {
    log.info("exiting loop", { sessionID })
    break  // ✅ 循环应该退出
  }
  ```

- 值得注意的是消息历史格式大概如下所示，基本是 `U1, A1, U2, A2, U3, A3, ...` 这种`UA`交错的模式。

  ```text
  ├────────────────────────────────────────────────────────────────┤
  │ [user]:     "帮我创建一个 React 项目"                            │
  │ [assistant]: "好的，我来帮你创建..." + [tool results]             │
  │ [user]:     "添加 TypeScript 支持"                              │
  │ [assistant]: "已添加 TypeScript..." + [tool results]            │
  │ ...                                                            │
  │ [user]:     "优化性能"                                           │
  │ [assistant]: "已完成优化..."                                     │
  ```

  【💡】这就存在一种问题，LLM 认为某一步 `A_k`  所代表的子任务已经完成了，就会设置`A_k.finish = "stop"`，但此时整体任务还没有完成。如果没有最新的用户需求，按照上面的逻辑就会退出循环，导致任务执行不完整。为了解决这个问题，Opencode 会`合成新的用户消息`，来促使整个循环继续推进，直到整体任务完成。同时，下面的合成消息可以为多轮对话中，排队的新用户消息添加提醒。

  ```typescript
  // 如果显式上一条AI消息代表的任务已完成，为了避免局部子任务完成就退出的情况，会额外合成用户消息
  // 确保继续推进后续任务的执行。
  // 如果多轮对话，为排队的新用户消息添加提醒
  if (step > 1 && lastAssistant.finish) {
      for (const msg of msgs) {
          if (msg.info.role !== "user" || msg.info.id <= lastAssistant.id) continue
          for (const part of msg.parts) {
              if (part.type !== "text" || part.ignored || part.synthetic) continue
              if (!part.text.trim()) continue
              part.text = [
                  "<system-reminder>",
                  "The user sent the following message:",
                  part.text,
                  "",
                  "Please address this message and continue with your tasks.",
                  "</system-reminder>",
              ].join("\n")
          }
      }
  }
  ```

  

- 收集特殊类型的任务，如`CompactPart` 和 `SubtaskPart`

```typescript
while (true) {

  // 获取所有消息（过滤已压缩的）
  let msgs = await MessageV2.filterCompacted(MessageV2.stream(sessionID))

  // 逐层分析消息，从后往前，从新往旧
  let lastUser: MessageV2.User | undefined          // 最后一条用户消息
  let lastAssistant: MessageV2.Assistant | undefined  // 最后一条助手消息
  let lastFinished: MessageV2.Assistant | undefined  // 最后一条完成的助手消息
  let tasks: (MessageV2.CompactionPart | MessageV2.SubtaskPart)[] = []

  for (let i = msgs.length - 1; i >= 0; i--) {
    const msg = msgs[i]
    if (!lastUser && msg.info.role === "user") 
      lastUser = msg.info as MessageV2.User
    
    if (!lastAssistant && msg.info.role === "assistant") 
      lastAssistant = msg.info as MessageV2.Assistant
    
    if (!lastFinished && msg.info.role === "assistant" && msg.info.finish)
      lastFinished = msg.info as MessageV2.Assistant
    
    if (lastUser && lastFinished) break  // 找到关键消息，提前退出

    // 收集待处理任务
    const task = msg.parts.filter((part) => 
      part.type === "compaction" || part.type === "subtask"
    )
    if (task && !lastFinished) {
      tasks.push(...task)
    }
  }
```

#### 会话标题生成

判断是否是第一次执行执行用户需求，如果是，则为此次会话生成标题/摘要。摘要由 `ensureTitle` 生成，执行逻辑比较简单。把标题生成当作是一个`subtask`，然后创建专门负责生成标题的`subAgent` 执行该摘要任务。`agent = await Agent.get("title")`

```typescript
step++
if (step === 1)
    ensureTitle({
        session,
        modelID: lastUser.model.modelID,
        providerID: lastUser.model.providerID,
        history: msgs,
    })
```

#### 特殊任务处理

主要处理 `subtask` 和 `compaction` 任务。

1. 优先执行 `subtask`，这类任务通常是被拆解出来的子任务，或者是用户通过`/command`指定的子任务，需要 LLM 调用 `TaskTool` 来创建 `subAgent` 完成。执行流程如下：

```text
Subtask part detected
  ↓
Create Assistant Message (mode=subagent) // 需要子代理独立执行
  ↓
Create Tool Part (status=running)
  ↓
Plugin.trigger("tool.execute.before")
  ↓
TaskTool.execute(args, context)
  │
  ├─ 调用子代理 (e.g., "explore" 代理)
  ├─ 子代理独立运行
  └─ 返回结果
  ↓
Plugin.trigger("tool.execute.after")
  ↓
Update Tool Part
  ├─ status = "completed" ✓
  ├─ output = 结果
  └─ attachments = 文件
  ↓
Add Synthetic User Message (if task.command)
  ↓
continue → 回到 while (true)
```

相关代码如下：

```typescript
if (task?.type === "subtask") {
  const taskTool = await TaskTool.init() // 获取 taskTool 工具，以便于创建子Agent执行当前子任务。
  
  // 创建新的 Assistant 消息来执行子任务
  const assistantMessage = (await Session.updateMessage({
    id: Identifier.ascending("message"),
    role: "assistant",
    parentID: lastUser.id,
    sessionID,
    mode: task.agent,        // 子代理类型
    agent: task.agent,
    // ... 其他字段
  })) as MessageV2.Assistant

  // 创建工具调用 Part
  let part = (await Session.updatePart({
    type: "tool",
    tool: TaskTool.id,
    callID: ulid(),
    state: {
      status: "running",      // 立即标记为运行中
      input: {
        prompt: task.prompt,
        description: task.description,
        subagent_type: task.agent,
        command: task.command,
      },
      time: { start: Date.now() },
    },
  })) as MessageV2.ToolPart

  // 执行前钩子
  await Plugin.trigger("tool.execute.before", {...}, { args: taskArgs })

  // 执行子任务
  const result = await taskTool.execute(taskArgs, taskCtx).catch((error) => {
    executionError = error
    return undefined
  })

  // 执行后钩子
  await Plugin.trigger("tool.execute.after", {...}, result)

  // 根据结果更新部分状态
  if (result && part.state.status === "running") {
    await Session.updatePart({
      ...part,
      state: {
        status: "completed",
        title: result.title,
        output: result.output,
        // ...
      },
    })
  }

  if (!result) {
    await Session.updatePart({
      ...part,
      state: {
        status: "error",
        error: executionError?.message,
      },
    })
  }

  // 如果是用户通过 /command 主动调用的内置任务（例如/review），则添加合成用户消息
  if (task.command) {
    const summaryUserMsg: MessageV2.User = {
      id: Identifier.ascending("message"),
      sessionID,
      role: "user",
      agent: lastUser.agent,
      model: lastUser.model,
      time: { created: Date.now() },
    }
    await Session.updateMessage(summaryUserMsg)
    await Session.updatePart({
      messageID: summaryUserMsg.id,
      type: "text",
      text: "Summarize the task tool output above and continue with your task.",
      synthetic: true,
    })
  }

  continue  // 返回 while 循环顶部，重新分析
}
```



2. 次优先处理 `compaction` 类型的任务。每次 Loop 时，都会判断上下文是否超过阈值，如需压缩，就显式添加一个 `compaction` 类型的任务。具体的处理过程见`Context 上下文管理` 章节。

```typescript
// 处理压缩任务
if (task?.type === "compaction") {
  const result = await SessionCompaction.process({
    messages: msgs,
    parentID: lastUser.id,
    abort,
    sessionID,
    auto: task.auto,
  })
  if (result === "stop") break  // 压缩失败，退出 loop
  continue  // 压缩成功，继续 loop
}

// 当token超过阈值时，显式添加 compact 类型的任务
if (
  lastFinished &&
  lastFinished.summary !== true &&
  (await SessionCompaction.isOverflow({ tokens: lastFinished.tokens, model }))
) {
  // 自动触发压缩
  await SessionCompaction.create({
    sessionID,
    agent: lastUser.agent,
    model: lastUser.model,
    auto: true,  // 自动模式
  })
  continue
}
```

触发条件：

```text
当满足：
  ✓ 上一条 Assistant 消息已完成 (lastFinished exists)
  ✓ 还未被压缩过 (summary !== true)
  ✓ Token 使用超过模型上下文
  
则：
  创建压缩任务 → 返回 while 顶部
```

#### 正常 LLM 处理流程

具体流程如下：

1. 首先通过 `insertReminders` 插入消息提醒，该提醒主要和 `Plan` 模式的转换有关。有两种功能：首先是在刚进入 `Plan` 阶段，注入 `Plan` 相关的提示词，告知 LLM 准备开始规划了；其次，Agent 完成规划之后，会调用 `plan_exit` 工具来向用户确定是否接受此次规划，一旦接受，Agent 就从 `Plan` 模式 转换到 `Build` 模式，执行实际工作。`insertReminders` 就是为了合成系统消息，具有承上启下的作用，让 LLM 知道规划完成，现在开始执行具体的任务了。

```typescript
// 插入提醒消息（如果多轮对话）
msgs = await insertReminders({
  messages: msgs,
  agent,
  session,
})

async function insertReminders(input: { messages: MessageV2.WithParts[]; agent: Agent.Info; session: Session.Info }) {
    // 寻找最新的AI 消息
    const assistantMessage = input.messages.findLast((msg) => msg.info.role === "assistant")
    
    // 之前不是 plan 模式，现在 AI 要进行 plan 规划，需要注入 plan 提示词。
    if (input.agent.name !== "plan" && assistantMessage.info.agent === "plan") {
        userMessage.parts.push({
            id: Identifier.ascending("part"),
            messageID: userMessage.info.id,
            sessionID: userMessage.info.sessionID,
            type: "text",
            text: PROMPT_PLAN,
            synthetic: true,
        })
    }
    
    // Plan 完成，转入正常的工作阶段，注入转换提示词 BUILD_SWITCH
    const wasPlan = input.messages.some((msg) => msg.info.role === "assistant" && msg.info.agent === "plan")
    if (wasPlan && input.agent.name === "build") {
        userMessage.parts.push({
            id: Identifier.ascending("part"),
            messageID: userMessage.info.id,
            sessionID: userMessage.info.sessionID,
            type: "text",
            text: BUILD_SWITCH + "\n\n" + `A plan file exists at ${plan}. You should execute on the plan defined within it`,
            synthetic: true,
        })
    }
    return input.messages
```

2. 创建 `Assistant` 消息，代表 AI 的回复，可能包含 `TextPart`, `ToolPart` 等。

3. 解析可用的工具，`resolveTools` 会根据当前 Agent 的权限，以及`src/tools/`目录下已经注册好的工具，返回可用的工具列表。

```typescript
// 检查用户是否显式调用了特定代理 (@agent)
// 如果显式调用，则绕过后续的权限检查
const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
const bypassAgentCheck = lastUserMsg?.parts.some((p) => p.type === "agent") ?? false

// 解析可用工具
const tools = await resolveTools({
  agent,
  session,
  model,
  tools: lastUser.tools,
  processor,
  bypassAgentCheck,
})
```

4. 核心：调用 LLM 处理消息

```typescript
// 允许插件修改消息
await Plugin.trigger("experimental.chat.messages.transform", {}, { messages: sessionMessages })

// 核心：调用 LLM 处理
const result = await processor.process({
  user: lastUser,
  agent,
  abort,
  sessionID,
  system: [
    ...(await SystemPrompt.environment()),
    ...(await SystemPrompt.custom())
  ],
  messages: [
    ...MessageV2.toModelMessages(sessionMessages, model),
    // 如果是最后一步，添加 max steps 提醒
    ...(isLastStep ? [{ role: "assistant" as const, content: MAX_STEPS }] : []),
  ],
  tools,
  model,
})
```

在交给 LLM 处理消息前，要调用 `toModelMessages` 对消息做一些额外的处理。其核心逻辑有两点：

- 如果是用户消息，则根据类别额外做一些处理。当类别为 `compaction` 时，说明是 Opencode 自动合成的消息，以触发压缩操作。此时，会将这条消息替换成 `text` 类型的消息，实际内容为 `"What did we do so far?"`。当压缩操作完成之后，下一条 AI 消息就会是压缩之后的摘要。布局如下：如此操作能让消息历史压缩显得更加自然、流畅。

  ```text
  ├────────────────────────────────────────────────────────────────┤
  │ [user]:     "What did we do so far?"                           │
  │ [assistant]: "压缩后的摘要为：xxx"                                │
  │ ...                                                            │
  │ [user]:     xxx                                                │
  ```

  再比如说，类别为 `subtask` 时，表明下一条 AI 消息是 `subAgent` 执行完后汇总的结果。`toModelMessages` 会将当前消息的内容替换为：`The following tool was executed by the user`。消息布局如下：

  ```text
  ├────────────────────────────────────────────────────────────────┤
  │ [user]:     "The following tool was executed by the user"      │
  │ [assistant]: "xxx 子任务经过 xxx子代理执行完毕，
                 结果如下：执行了 XX 工具，取得了 XX结果"               │
  │ ...                                                            │
  │ [user]:     xxx                                                │
  ```

  

- 如果是 AI 消息，根据消息是否被压缩而做一些额外的删减操作。具体操作如下表格汇总所示：

| AI Part 类型                                                 | 状态                  | 转换结果                                               |
| ------------------------------------------------------------ | --------------------- | ------------------------------------------------------ |
| text | -                     | 原样保留                                               |
| step-start | -                     | 保留步骤标记                                           |
| tool | completed             | output-available + 完整输出                            |
| tool | completed + compacted | output-available + "[Old tool result content cleared]" |
| tool | error                 | output-error + 错误信息                                |
| tool | pending/running       | output-error + "[Tool execution was interrupted]"      |
| reasoning | -                     | 推理文本（Claude extended thinking）                   |



5. 根据处理结果做决策

```typescript
// 根据处理结果做决策
if (result === "stop") break
if (result === "compact") {
  await SessionCompaction.create({
    sessionID,
    agent: lastUser.agent,
    model: lastUser.model,
    auto: true,
  })
}
```



#### LLM 具体交互层

`src/session/processor.ts` 中的函数 `process` 负责处理 LLM 流式响应并管理整个会话的生命周期。它是一个异步函数，返回三种可能的结果：`"compact"`、`"stop"` 或 `"continue"`。简要流程如下所示：

```text
process 函数
├── 初始化配置
│   ├── needsCompaction = false (是否需要压缩上下文)
│   └── shouldBreak = 配置检查 (权限拒绝时是否中断)
├── 主循环 (while true)
│   ├── try块：流式处理
│   │   ├── 初始化本地状态 (currentText, reasoningMap)
│   │   ├── 调用 LLM.stream() 获取流
│   │   └── for await 处理流事件
│   ├── catch块：错误处理与重试
│   └── 循环后清理与返回
```

---



**SessionProcessor.process**

`process` 能够处理的流事件如下表所示：

| 事件类型                                                     | 作用               | 处理逻辑                                                     |
| ------------------------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| start | LLM 开始响应       | 设置会话状态为 `busy`                                        |
| reasoning-start/delta/end | 推理过程（思维链） | 创建/更新/完成 ReasoningPart |
| text-start/delta/end | 文本输出           | 创建/流式更新/完成 TextPart |
| tool-input-start/delta/end | 工具输入解析       | 创建 ToolPart 并设置为 `pending` |
| tool-call | 工具调用开始执行   | 更新状态为 `running`，检测 doom loop （是否重复调用同一个工具多次） |
| tool-result | 工具执行完成       | 更新状态为 completed |
| tool-error | 工具执行失败       | 更新状态为 error，检查是否需要阻断 |
| start-step | 步骤开始           | 创建快照，记录 StepStartPart |
| finish-step | 步骤结束           | 计算 token 使用，检查是否需要压缩                            |
| error | 流错误             | 抛出异常进入 catch 块                                        |

如果异常被 `try catch` 捕获到，会重试若干次

```typescript
const retry = SessionRetry.retryable(error)
if (retry !== undefined) {
  attempt++
  const delay = SessionRetry.delay(attempt, ...)
  // 等待后继续循环
}
```

此外，在当前这一步的任务执行完毕时（`finish-step`），会额外检查上下文是否溢出，如果溢出则返回 `compact`，在 `loop` 中做压缩处理。

```typescript
if (await SessionCompaction.isOverflow({ tokens, model })) {
  needsCompaction = true
}
```

---



**LLM.stream**

`LLM.stream` 是 `llm.ts` 中的核心函数，负责与 LLM 提供商建立流式连接并返回流式响应。它封装了 AI SDK 的 `StreamText` 函数，处理系统提示词构建、工具配置、模型参数设置等复杂逻辑。它的执行流程如下：

1. 并行获取配置

```typescript
const [language, cfg, provider, auth] = await Promise.all([
  Provider.getLanguage(input.model),   // 获取语言模型实例
  Config.get(),                         // 获取全局配置
  Provider.getProvider(input.model.providerID),  // 获取提供商信息
  Auth.get(input.model.providerID),    // 获取认证信息
])
```



2. 系统提示词构建

```text
系统提示词结构
├── Header (SystemPrompt.header)
│   └── 基础身份定义、时间、环境信息
└── Body (合并的提示词)
    ├── Agent 提示词 (input.agent.prompt)
    │   └── 或 Provider 默认提示词 (SystemPrompt.provider)
    ├── 额外系统提示词 (input.system)
    └── 用户自定义提示词 (input.user.system)
```



3. 模型参数配置，使用 `pipe` + `mergeDeep` 层层叠加配置，优先级递增。

```typescript
const options = pipe(
  base,                        // 基础配置 (small 或 normal)
  mergeDeep(input.model.options),   // 模型特定配置
  mergeDeep(input.agent.options),   // 代理特定配置
  mergeDeep(variant),               // 变体配置
)
```



4. 插件支持：

- 参数插件：

```typescript
const params = await Plugin.trigger("chat.params", context, {
  temperature: ...,
  topP: ...,
  topK: ...,
  options,
})
```

- 请求头插件

```typescript
const { headers } = await Plugin.trigger("chat.headers", context, {
  headers: {},
})
```



5. streamText 调用。核心参数如下：

| 参数                                                         | 说明                           |
| ------------------------------------------------------------ | ------------------------------ |
| `temperature` | 控制输出随机性                 |
| `topP` / `topK` | 采样参数                       |
| `tools` | 可用工具集                     |
| `maxOutputTokens` | 最大输出 token 数 (默认 32000) |
| `abortSignal` | 中止信号                       |
| `messages` | 系统消息 + 历史消息            |
| `model` | 包装后的语言模型               |

此外，还支持模型中间件：

- **消息转换中间件**：根据模型特性转换消息格式
- **推理提取中间件**：提取 `<think>` 标签中的推理内容

```typescript
model: wrapLanguageModel({
  model: language,
  middleware: [
    {
      async transformParams(args) {
        // 消息转换处理，以支持不同的 LLM Provider
        args.params.prompt = ProviderTransform.message(...)
        return args.params
      },
    },
    extractReasoningMiddleware({ tagName: "think", startWithReasoning: false }),
  ],
})
```



---



【💡】**具体与外部 LLM 交互是如何实现的，这些流事件是如何抽象出来的**？

基于 nodejs 的 `ai-sdk`，可以直接通过 `npm install ai` 进行安装。OpenCode 主要通过 `ai-sdk` 中的API [StreamText](https://ai-sdk.dev/docs/reference/ai-sdk-core/stream-text) 实现与外部 LLM 之间的交互。由于不同提供商的消息格式差异（如 Anthropic 的 toolCallId 规范化），OpenCode 通过 sdk 中的 `wrapLanguageModel` 注入中间件，实现对不同 `provider` 的适配。

`doStream` 实现 `ai-sdk 消息` 与 LLM 原生json数据的转换，SDK提供默认实现，但也可以通过重写`LanguageModelV2.doStream` 实现自定义的逻辑。比如自定义各种类别的 `MessagePart` (`tool-input-start`, `tool-call`, ...)。

Opencode 提供了多种 `LanguageModelV2.doStream` 的实现，以支持不同的 LLM 后端（openai, gemini, claude ...）。同时，在发送消息前，通过 `wrapLanguageModel` 注入中间件，以便于上层消息在进入 `doStream` 之前，做一些额外的处理。Opencode 同样是通过中间件对不同`provider` 做适配。

```typescript
function isResponseOutputItemDoneChunk(
  chunk: z.infer<typeof openaiResponsesChunkSchema>,
): chunk is z.infer<typeof responseOutputItemDoneSchema> {
  return chunk.type === "response.output_item.done"
}

function isResponseOutputItemAddedChunk(
  chunk: z.infer<typeof openaiResponsesChunkSchema>,
): chunk is z.infer<typeof responseOutputItemAddedSchema> {
  return chunk.type === "response.output_item.added"
}

function isResponseFunctionCallArgumentsDeltaChunk(
  chunk: z.infer<typeof openaiResponsesChunkSchema>,
): chunk is z.infer<typeof responseFunctionCallArgumentsDeltaSchema> {
  return chunk.type === "response.function_call_arguments.delta"
}

// 在 doStream() 中使用 TransformStream 转换事件
return {
  stream: response.pipeThrough(
    new TransformStream<ParseResult<...>, LanguageModelV2StreamPart>({
      transform(chunk, controller) {
        const value = chunk.value

        // 函数调用开始
        if (isResponseOutputItemAddedChunk(value)) {
          if (value.item.type === "function_call") {
            controller.enqueue({
              type: "tool-input-start",
              id: value.item.call_id,
              toolName: value.item.name,
            })
          }
        }
        
        // 函数调用完成
        else if (isResponseOutputItemDoneChunk(value)) {
          if (value.item.type === "function_call") {
            controller.enqueue({ type: "tool-input-end", id: value.item.call_id })
            controller.enqueue({
              type: "tool-call",
              toolCallId: value.item.call_id,
              toolName: value.item.name,
              input: value.item.arguments,  // JSON 字符串
            })
          }
        }
        
        // 参数增量
        else if (isResponseFunctionCallArgumentsDeltaChunk(value)) {
          controller.enqueue({
            type: "tool-input-delta",
            id: toolCall.toolCallId,
            delta: value.delta,
          })
        }
      }
    })
  )
}
```







## Session 会话系统

Session 会话系统由以下核心模块组成：

| 模块             | 文件                                                         | 职责                                                         |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Session          | `index.ts` | 会话 CRUD、消息管理                                          |
| Storage          | `storage.ts` | 本地 JSON 文件持久化                                         |
| MessageV2        | `message-v2.ts` | 消息和 Part 的类型定义、序列化                               |
| SessionPrompt    | `prompt.ts` | 用户输入处理、LLM 调用循环                                   |
| SessionProcessor | `processor.ts` | 流式处理 LLM 响应                                            |
| LLM              | `llm.ts` | 底层 `streamText`(AI-SDK) 调用 <br>将上层消息转换成后端LLM能处理的格式 |

Session 主要由历史消息组成。

### Message 定义

`Message` 是 Agent 和 LLM 交互的基本单位，主要分为 `user` | `assistant` 2种类型，`user` 表示 用户发起 / 系统合成 的消息；`assistant` 表示 LLM 回复的消息。这里专门解释一下`系统合成`消息。有时候，为了推进 LLM 继续执行，Opencode 会显示添加一些 `user` 消息，这些消息会被标记为合成消息。比如在压缩完上下文后，opencode 会额外添加`user`消息： `Continue if you have next steps` ，以使得 LLM 在理解完压缩信息之后继续完成用户指定的任务。

每条 `Message` 又划分成不同的 `Part`，`Part` 类型如下所示：

```json
// src/session/message-v2.ts
消息部分类型：
├─ TextPart：纯文本
|- SubtaskPart: 表示拆解出来的子任务，通常会交给 SubAgent 执行。用户通过输入 /command 可以触发，command是内置指令，比如/review
├─ ReasoningPart：推理过程
├─ ToolPart：工具调用 + 结果状态
├─ FilePart：文件附件
├─ SnapshotPart：快照（VCS状态）
├─ PatchPart：代码补丁
├─ AgentPart：子代理调用。用户通过输入 @xxxAgent 触发，xxx表示Agent类别，比如 @plan, @build 等。
├─ CompactionPart：消息压缩
└─ TodoPart：任务列表
```



### Message 初始化 / 用户输入初始化

Opencode 通过 `createUserMessage` 函数将用户的输入转换成 `Message`. 大致流程如下所示：

```text
用户输入 → SessionPrompt.prompt() 或 SessionPrompt.command()
         ↓
     createUserMessage()
         ↓
     创建 MessageV2.User 消息
         ↓
     解析 parts（文本、文件、@agent 引用等）
```

用户主要有两种输入方式，`prompt` 和 `command`，前者很好理解，就是正常输入问题；后者表示用户调用预置的一些指令，比如`/review` 表示代码审计命令，后面紧跟要审计的源码文件。Opencode 会将用户输入的 `@xxx` 解析为 `AgentPart`，xxx表示 Agent 类别，比如 `@plan`, `@build` 等。 同时会将用户输入的 `/xxx` 解析为`SubtaskPart`，然后交由子Agent执行。

`createUserMessage` 主要是处理用户输入，必要时添加额外的`合成消息`。比如用户输入了文件资源，Opencode 除了直接将文件读进来，还会额外合成消息模拟 LLM 调用工具读取文件的过程。本质上是丰富 LLM 的上下文，提供更多背景信息。

```typescript
[
    {
        id: Identifier.ascending("part"),
        messageID: info.id,
        sessionID: input.sessionID,
        type: "text",
        synthetic: true,
        text: `Called the Read tool with the following input: ${JSON.stringify({ filePath: part.filename })}`,
    },
    {
        id: Identifier.ascending("part"),
        messageID: info.id,
        sessionID: input.sessionID,
        type: "text",
        synthetic: true,
        text: Buffer.from(part.filename, "base64url").toString(),
    },
    {
        ...part, // 用户的原始提问
        id: part.id ?? Identifier.ascending("part"),
        messageID: info.id,
        sessionID: input.sessionID,
    },
]
```



### Message 合成 (synthetic)

`synthetic` 字段定义在 [message-v2.ts:65](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)，用于标记那些**不是用户直接输入、而是系统自动生成**的消息 part。Message 合成主要是为了推动 LLM 继续工作，起到一个承上启下的作用。主要覆盖以下几大类场景：

1. **附件/资源展开**：将用户附加的文件、目录、MCP 资源转化为模型可理解的文本
2. **模式切换指令**：Plan/Build 模式之间切换时的引导提示
3. **子代理调用引导**：引导模型调用 task 工具
4. **模型兼容性**：防止推理模型因消息结构问题报错
5. **流程控制**：Shell 执行标记和 Compaction 后的继续指令



大部分消息合成的逻辑都在 `src/session/prompts.ts: createUserMessage` 中实现。

#### 防止推理模型报错

`TaskTool` 工具输出后的合成用户消息（`prompt.ts:458-479`）

当 task 有 `command` 时（比如用户输入`/review`，就是内置的代码审计的`command`），插入一条合成用户消息 `"Summarize the task tool output above and continue with your task."`，防止某些推理模型（如 Gemini）因缺少 thinking signature 而报错。

#### MCP资源读取

[prompt.ts:852-923](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

当用户附加的文件来源是 MCP resource 时，生成合成 text part：

- **读取提示**：`"Reading MCP resource: ..."`
- **资源文本内容**：MCP 返回的文本
- **二进制内容占位符**：`"[Binary content: ...]"`
- **读取失败提示**：`"Failed to read MCP resource ..."`

#### 文件附件展开（调用 Read 工具）

[prompt.ts:931-940](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

当用户附加纯文本文件（[text/plain](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)）时，合成调用 Read 工具的提示和文件内容。当用户附加的文件是本地文件路径时：

- 合成 `"Called the Read tool with the following input: ..."`
- 调用 [ReadTool](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 执行后，将结果作为合成 text part 插入
- 若读取失败，合成错误提示 `"Read tool failed to read ..."`

#### 目录附件展开（调用 List 工具）

[prompt.ts:1090-1111](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

当用户附加目录（[application/x-directory](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)）时，合成调用 List 工具的提示和列出的目录内容：

- 合成 `"Called the list tool with the following input: ${JSON.stringify(args)}"`
- 调用 `ListTool` 后，将结果作为和成 text part 插入

#### 非纯文本文件附件（通用 Read 提示） 

[prompt.ts:1111-1143](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

其他 MIME 类型的文件附件，合成 `"Called the Read tool with the following input: ..."`。

#### Agent 子代理调用提示

[prompt.ts:1143](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

当用户消息包含 `agent` 类型的 part 时（比如用户输入 `@explore` 就会创建 `type=agent, name=explore` 的 Message.part），合成指令 `" Use the above message and context to generate a prompt and call the task tool with subagent: ..."` 引导模型调用 task 工具，创建子Agent来执行。

#### Plan 模式注入

在 Plan 模式进入或退出时回额外注入一些提示词。



**进入 Plan 模式**：

用户通过直接指定 `Plan` 模式，或让 LLM 自行调用 `plan_enter` 工具来进入 `Plan` 模式。当调用 `plan_enter` 时，会先询问用户是否同意进入 `Plan` 模式。如果同意，则合成 `"User has requested to enter plan mode. Switch to plan mode and begin planning."`

第一次进入 `Plan` 模式是，在 `insertReminder` 函数中，会向最后一条用户消息追加 `PROMPT_PLAN` 提示。定义 plan 模式的工作流程和约束。



**退出 Plan 模式**：

LLM 通过调用 `plan_exit` 工具来退出 `Plan` 模式。退出前，会询问用户是否接受当前的规划。如果 `plan` 被用户批准，合成：`"The plan at .opencode/plans/*.md has been approved, you can now edit files. Execute the plan"` 切换到 build agent. 

退出 `Plan` 模式后，第一次进入其他模式，`insertReminder` 函数会合成 `BUILD_SWITCH` 提示，告知 LLM 规划已经完成，现在进入实际执行的阶段。



[prompt.ts:1204-1236](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

- **plan agent 提醒**：向最后一条用户消息追加 `PROMPT_PLAN` 提示
- **从 plan 切换到 build 时**：追加 `BUILD_SWITCH` 提示，包含所创建的规划的路径名（.opencode/plans/*.md）



#### 进入 Plan 模式的系统提醒

[prompt.ts:1253-1323](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

当切换到 `plan agent` 时，注入一大段 `<system-reminder>` 合成文本，定义 plan 模式的工作流程和约束。



#### Shell 命令执行

[prompt.ts:1376 (shell)](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

用户通过 Shell 入口执行命令时：

- 合成用户消息 `"The following tool was executed by the user"`。

- 合成 Assistant 消息，并添加 `ToolPart`:

  ```typescript
  const part: MessageV2.Part = {
        type: "tool",
        id: Identifier.ascending("part"),
        messageID: msg.id,
        sessionID: input.sessionID,
        tool: "bash",
        callID: ulid(),
        state: {
          status: "running",
          time: {
            start: Date.now(),
          },
          input: {
            command: input.command,
          },
        },
      }
  ```

- 调用 `bash` 工具执行具体的 shell 命令

- 将结果更新回 `ToolPart`:

  ```typescript
  if (part.state.status === "running") {
      part.state = {
          status: "completed",
          time: {
              ...part.state.time,
              end: Date.now(),
          },
          input: part.state.input,
          title: "",
          metadata: {
              output,
              description: "",
          },
          output,
      }
  ```

  

#### Compaction（上下文压缩）后的继续提示

[compaction.ts:182](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

自动 compaction 完成后，如果结果是 `"continue"` 且是自动触发的，插入合成用户消息 `"Continue if you have next steps"`。





### Session 持久化

Opencode 会将会话历史都持久化存储到本地磁盘上，位于 `~/.local/share/opencode/storage/`. 具体的文件结构如下：

```bash
storage/
├── session/{projectID}/{sessionID}.json    ← 会话元信息
├── message/{sessionID}/{messageID}.json    ← 每条消息
├── part/{messageID}/{partID}.json          ← 每条消息的各个组件(文本、工具调用等)
└── migration                               ← 数据迁移版本号
```

会话处理流程大致如下：

```bash
loop(sessionID)
  ├── 1. 加载所有消息历史  → MessageV2.stream(sessionID)
  │     (从 storage/message/{sessionID}/ 读取所有 JSON)
  ├── 2. 过滤压缩后的消息  → MessageV2.filterCompacted()
  ├── 3. 找到最后的 User/Assistant 消息
  ├── 4. 创建空的 Assistant Message 并写入磁盘
  ├── 5. 调用 SessionProcessor.process() → LLM.stream()
  │     ├── 将消息历史转换为 ModelMessage 格式
  │     ├── 拼接 system prompt
  │     └── 调用 AI SDK 的 streamText()
  └── 6. 处理流式响应（tool call / text / finish）
```

### 会话状态管理

```typescript
// src/session/status.ts
export namespace SessionStatus {
  export type Info = 
    | { type: "idle" }
    | { 
        type: "retry"
        attempt: number
        message: string
        next: number  // 下次重试时间戳，主要是网络错误导致的
      }
    | { type: "busy" }

  const state = Instance.state(() => {
    const data: Record<string, Info> = {}
    return data
  })

  export function set(sessionID: string, status: Info) {
    Bus.publish(Event.Status, {
      sessionID,
      status,
    })
    
    if (status.type === "idle") {
      delete state()[sessionID]  // 清空
    } else {
      state()[sessionID] = status
    }
  }
}
```

---

状态转移图如下所示：

```text
┌─────────────────────────────────┐
│        idle (初始状态)           │
└──────────────┬──────────────────┘
               │ 用户提交消息
               ↓
        ┌──────────────┐
        │    busy      │ ← 处理中
        └──────┬───────┘
               │
         ┌─────┴─────────────┐
         │                   │
    ✓ 成功                 ✗ 错误
         │                   │
         ↓                   ↓
    ┌────────┐        ┌─────────────┐
    │  idle  │        │   retry     │
    └────────┘        │ attempt: N  │
                      │ next: Tms   │
                      └──────┬──────┘
                             │
                    ┌────────┴────────┐
                    │                 │
               Max retries         Timer fires
               exceeded            │
                    │              ↓
                    └─→ ┌────────┐  │
                        │  busy  │←─┘
                        └──┬──┬──┘
                           ...
```





## 工具系统

Opencode 自带 30 多种工具，位于 `src/tool/`，该目录下有各类工具的实现和说明：

- **文件操作：** read, write, edit, glob, list, apply_patch
- **代码搜索：** grep, codesearch, lsp（语言服务器）
- **执行命令：** bash, batch
- **网络操作：** websearch, webfetch
- **复杂任务拆解**：plan, task(创建SubAgent执行复杂任务)
- **特殊功能：** question（询问用户）, todo（任务管理，主要是任务状态的追踪）



其中，有一项特殊的工具是 `todo`，它主要负责`任务管理`，追踪任务的执行情况。当任务比较复杂时，LLM 会通过调用 `Plan` / `Task` 工具对任务进行拆解，拆分成多个`step` 或 `subagents`。当然，如果 LLM 本身比较强大，它也可以自动对复杂任务做拆解。除此之外，用户本身也可能同时输入多个任务，待LLM处理。

`todo` 就是维护拆解出来的任务列表，追踪哪些任务完成，哪些任务没有完成。`todo_write`负责更新任务列表的状态，`todo_read`读取最新任务列表的状态。下面的代码是`todo_write`的参数，更新某一项任务的完成情况。

```typescript
export const Info = z
.object({
    content: z.string().describe("Brief description of the task"),
    status: z.string().describe("Current status of the task: pending, in_progress, completed, cancelled"),
    priority: z.string().describe("Priority level of the task: high, medium, low"),
    id: z.string().describe("Unique identifier for the todo item"),
})
```

此外，Opencode 还支持通过 `MCP` 获取可用的工具：

```typescript
// src/mcp/index.ts
export namespace MCP {
  // 支持的传输方式：
  - StdioClientTransport (标准输入输出)
  - SSEClientTransport (服务器发送事件)
  - StreamableHTTPClientTransport

  // 工具和资源动态获取
  export type Resource = {
    name: string
    uri: string
    description?: string
    mimeType?: string
    client: string
  }

  // OAuth 认证支持
  export class McpOAuthProvider
  export class McpOAuthCallback
}
```



## Context 上下文管理

Agent 在解决复杂问题时，很容易引起上下文爆炸，因此有效管理上下文成为 Agent 能否长久运行的关键。Opencode 的上下文管理思路比较简单，就是设定一个token阈值，超过阈值就会触发压缩操作，让 LLM 对现有历史消息进行总结和摘要。整体架构图如下所示。



<img src="../../picture/LLM/ContextManagement.png" alt="image-20260206141748462" style="zoom: 67%;" />

---



### 上下文压缩

压缩场景如下所示：

```text
Old Messages (可以丢弃)：
  M1-M10: 旧的对话记录
  ├─ Tool results (tokens 占用)
  ├─ Reasoning (不再需要)
  └─ 累积 >80% context

Compaction Process:
  1️⃣ 摘要 M1-M10 的关键信息
  2️⃣ 创建 CompactionPart
  3️⃣ 删除原消息
  4️⃣ 保留 M11+ (最近的)

Result:
  New Context:
    CompactionPart (摘要)
    M11-M15 (最近的消息)
    └─ Token 使用: ~40%
```

具体压缩逻辑的实现在 `src/session/compaction.ts` 中，该文件中有两个核心函数：`create` 和 `process`. 

`create`函数的功能如下：

- **仅创建一个"压缩任务"标记**，不执行实际的压缩操作
- 创建一条 [user](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 角色的消息
- 在该消息中添加一个 [CompactionPart](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)（类型为 `"compaction"`）
- [auto](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 字段标记这是自动触发还是手动触发

---



而 `process` 函数负责对历史消息进行压缩。其核心思路比较简单，相当于创建类型为`compaction`的`subAgent`，后续的压缩交由子代理来处理。核心流程如下：

1. 确定用户消息、并获取类别为 compaction 的子Agent

```typescript
const userMessage = input.messages.findLast((m) => m.info.id === input.parentID)!.info as MessageV2.User
const agent = await Agent.get("compaction") // 获取类别为 compaction 的子Agent
const model = agent.model
  ? await Provider.getModel(agent.model.providerID, agent.model.modelID)
  : await Provider.getModel(userMessage.model.providerID, userMessage.model.modelID)
```

2. 创建AI消息

```typescript
const msg = await Session.updateMessage({
  role: "assistant",
  mode: "compaction",
  agent: "compaction",
  summary: true,  // 标记为摘要消息
  // ...其他字段
})
```

3. 调度子Agent，并立即执行
   - 将所有历史消息转换为模型消息
   - 附加一条用户消息，要求 LLM 总结对话内容

```typescript
const processor = SessionProcessor.create({assistantMessage: msg, ...})

// 插件可以注入上下文或替换压缩提示
const compacting = await Plugin.trigger(
  "experimental.session.compacting",
  { sessionID: input.sessionID },
  { context: [], prompt: undefined },
)

// 默认提示词
const defaultPrompt = "Provide a detailed prompt for continuing our conversation above..."
const promptText = compacting.prompt ?? [defaultPrompt, ...compacting.context].join("\n\n")

const result = await processor.process({
  messages: [
    ...MessageV2.toModelMessages(input.messages, model),
    {
      role: "user",
      content: [{ type: "text", text: promptText }],
    },
  ],
  // ...
})
```

4. 压缩完毕后，合成用户消息，引导 LLM 继续处理接下来的工作。

```typescript
if (result === "continue" && input.auto) {
  // 创建一条 "Continue if you have next steps" 的消息
  const continueMsg = await Session.updateMessage({...})
  await Session.updatePart({
    messageID: continueMsg.id                                               
    type: "text",
    synthetic: true,
    text: "Continue if you have next steps",
  })
}
```



具体的压缩实例请参考上图。





## Skills 支持

Skills 是 OpenCode 中一种用于为 AI Agent 提供`专门知识`和`分步骤指导`的机制。当用户任务匹配某个可用技能的描述时，Agent 可以加载该技能获取详细指令。

### Skills 是什么

我理解的 Skills 就是对特定工作流的封装。Skills 不是一个简单的工具，它可以理解为：给智能体（Agent）增加“`可复用能力包`”的一种标准化方式。一般来将，Skill 由如下部分组成，Agent 在合适的场景下按需加载并使用。

- 技能元数据（技能名字、简要说明、在什么情况下触发对该技能的调用）
- 技能详细说明
  - 输入输出
  - 需要用到的工具
  - 详细的`工作流`/`执行流程`
- （可选）`可执行脚本/模板/资源`

举一个简单的例子，下面是技能 `sql-triage` 的说明文档 `SKILL.md`，其中 `---` XX `---` 之间的内容被称为 `frontmatter`，它提供了该技能的一些元数据和简要说明，供 LLM 理解。只有当 LLM 确定选择该技能时，才会加载完整的说明文档。

```text
---
name: sql-triage
description: 用于“定位 SQL 性能/正确性问题”的排障流程；当用户提到慢查询、索引、执行计划、SQL 报错时优先使用。
triggers:
  - "慢查询"
  - "索引"
  - "执行计划"
  - "EXPLAIN"
  - "SQL 报错"
inputs:
  - db_type
  - sql
outputs:
  - triage_report
tools:
  - run_explain
  - fetch_schema
---

# sql-triage

## 目标
1) 快速判断：是数据量/索引/写法/统计信息/锁等待/参数嗅探等哪类问题  
2) 形成结构化报告：现象、证据、假设、验证步骤、修复建议、回滚方案

## 工作流（强制步骤）
- Step 1: 收集上下文（DB 类型、表量级、SQL、慢到什么程度、是否偶发）
- Step 2: 调用 `fetch_schema` 获取表结构、索引
- Step 3: 调用 `run_explain` 获取执行计划
- Step 4: 生成 triage_report（使用 prompts/report_template.md 模板）
```



### Skill in Opencode

Opencode 首先会加载现有所有 Skills 的元数据（名字 + 简介），然后将元数据整合在一起，封装成 `SkillTool` 工具，方便 LLM 调用。当 LLM 调用 `SkillTool` 工具时，会选择具体的技能，随后再将完整的技能文档输入给 LLM. 

1. 首先定义如下 `skill` 结构信息

```typescript
// 定义 Skill 信息结构
export const Info = z.object({
  name: z.string(),
  description: z.string(),
  location: z.string(), // SKILL.md 文件路径
})
```

2. 加载所有 Skills 的元数据（名字 + 简介）：

```typescript
const OPENCODE_SKILL_GLOB = new Bun.Glob("{skill,skills}/**/SKILL.md")
const CLAUDE_SKILL_GLOB = new Bun.Glob("skills/**/SKILL.md")

const addSkill = async (match: string) => {
  // 1. 解析 SKILL.md 文件的 frontmatter
  const md = await ConfigMarkdown.parse(match)
  
  // 2. 验证必要字段 (name, description)
  const parsed = Info.pick({ name: true, description: true }).safeParse(md.data)
  
  // 3. 注册到 skills 记录中
  skills[parsed.data.name] = {
    name: parsed.data.name,
    description: parsed.data.description,
    location: match,
  }
}
```

3. 定义 `SkillTool` 工具，将所有 Skills 的简介整合到一起，供 LLM 选择。在这里并不加载完整的 Skill 说明，防止 token 爆炸。

```typescript
export const SkillTool = Tool.define("skill", async (ctx) => {
  const skills = await Skill.all()
  
  // 根据 agent 权限过滤可用技能
  const accessibleSkills = agent
    ? skills.filter((skill) => {
        const rule = PermissionNext.evaluate("skill", skill.name, agent.permission)
        return rule.action !== "deny"
      })
    : skills

  // 动态生成工具描述，列出所有可用技能
  const description = [
    "Load a skill to get detailed instructions for a specific task.",
    "<available_skills>",
    ...accessibleSkills.flatMap((skill) => [
      `  <skill>`,
      `    <name>${skill.name}</name>`,
      `    <description>${skill.description}</description>`,
      `  </skill>`,
    ]),
    "</available_skills>",
  ].join(" ")

  return {
    description,
    parameters: z.object({
      name: z.string().describe(`The skill identifier from available_skills`),
    }),
    async execute(params, ctx) {
      // 加载并解析技能内容
      const parsed = await ConfigMarkdown.parse(skill.location)
      
      return {
        title: `Loaded skill: ${skill.name}`,
        output: [`## Skill: ${skill.name}`, parsed.content.trim()].join("\n"),
      }
    },
  }
})
```

4. LLM 会通过调用 `SkillTool` 工具，选择具体的 Skill（参数 `parameters.name` 反映了 LLM 的选择），此时才会加载被选择的 Skill 的详细说明。下面代码中的 `output` 是工具的执行结果（`output = parsed.content`），包含了详细的工作流说明文档。`output` 将在下一轮对话中作为工具调用的`Observation` 传递给 LLM。LLM 由此获得特定 Skill 的完整上下文。

```typescript
async execute(params, ctx) {
  const skill = await Skill.get(params.name)
  
  // 请求权限
  await ctx.ask({
    permission: "skill",
    patterns: [params.name],
    always: [params.name],
    metadata: {},
  })
  
  // 加载技能内容
  const parsed = await ConfigMarkdown.parse(skill.location)
  
  return {
    title: `Loaded skill: ${skill.name}`,
    output: `## Skill: ${skill.name}\n\n${parsed.content.trim()}`,
    metadata: { name: skill.name, dir: path.dirname(skill.location) }
  }
}
```

5. 后面 LLM 会按照 `Skill.md` 中的特定工作流来完成用户提出的任务。

```text
---
name: my-skill-name
description: A brief description of what this skill does.
---

# Skill Title

Detailed instructions and guidance content here...
```



## SubAgent & Plan

`agent.ts` 中定义了各种类型的 agent，以及各类 agent 具备的初始权限。



首先，可以根据用户输入决定 agent 的类型，比如用户输入 `@agent_name`，显式指定执行什么类型的 agent. `@general` 显式调用处理通用任务的 agent，该 agent 的 `mode=subagent`，指明可以创建新的`subagent`来处理复杂问题中的一部分。一般来说，`subagent`的上下文和主agent是隔离开的，但也可以根据需要共享一部分上下文。`@plan` 显式调用规划 agent，用于将复杂问题一步一步拆解。

除了显式调用不同类型的 agent，还能通过 `tool call` 触发针对复杂任务的处理。`opencode` 将 `subagent生成`和`plan`分别封装成 `task.ts` 和 `PlanEnter/PlanExit` 工具，由 LLM 决定子agent拆分和`plan`的时机。



## 权限系统

Opencode 允许为不同 Agent 和 Tool 预置权限集合：

```typescript
// src/permission/next.ts
export namespace PermissionNext {
  // 权限规则：
  export type Rule = {
    permission: string           // "bash", "edit", "read", etc.
    pattern: string              // 文件/目录模式
    action: "allow" | "deny" | "ask"
  }

  // 示例规则：
  {
    permission: "edit"
    pattern: "src/*"
    action: "allow"
  }

  // 特殊权限：
  - doom_loop: 检测无限循环 (是否重复调用某个工具多次)
  - external_directory: 访问项目外目录
  - question: 询问用户
  - plan_enter/plan_exit: 计划模式切换
}
```



### 权限检查流程

```text
Tool.execute()
  ↓
ctx.ask({
  permission: "bash",
  patterns: ["rm", "rm -rf"],
  always: ["ls"]  // 无需询问的模式
})
  ↓
PermissionNext.evaluate()
  ├─ 匹配 pattern
  ├─ 查找 action: allow/deny/ask
  └─ return decision
  ↓
If action === "ask":
  ├─ 显示用户确认对话
  ├─ 用户选择：Allow/Reject
  └─ 返回 RejectedError (如拒绝)
     ↓
Processor detects RejectedError
  └─ blocked = true → exit loop
```



### 权限错误处理

```typescript
// src/session/processor.ts
case "tool-error": {
  const match = toolcalls[value.toolCallId]
  if (match && match.state.status === "running") {
    await Session.updatePart({
      ...match,
      state: {
        status: "error",
        input: value.input,
        error: value.error.toString(),
        time: {...},
      },
    })

    // 区分权限拒绝和其他错误
    if (
      value.error instanceof PermissionNext.RejectedError ||
      value.error instanceof Question.RejectedError
    ) {
      blocked = shouldBreak  // 中断循环
    }
  }
  break
}

// 权限决策配置
export const shouldBreak = 
  (await Config.get()).experimental?.continue_loop_on_deny !== true
```



## 错误处理与恢复

Opencode 聚焦3种错误的处理：

1. API调用错误，一般是接口错误活网络超时引起的；
2. 同一个工具重复调用多次，陷入死循环。
3. 权限拒绝处理

### API 错误

```typescript
// src/session/retry.ts
export namespace SessionRetry {
  // 1️⃣ API 错误的可重试性判断
  export function retryable(error: NamedError) {
    if (MessageV2.APIError.isInstance(error)) {
      if (!error.data.isRetryable) return undefined
      return error.data.message.includes("Overloaded")
        ? "Provider is overloaded"
        : error.data.message
    }
    
    // 检测特定错误类型
    const json = JSON.parse(error.data.message)
    if (json.error?.type === "too_many_requests")
      return "Too Many Requests"
    if (json.error?.type === "rate_limit")
      return "Rate Limited"
    if (json.error?.type === "server_error")
      return "Provider Server Error"
    
    return undefined
  }

  // 2️⃣ 指数退避重试延迟
  export function delay(attempt: number, error?: MessageV2.APIError) {
    const RETRY_INITIAL_DELAY = 2000      // 2秒
    const RETRY_BACKOFF_FACTOR = 2        // 指数因子
    const RETRY_MAX_DELAY_NO_HEADERS = 30_000  // 最大30秒

    // 优先读取服务器 Retry-After 头
    if (error?.data.responseHeaders?.["retry-after-ms"]) {
      const ms = Number.parseFloat(
        error.data.responseHeaders["retry-after-ms"]
      )
      return ms
    }

    // 计算指数退避
    return Math.min(
      RETRY_INITIAL_DELAY * Math.pow(RETRY_BACKOFF_FACTOR, attempt - 1),
      RETRY_MAX_DELAY_NO_HEADERS
    )
  }

  // 3️⃣ 带取消信号的睡眠
  export async function sleep(ms: number, signal: AbortSignal) {
    return new Promise((resolve, reject) => {
      const abortHandler = () => {
        clearTimeout(timeout)
        reject(new DOMException("Aborted", "AbortError"))
      }
      const timeout = setTimeout(() => {
        signal.removeEventListener("abort", abortHandler)
        resolve()
      }, Math.min(ms, RETRY_MAX_DELAY))
      
      signal.addEventListener("abort", abortHandler, { once: true })
    })
  }
}
```

---

**重试流程**：

```text
LLM API Call
  ↓ error
Is Retryable? ──no──→ throw error
  │ yes
  ↓
Get Retry-After
  ↓
Calculate Delay:
  ├─ Parse retry-after header (优先级最高)
  ├─ Exponential backoff (2s × 2^n)
  └─ Cap at 30s (没有header时)
  ↓
sleep(delay, abortSignal)
  ↓
Retry API Call

Max: 3-5 attempts
```



### 工具调用无限循环检测

```typescript
// src/session/processor.ts
const DOOM_LOOP_THRESHOLD = 3

case "tool-call": {
  const match = toolcalls[value.toolCallId]
  if (match) {
    const parts = await MessageV2.parts(input.assistantMessage.id)
    const lastThree = parts.slice(-DOOM_LOOP_THRESHOLD)

    // 检测：最后3次调用相同工具，参数相同
    if (
      lastThree.length === DOOM_LOOP_THRESHOLD &&
      lastThree.every(
        (p) =>
          p.type === "tool" &&
          p.tool === value.toolName &&
          JSON.stringify(p.state.input) === JSON.stringify(value.input)
      )
    ) {
      // 触发权限请求以中断循环
      await PermissionNext.ask({
        permission: "doom_loop",
        patterns: [value.toolName],
        metadata: {
          tool: value.toolName,
          input: value.input,
        },
      })
    }
  }
  break
}
```

---

**无限循环中断流程**：

```text
Tool Call #1: read("config.json")
Tool Call #2: read("config.json")   ← 相同参数
Tool Call #3: read("config.json")   ← 检测到！

↓ Trigger doom_loop permission request
↓ User Confirmation Required
  ├─ Allow Once
  ├─ Always Allow  
  └─ Reject        ← 中断循环
```







## 事件总线 BUS

全局事件发布-订阅系统，用于实时更新前端 TUI

```typescript
// src/bus/index.ts, bus-event.ts
export namespace Bus {
  export async function publish<D extends BusEvent.Definition>(
    def: D,
    properties: z.output<D["properties"]>
  ): Promise<void>

  export function subscribe(event: BusEvent.Definition, cb): Subscription
}

// 核心事件：
- session.created
- session.updated
- project.updated
- mcp.tools.changed
- server.instance.disposed
```







## 实际案例分析

这里主要总结 3 种场景的案例分析：

- 用户输入 `@explore xxx`，直接创建 `AgentPart` 消息，调用子代理完成需求。`createUserMessage` 在碰到初始 `AgentPart` 消息时，会额外合成用户消息，来引导 LLM 调用 `TaskTool` 工具创建 `subagent` 来处理任务。合成消息的内容移步至第一个案例的具体分析部分。
- 用户输入 `/review xxx`，创建 `SubtaskPart` 消息，通过 `TaskTool` 工具创建 `subagent` 来执行代码审计任务
- 用户通过执行 `Plan` 任务。



### 用户输入 @explore 命令

用户输入 `@explore 查找项目中所有的API端点`。假设用户已有一个 session，ID 为 `session_01JTEST123`，当前消息历史为空。

#### 步骤 1：用户输入被接收

用户在 UI 中输入：`@explore 查找项目中所有的 API 端点`

这个输入会被前端解析，识别出 [@explore](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 是一个 agent 引用，然后调用 API。

```typescript
// src/session/prompt.ts:148-170
export const prompt = fn(PromptInput, async (input) => {
  // input = {
  //   sessionID: "session_01JTEST123",
  //   parts: [
  //     { type: "text", text: "查找项目中所有的 API 端点" },
  //     { type: "agent", name: "explore" }
  //   ],
  //   agent: "build"  // 默认 agent
  // }
  
  const session = await Session.get(input.sessionID)
  await SessionRevert.cleanup(session)
  
  const message = await createUserMessage(input)  // ← 关键步骤
  await Session.touch(input.sessionID)
  
  // ...权限处理...
  
  return loop(input.sessionID)  // ← 进入主循环
})
```

#### 步骤 2：创建用户消息 createUserMessage

```typescript
async function createUserMessage(input: PromptInput) {
  const agent = await Agent.get(input.agent ?? (await Agent.defaultAgent()))
  // agent = { name: "build", mode: "primary", ... }
  
  const info: MessageV2.Info = {
    id: "message_01ABC001",  // Identifier.ascending("message") 生成
    role: "user",
    sessionID: "session_01JTEST123",
    time: { created: 1738500000000 },
    agent: "build",
    model: { providerID: "anthropic", modelID: "claude-sonnet-4-20250514" },
  }
```

处理Parts: 遍历输入的 parts，对每个 part 进行转换。这里`createUserMessage` 在碰到初始 `AgentPart` 消息时，会额外合成用户消息，来引导 LLM 调用 `TaskTool` 工具创建 `subagent` 来处理任务。

```typescript
  const parts = await Promise.all(
    input.parts.map(async (part): Promise<MessageV2.Part[]> => {
      // 处理 agent part (type === "agent")
      if (part.type === "agent") {
        // 检查权限
        const perm = PermissionNext.evaluate("task", part.name, agent.permission)
        const hint = perm.action === "deny" ? " . Invoked by user; guaranteed to exist." : ""
        
        return [
          // 1. 保留原始的 AgentPart
          {
            id: "part_01ABC002",
            type: "agent",
            name: "explore",
            messageID: "message_01ABC001",
            sessionID: "session_01JTEST123",
          },
          // 2. 添加合成文本提示，指导 LLM 调用 task 工具
          {
            id: "part_01ABC003",
            messageID: "message_01ABC001",
            sessionID: "session_01JTEST123",
            type: "text",
            synthetic: true,
            text: " Use the above message and context to generate a prompt and call the task tool with subagent: explore"
          },
        ]
      }
      
      // 处理 text part
      return [{
        id: "part_01ABC004",
        type: "text",
        text: "查找项目中所有的 API 端点",
        messageID: "message_01ABC001",
        sessionID: "session_01JTEST123",
      }]
    })
  ).then((x) => x.flat())
```

此时消息历史：

```text
Session: session_01JTEST123
消息历史:
├── Message[user] id=message_01ABC001
│   ├── TextPart id=part_01ABC004: "查找项目中所有的 API 端点"
│   ├── AgentPart id=part_01ABC002: name="explore"
│   └── TextPart id=part_01ABC003 (synthetic): 
│       " Use the above message and context to generate a prompt and call the task tool with subagent: explore"
```



#### 步骤3：进入主循环 loop

由于没有初始 `compaction` 和 `subtask` 消息，直接进入正常的处理流程。

```typescript
// 收集未完成的 subtask 和 compaction 任务
      const task = msg.parts.filter((part) => 
        part.type === "compaction" || part.type === "subtask"
      )
      if (task && !lastFinished) {
        tasks.push(...task)
      }
    }
    
    // 此时: lastUser = message_01ABC001, lastAssistant = undefined
    // tasks = [] (用户消息中没有 SubtaskPart，只有 AgentPart)
```



#### 步骤4：正常 LLM 处理

由于 [tasks](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 数组为空（没有 SubtaskPart），进入正常的 LLM 处理流程，此时的历史消息如下：

```text
Session: session_01JTEST123
消息历史:
├── Message[user] id=message_01ABC001
│   ├── TextPart: "查找项目中所有的 API 端点"
│   ├── AgentPart: name="explore"
│   └── TextPart (synthetic): "Use the above message... subagent: explore"
│
└── Message[assistant] id=message_01ABC005 (刚创建，空的)
    └── (暂无 parts)
```



#### 步骤 5：解析工具并调用 LLM

```typescript
// step 5.1: 检查用户是否通过 @ 显式调用了 agent
const lastUserMsg = msgs.findLast((m) => m.info.role === "user")
const bypassAgentCheck = lastUserMsg?.parts.some((p) => p.type === "agent") ?? false
// bypassAgentCheck = true (因为有 AgentPart)

// step 5.2: 解析可用工具
const tools = await resolveTools({
    agent,
    session,
    model,
    tools: lastUser.tools,
    processor,
    bypassAgentCheck,  // true - 会影响 task 工具的权限检查
})
// tools 包含: read, write, edit, bash, task, grep, glob, ...

// step 5.3: 生成会话标题（异步，不阻塞）
if (step === 1) {
    ensureTitle({ session, modelID, providerID, history: msgs })
}

// step 5.4: 调用 LLM
const result = await processor.process({
    user: lastUser,
    agent,
    abort,
    sessionID,
    system: [...(await SystemPrompt.environment()), ...(await SystemPrompt.custom())],
    messages: MessageV2.toModelMessages(sessionMessages, model),
    tools,
    model,
})
```



#### 步骤 6：SessionProcessor 处理 LLM 响应

假设 LLM 首先生成一段思考文本，此时消息历史：

```text
Session: session_01JTEST123
消息历史:
├── Message[user] id=message_01ABC001
│   ├── TextPart: "查找项目中所有的 API 端点"
│   ├── AgentPart: name="explore"
│   └── TextPart (synthetic): "Use the above message... subagent: explore"
│
└── Message[assistant] id=message_01ABC005
    └── TextPart id=part_01ABC006: "我来帮你查找项目中的 API 端点。"
```



#### 步骤 7：LLM 决定调用 Task 工具

LLM 根据合成文本的提示，决定调用 [task](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 工具，消息历史如下：

```text
Session: session_01JTEST123
消息历史:
├── Message[user] id=message_01ABC001
│   ├── TextPart: "查找项目中所有的 API 端点"
│   ├── AgentPart: name="explore"
│   └── TextPart (synthetic): "Use the above message... subagent: explore"
│
└── Message[assistant] id=message_01ABC005
    ├── TextPart id=part_01ABC006: "我来帮你查找项目中的 API 端点。"
    └── ToolPart id=part_01ABC007:
        tool: "task"
        callID: "call_01XYZ001"
        state: {
          status: "running",
          input: {
            description: "查找 API 端点",
            prompt: "查找项目中所有的 API 端点...",
            subagent_type: "explore"
          },
          time: { start: 1738500003000 }
        }
```



#### 步骤 8：执行 TaskTool

在 [resolveTools](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 函数中注册的 [task](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 工具的 [execute](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 函数被调用：

```typescript
// src/tools/task.ts
async execute(params, ctx) {
  // params = {
  //   description: "查找 API 端点",
  //   prompt: "查找项目中所有的 API 端点...",
  //   subagent_type: "explore"
  // }
  // ctx = {
  //   sessionID: "session_01JTEST123",
  //   messageID: "message_01ABC005",
  //   callID: "call_01XYZ001",
  //   extra: { bypassAgentCheck: true },  // 因为用户用了 @
  //   ...
  // }

  // 8.1: 权限检查（因为 bypassAgentCheck=true，跳过）
  if (!ctx.extra?.bypassAgentCheck) {
    await ctx.ask({ permission: "task", patterns: [params.subagent_type] })
  }

  // 8.2: 获取 agent
  const agent = await Agent.get(params.subagent_type)
  // agent = { name: "explore", mode: "subagent", ... }

  // 8.3: 创建子 session
  const session = await Session.create({
    parentID: ctx.sessionID,  // "session_01JTEST123"
    title: "查找 API 端点 (@explore subagent)",
    permission: [
      { permission: "todowrite", pattern: "*", action: "deny" },
      { permission: "todoread", pattern: "*", action: "deny" },
      { permission: "task", pattern: "*", action: "deny" },  // explore 没有 task 权限
    ],
  })
  // session.id = "session_01JCHILD001"
```

子 Session 创建完成：

```text
Session: session_01JTEST123 (父)
├── parentID: undefined
└── title: "..."

Session: session_01JCHILD001 (子)
├── parentID: "session_01JTEST123"
└── title: "查找 API 端点 (@explore subagent)"
```

```typescript
  // 8.4: 更新工具元数据
  ctx.metadata({
    title: params.description,
    metadata: { sessionId: session.id },
  })

  // 8.5: 订阅子 session 的 part 更新事件
  const unsub = Bus.subscribe(MessageV2.Event.PartUpdated, async (evt) => {
    if (evt.properties.part.sessionID !== session.id) return
    if (evt.properties.part.type !== "tool") return
    // 更新进度...
  })

  // 8.6: 在子 session 中执行 prompt
  const result = await SessionPrompt.prompt({
    messageID: "message_01JCHILD001",
    sessionID: session.id,  // "session_01JCHILD001"
    model: { modelID: "claude-sonnet-4-20250514", providerID: "anthropic" },
    agent: "explore",  // 使用 explore agent
    tools: {
      todowrite: false,
      todoread: false,
      task: false,  // 禁止嵌套 task
    },
    parts: [{ type: "text", text: "查找项目中所有的 API 端点..." }],
  })
```



#### 步骤 9：子 Session 执行

这里会递归调用 loop，具体细节不再赘述。假设 explore agent 决定使用 [grep](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 和 [glob](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 工具来查找 API 端点：

```text
Session: session_01JCHILD001 (子)
消息历史:
├── Message[user] id=message_01JCHILD001
│   └── TextPart: "查找项目中所有的 API 端点..."
│
└── Message[assistant] id=message_01JCHILD002
    ├── TextPart: "我来查找项目中的 API 端点..."
    ├── ToolPart: tool="glob" input={pattern:"**/routes/**"} 
    │   state: { status: "completed", output: "src/routes/api.ts\nsrc/routes/users.ts" }
    ├── ToolPart: tool="grep" input={query:"@Get|@Post|@Put"} 
    │   state: { status: "completed", output: "Found 15 matches..." }
    └── TextPart: "找到以下 API 端点:\n1. GET /api/users\n2. POST /api/users\n..."
```



#### 步骤 10：TaskTool 返回结果

回到 [TaskTool.execute](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)：

```typescript
  // 子 session prompt 完成，result 是子 session 的最后一条 assistant 消息
  unsub()  // 取消订阅
  
  // 收集子 session 中所有工具调用的摘要
  const messages = await Session.messages({ sessionID: session.id })
  const summary = messages
    .filter((x) => x.info.role === "assistant")
    .flatMap((msg) => msg.parts.filter((x) => x.type === "tool"))
    .map((part) => ({
      id: part.id,
      tool: part.tool,
      state: {
        status: part.state.status,
        title: part.state.status === "completed" ? part.state.title : undefined,
      },
    }))
  
  // 获取最终文本
  const text = result.parts.findLast((x) => x.type === "text")?.text ?? ""
  // text = "找到以下 API 端点:\n1. GET /api/users\n..."
  
  const output = text + "\n\n" + [
    "<task_metadata>",
    `session_id: ${session.id}`,  // session_01JCHILD001
    "</task_metadata>"
  ].join("\n")

  return {
    title: params.description,  // "查找 API 端点"
    metadata: {
      summary,  // [{tool:"glob",...}, {tool:"grep",...}]
      sessionId: session.id,
    },
    output,
  }
}
```



#### 步骤 11：Processor 处理 Tool 结果

```typescript
case "tool-result":
// value = {
//   toolCallId: "call_01XYZ001",
//   input: {...},
//   output: {
//     title: "查找 API 端点",
//     metadata: { summary: [...], sessionId: "session_01JCHILD001" },
//     output: "找到以下 API 端点:...\n\n<task_metadata>..."
//   }
// }
const match = toolcalls["call_01XYZ001"]
if (match && match.state.status === "running") {
    await Session.updatePart({
        ...match,
        state: {
            status: "completed",
            input: match.state.input,
            output: value.output.output,
            metadata: value.output.metadata,
            title: value.output.title,
            time: {
                start: match.state.time.start,
                end: Date.now(),
            },
        },
    })
    delete toolcalls["call_01XYZ001"]
}
break
```

此时父 Session 的消息历史：

```text
Session: session_01JTEST123 (父)
消息历史:
├── Message[user] id=message_01ABC001
│   ├── TextPart: "查找项目中所有的 API 端点"
│   ├── AgentPart: name="explore"
│   └── TextPart (synthetic): "Use the above message... subagent: explore"
│
└── Message[assistant] id=message_01ABC005
    ├── TextPart id=part_01ABC006: "我来帮你查找项目中的 API 端点。"
    └── ToolPart id=part_01ABC007:
        tool: "task"
        callID: "call_01XYZ001"
        state: {
          status: "completed",  ← 状态更新为 completed
          input: { description: "查找 API 端点", ... },
          output: "找到以下 API 端点:\n1. GET /api/users\n...\n\n<task_metadata>...",
          title: "查找 API 端点",
          metadata: { summary: [...], sessionId: "session_01JCHILD001" },
          time: { start: 1738500003000, end: 1738500010000 }
        }
```



#### 步骤 12：LLM 继续响应

LLM 看到 task 工具的结果后，继续生成最终响应：

```typescript
case "text-start":
currentText = {
    id: "part_01ABC008",
    messageID: "message_01ABC005",
    sessionID: "session_01JTEST123",
    type: "text",
    text: "",
    time: { start: Date.now() },
}
break

case "text-delta":
// LLM 输出总结
currentText.text += value.text
await Session.updatePart({ part: currentText, delta: value.text })
break

case "text-end":
currentText.time.end = Date.now()
await Session.updatePart(currentText)
break

case "finish-step":
// 记录 token 使用量
input.assistantMessage.finish = value.finishReason  // "stop"
input.assistantMessage.cost += usage.cost
input.assistantMessage.tokens = usage.tokens
await Session.updateMessage(input.assistantMessage)
break
```



#### 步骤 13：Loop 退出

回到父 session 的 [loop](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)：

```typescript
      // processor.process 返回 "continue"
      const result = await processor.process({...})
      if (result === "stop") break
      if (result === "compact") {
        await SessionCompaction.create({...})
      }
      continue  // 继续下一次循环
    }
```

下一次循环：

```typescript
    // step = 2
    let msgs = await MessageV2.filterCompacted(MessageV2.stream(sessionID))
    
    // 现在 lastAssistant = message_01ABC005, lastAssistant.finish = "stop"
    // lastUser = message_01ABC001
    
    if (
      lastAssistant?.finish &&                          // true, "stop"
      !["tool-calls", "unknown"].includes(lastAssistant.finish) &&  // true
      lastUser.id < lastAssistant.id                    // true
    ) {
      log.info("exiting loop", { sessionID })
      break  // 退出循环！
    }
```

最终状态：父 Session 的消息历史：

```text
Session: session_01JTEST123 (父)
消息历史:
├── Message[user] id=message_01ABC001
│   ├── TextPart: "查找项目中所有的 API 端点"
│   ├── AgentPart: name="explore"
│   └── TextPart (synthetic): "Use the above message... subagent: explore"
│
└── Message[assistant] id=message_01ABC005
    │ role: "assistant"
    │ agent: "build"
    │ finish: "stop"
    │ cost: 0.0045
    │ tokens: { input: 1500, output: 200, ... }
    │ time: { created: 1738500001000, completed: 1738500015000 }
    │
    ├── StepStartPart: snapshot="snap_001"
    ├── TextPart id=part_01ABC006: "我来帮你查找项目中的 API 端点。"
    ├── ToolPart id=part_01ABC007:
    │   tool: "task"
    │   state: {
    │     status: "completed",
    │     input: { description: "查找 API 端点", prompt: "...", subagent_type: "explore" },
    │     output: "找到以下 API 端点:\n1. GET /api/users\n2. POST /api/users\n...",
    │     title: "查找 API 端点",
    │     metadata: { summary: [...], sessionId: "session_01JCHILD001" }
    │   }
    ├── TextPart id=part_01ABC008: "根据搜索结果，你的项目中有以下 API 端点：..."
    └── StepFinishPart: reason="stop", cost=0.0045, tokens={...}
```

子 Session 的消息历史：

```text
Session: session_01JCHILD001 (子)
parentID: "session_01JTEST123"
title: "查找 API 端点 (@explore subagent)"
消息历史:
├── Message[user] id=message_01JCHILD001
│   └── TextPart: "查找项目中所有的 API 端点..."
│
└── Message[assistant] id=message_01JCHILD002
    │ role: "assistant"
    │ agent: "explore"
    │ finish: "stop"
    │
    ├── TextPart: "我来查找项目中的 API 端点..."
    ├── ToolPart: tool="glob", state={status:"completed", output:"src/routes/..."}
    ├── ToolPart: tool="grep", state={status:"completed", output:"Found 15 matches..."}
    └── TextPart: "找到以下 API 端点:\n1. GET /api/users\n2. POST /api/users\n..."
```



### 用户执行 /review 命令

`/review xxx` 这类命令会被解析为 `SubTaskPart` 消息。 

#### 步骤 1：命令解析

调用入口：`prompt.ts` 的 `command` 函数

```typescript
// src/session/prompt.ts:1579-1700
export async function command(input: CommandInput) {
  // input = {
  //   sessionID: "session_01JREVIEW001",
  //   command: "review",
  //   arguments: "",
  // }
  
  log.info("command", input)
  
  // 获取命令配置
  const command = await Command.get("review")
  // command = {
  //   name: "review",
  //   description: "review changes [commit|branch|pr], defaults to uncommitted",
  //   subtask: true,  ← 关键！这是一个 subtask 命令
  //   template: "Review the changes..."
  // }
```



判断是否为 subtask:

```typescript
  const agent = await Agent.get(agentName)
  const isSubtask = (agent.mode === "subagent" && command.subtask !== false) || command.subtask === true
  // agent.mode = "primary" (build 是 primary)
  // command.subtask = true
  // 所以 isSubtask = true
```



#### 步骤 2：创建 SubtaskPart

```typescript
  const templateParts = await resolvePromptParts(template)
  // templateParts = [{ type: "text", text: "Review the changes..." }]
  
  const parts = isSubtask
    ? [
        {
          type: "subtask" as const,
          agent: agent.name,  // "build"
          description: command.description ?? "",  // "review changes..."
          command: input.command,  // "review"
          model: {
            providerID: taskModel.providerID,
            modelID: taskModel.modelID,
          },
          prompt: templateParts.find((y) => y.type === "text")?.text ?? "",
          // prompt = "Review the changes..."
        },
      ]
    : [...templateParts]
  
  // parts = [{
  //   type: "subtask",
  //   agent: "build",
  //   description: "review changes...",
  //   command: "review",
  //   prompt: "Review the changes...",
  //   model: { providerID: "anthropic", modelID: "claude-sonnet-4-20250514" }
  // }]
```



#### 步骤 3：调用 prompt

```typescript
  const userAgent = isSubtask ? (input.agent ?? (await Agent.defaultAgent())) : agentName
  // userAgent = "build"
  
  const result = await prompt({
    sessionID: input.sessionID,
    model: userModel,
    agent: userAgent,
    parts,  // 包含 SubtaskPart
  })
```



#### 步骤 4：createUserMessage 处理 SubtaskPart

```typescript
// createUserMessage 中
const parts = await Promise.all(
  input.parts.map(async (part): Promise<MessageV2.Part[]> => {
    // SubtaskPart 不需要特殊转换，直接保存
    return [{
      id: "part_01REV001",
      ...part,
      messageID: info.id,
      sessionID: input.sessionID,
    }]
  })
).then((x) => x.flat())
```

此时消息历史：

```
Session: session_01JREVIEW001
消息历史:
└── Message[user] id=message_01REV001
    └── SubtaskPart id=part_01REV001:
        type: "subtask"
        agent: "build"
        description: "review changes..."
        command: "review"
        prompt: "Review the changes..."
        model: { providerID: "anthropic", modelID: "..." }
```



#### 步骤 5：Loop 检测到 SubtaskPart

```typescript
// loop 中
let tasks: (MessageV2.CompactionPart | MessageV2.SubtaskPart)[] = []

for (let i = msgs.length - 1; i >= 0; i--) {
  const msg = msgs[i]
  // ...
  const task = msg.parts.filter((part) => 
    part.type === "compaction" || part.type === "subtask"
  )
  if (task && !lastFinished) {
    tasks.push(...task)
  }
}

// tasks = [{ type: "subtask", agent: "build", ... }]
const task = tasks.pop()
// task = { type: "subtask", ... }
```



#### 步骤 6：执行 Subtask

```typescript
// pending subtask
if (task?.type === "subtask") {
  const taskTool = await TaskTool.init()
  const taskModel = task.model 
    ? await Provider.getModel(task.model.providerID, task.model.modelID) 
    : model
  
  // 创建 assistant 消息
  const assistantMessage = await Session.updateMessage({
    id: "message_01REV002",
    role: "assistant",
    parentID: lastUser.id,  // "message_01REV001"
    sessionID,
    mode: task.agent,  // "build"
    agent: task.agent,  // "build"
    // ...
  })
```

此时历史消息：

```
Session: session_01JREVIEW001
消息历史:
├── Message[user] id=message_01REV001
│   └── SubtaskPart: agent="build", command="review", prompt="Review..."
│
└── Message[assistant] id=message_01REV002 (刚创建)
    └── (暂无 parts)
```

创建 ToolPart:

```typescript
  let part = await Session.updatePart({
    id: "part_01REV002",
    messageID: assistantMessage.id,
    sessionID: assistantMessage.sessionID,
    type: "tool",
    callID: "call_01REV001",
    tool: TaskTool.id,  // "task"
    state: {
      status: "running",
      input: {
        prompt: task.prompt,
        description: task.description,
        subagent_type: task.agent,
        command: task.command,
      },
      time: { start: Date.now() },
    },
  })
```

此时历史消息：

```
Session: session_01JREVIEW001
消息历史:
├── Message[user] id=message_01REV001
│   └── SubtaskPart: agent="build", command="review", prompt="Review..."
│
└── Message[assistant] id=message_01REV002
    └── ToolPart id=part_01REV002:
        tool: "task"
        callID: "call_01REV001"
        state: {
          status: "running",
          input: {
            prompt: "Review the changes...",
            description: "review changes...",
            subagent_type: "build",
            command: "review"
          }
        }
```



#### 步骤 7：执行 TaskTool

```typescript
  const taskArgs = {
    prompt: task.prompt,
    description: task.description,
    subagent_type: task.agent,
    command: task.command,
  }
  
  // 触发 plugin hook
  await Plugin.trigger("tool.execute.before", {...})
  
  // 创建 tool context
  const taskCtx: Tool.Context = {
    agent: task.agent,
    messageID: assistantMessage.id,
    sessionID: sessionID,
    abort,
    callID: part.callID,
    extra: { bypassAgentCheck: true },  // ← 命令触发的，跳过权限检查
    // ...
  }
  
  // 执行 task tool
  const result = await taskTool.execute(taskArgs, taskCtx)
```

TaskTool 内部会创建 subagent 并执行，流程与案例一类似。

#### 步骤 8：更新结果

```typescript
  // 触发 plugin hook
  await Plugin.trigger("tool.execute.after", {...})
  
  // 更新 assistant 消息
  assistantMessage.finish = "tool-calls"
  assistantMessage.time.completed = Date.now()
  await Session.updateMessage(assistantMessage)
  
  // 更新 ToolPart 状态
  if (result && part.state.status === "running") {
    await Session.updatePart({
      ...part,
      state: {
        status: "completed",
        input: part.state.input,
        title: result.title,
        metadata: result.metadata,
        output: result.output,
        time: {
          ...part.state.time,
          end: Date.now(),
        },
      },
    })
  }
```

此时历史消息：

```
Session: session_01JREVIEW001
消息历史:
├── Message[user] id=message_01REV001
│   └── SubtaskPart: agent="build", command="review", prompt="Review..."
│
└── Message[assistant] id=message_01REV002
    │ finish: "tool-calls"
    │ time: { created: ..., completed: ... }
    │
    └── ToolPart id=part_01REV002:
        tool: "task"
        state: {
          status: "completed",
          input: { ... },
          output: "Code review complete:\n- Found 3 issues...",
          title: "review changes...",
          metadata: { sessionId: "session_01JCHILD002" }
        }
```



#### 步骤 9：添加合成用户消息（针对 command subtask）

```typescript
  // 因为 task.command 存在（"review"），添加合成用户消息
  if (task.command) {
    const summaryUserMsg: MessageV2.User = {
      id: "message_01REV003",
      sessionID,
      role: "user",
      time: { created: Date.now() },
      agent: lastUser.agent,
      model: lastUser.model,
    }
    await Session.updateMessage(summaryUserMsg)
    await Session.updatePart({
      id: "part_01REV003",
      messageID: summaryUserMsg.id,
      sessionID,
      type: "text",
      text: "Summarize the task tool output above and continue with your task.",
      synthetic: true,
    })
  }
  
  continue  // 继续循环
```

此时历史消息：

```
Session: session_01JREVIEW001
消息历史:
├── Message[user] id=message_01REV001
│   └── SubtaskPart: agent="build", command="review"
│
├── Message[assistant] id=message_01REV002
│   │ finish: "tool-calls"
│   └── ToolPart: tool="task", state={status:"completed", output:"Code review..."}
│
└── Message[user] id=message_01REV003 (合成)
    └── TextPart (synthetic): "Summarize the task tool output above..."
```



#### 步骤 10：继续循环 - LLM 总结

```typescript
// 下一次循环
// lastUser = message_01REV003 (合成消息)
// lastAssistant = message_01REV002, finish = "tool-calls"
// 因为 finish = "tool-calls"，不会退出循环

// 正常 LLM 处理
const processor = SessionProcessor.create({
  assistantMessage: await Session.updateMessage({
    id: "message_01REV004",
    parentID: "message_01REV003",
    role: "assistant",
    // ...
  }),
  // ...
})

const result = await processor.process({...})
```

LLM 会看到之前的 task 工具结果，然后生成总结。最终状态如下：

```
Session: session_01JREVIEW001
消息历史:
├── Message[user] id=message_01REV001
│   └── SubtaskPart: agent="build", command="review"
│
├── Message[assistant] id=message_01REV002
│   │ finish: "tool-calls"
│   └── ToolPart: tool="task", state={status:"completed"}
│
├── Message[user] id=message_01REV003 (合成)
│   └── TextPart (synthetic): "Summarize the task tool output..."
│
└── Message[assistant] id=message_01REV004
    │ finish: "stop"
    └── TextPart: "代码审查完成！发现以下问题：\n1. ..."
```



### Plan 模式完整流程

用户选择 [plan](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) agent，输入：`实现一个用户认证系统`

#### 步骤 1：prompt 调用

```typescript
prompt({
  sessionID: "session_01JPLAN001",
  agent: "plan",  // ← 选择 plan agent
  parts: [{ type: "text", text: "实现一个用户认证系统" }],
})
```



#### 步骤 2：createUserMessage

```typescript
// createUserMessage
const agent = await Agent.get("plan")
// agent = {
//   name: "plan",
//   mode: "primary",
//   permission: [
//     { permission: "edit", pattern: "*", action: "deny" },
//     { permission: "edit", pattern: ".opencode/plans/*.md", action: "allow" },
//     // ...
//   ]
// }

const info: MessageV2.Info = {
  id: "message_01PLAN001",
  role: "user",
  sessionID: "session_01JPLAN001",
  agent: "plan",  // ← 记录使用的是 plan agent
  // ...
}
```

此时历史消息：

```
Session: session_01JPLAN001
消息历史:
└── Message[user] id=message_01PLAN001
    │ agent: "plan"
    └── TextPart: "实现一个用户认证系统"
```



#### 步骤 3：Loop 进入，insertReminders 注入 Plan 提示

```typescript
// loop 中
msgs = await insertReminders({
  messages: msgs,
  agent,  // plan agent
  session,
})
```

**insertReminders 函数**：

```typescript
// src/session/prompt.ts:1188-1216
async function insertReminders(input) {
  const userMessage = input.messages.findLast((msg) => msg.info.role === "user")
  
  if (!Flag.OPENCODE_EXPERIMENTAL_PLAN_MODE) {
    // 非实验模式
    if (input.agent.name === "plan") {
      userMessage.parts.push({
        id: "part_synthetic_plan",
        messageID: userMessage.info.id,
        sessionID: userMessage.info.sessionID,
        type: "text",
        text: PROMPT_PLAN,  // plan.txt 的内容
        synthetic: true,
      })
    }
    return input.messages
  }
  
  // 实验模式（更详细的工作流提示）
  // ...
}
```

**PROMPT_PLAN 内容**(来自 `plan.txt`)

```text
<system-reminder>
# Plan Mode - System Reminder

CRITICAL: Plan mode ACTIVE - you are in READ-ONLY phase. STRICTLY FORBIDDEN:
ANY file edits, modifications, or system changes...

## Responsibility
Your current responsibility is to think, read, search, and delegate explore agents 
to construct a well-formed plan...
</system-reminder>
```

消息历史：

```
Session: session_01JPLAN001
消息历史（发送给 LLM）:
└── Message[user] id=message_01PLAN001
    ├── TextPart: "实现一个用户认证系统"
    └── TextPart (synthetic): "<system-reminder>Plan mode ACTIVE..."
```



#### 步骤 4：LLM 处理（Plan Agent）

```typescript
const processor = SessionProcessor.create({
  assistantMessage: await Session.updateMessage({
    id: "message_01PLAN002",
    role: "assistant",
    agent: "plan",  // ← 记录是 plan agent 的响应
    // ...
  }),
  // ...
})

// 解析工具 - plan agent 的权限限制了大部分写入工具
const tools = await resolveTools({
  agent,  // plan agent
  session,
  model,
  // ...
})
// tools 包含: read, grep, glob, task (但 edit/write 被限制)
```

消息历史：

```
Session: session_01JPLAN001
消息历史:
├── Message[user] id=message_01PLAN001
│   │ agent: "plan"
│   └── TextPart: "实现一个用户认证系统"
│
└── Message[assistant] id=message_01PLAN002
    │ agent: "plan"
    └── (暂无 parts)
```



#### 步骤 5：Plan Agent 使用 explore 子任务

LLM（plan agent）决定派遣 explore agent 来了解代码库：

```
Session: session_01JPLAN001
消息历史:
├── Message[user] id=message_01PLAN001
│   └── TextPart: "实现一个用户认证系统"
│
└── Message[assistant] id=message_01PLAN002
    │ agent: "plan"
    ├── TextPart: "让我先了解一下项目的结构和现有的认证相关代码..."
    ├── ToolPart: tool="task"
    │   state: {
    │     status: "completed",
    │     input: {
    │       description: "探索认证相关代码",
    │       prompt: "查找项目中现有的认证、登录、用户相关代码",
    │       subagent_type: "explore"
    │     },
    │     output: "找到以下相关文件:\n- src/auth/index.ts\n- src/user/..."
    │   }
    └── TextPart: "基于探索结果，我将制定以下计划..."
```



#### 步骤 6：Plan Agent 创建计划文件

Plan agent 使用 [write](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 工具创建计划文件（这是 plan agent 唯一允许编辑的路径）：

```typescript
// plan agent 的权限配置
permission: [
  { permission: "edit", pattern: "*", action: "deny" },
  { permission: "edit", pattern: ".opencode/plans/*.md", action: "allow" },
]
```

消息历史：

```
Session: session_01JPLAN001
消息历史:
├── Message[user] id=message_01PLAN001
│   └── TextPart: "实现一个用户认证系统"
│
└── Message[assistant] id=message_01PLAN002
    │ agent: "plan"
    ├── TextPart: "让我先了解一下项目的结构..."
    ├── ToolPart: tool="task" (explore)
    ├── TextPart: "基于探索结果，我将制定以下计划..."
    └── ToolPart: tool="write"
        state: {
          status: "completed",
          input: {
            filePath: ".opencode/plans/1738500000-auth-system.md",
            content: "# 用户认证系统实现计划\n\n## 目标\n..."
          }
        }
```



#### 步骤 7：Plan Agent 调用 question 工具

Plan agent 可能会使用 [question](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 工具询问用户：

```
└── Message[assistant] id=message_01PLAN002
    │ agent: "plan"
    ├── ...之前的 parts...
    └── ToolPart: tool="question"
        state: {
          status: "completed",
          input: {
            question: "你希望使用哪种认证方式？",
            choices: [
              { value: "jwt", description: "JWT Token 认证" },
              { value: "session", description: "Session 认证" },
              { value: "oauth", description: "OAuth 2.0" }
            ]
          },
          output: "用户选择: jwt"
        }
```



#### 步骤 8：Plan Agent 完成，调用 plan_exit

当计划完成后，plan agent 调用 [plan_exit](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 工具，该工具会向用户确定是否接受此次规划，一旦接受，Agent 就从 `Plan` 模式 转换到 `Build` 模式，执行实际工作。

```
└── Message[assistant] id=message_01PLAN002
    │ agent: "plan"
    │ finish: "tool-calls" 或 "stop"
    ├── ...之前的 parts...
    └── ToolPart: tool="plan_exit"  (如果使用实验模式)
        或
    └── TextPart: "计划已完成，请查看 .opencode/plans/xxx.md"
```



#### 步骤 9：用户切换到 Build 模式

用户决定执行计划，切换到 [build](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) agent：

```typescript
prompt({
  sessionID: "session_01JPLAN001",
  agent: "build",  // ← 切换到 build
  parts: [{ type: "text", text: "开始执行计划" }],
})
```



#### 步骤 10：insertReminders 注入 Build 切换提示

```typescript
// insertReminders 中
const wasPlan = input.messages.some((msg) => 
  msg.info.role === "assistant" && msg.info.agent === "plan"
)
// wasPlan = true (之前有 plan agent 的响应)

if (wasPlan && input.agent.name === "build") {
  userMessage.parts.push({
    type: "text",
    text: BUILD_SWITCH + "\n\n" + `A plan file exists at ${plan}. You should execute on the plan defined within it`,  // build-switch.txt 内容
    synthetic: true,
  })
}
```

**BUILD_SWITCH 内容**（来自 [build-switch.txt](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)）：

```
<system-reminder>
Your operational mode has changed from plan to build.
You are no longer in read-only mode.
You are permitted to make file changes, run shell commands, and utilize your arsenal of tools as needed.
</system-reminder>
```

消息历史：

```
Session: session_01JPLAN001
消息历史:
├── Message[user] id=message_01PLAN001
│   │ agent: "plan"
│   └── TextPart: "实现一个用户认证系统"
│
├── Message[assistant] id=message_01PLAN002
│   │ agent: "plan"
│   ├── TextPart: "让我先了解一下项目的结构..."
│   ├── ToolPart: tool="task" (explore)
│   ├── ToolPart: tool="write" (创建计划文件)
│   └── TextPart: "计划已完成..."
│
└── Message[user] id=message_01PLAN003
    │ agent: "build"  ← 切换到 build
    ├── TextPart: "开始执行计划"
    └── TextPart (synthetic): "<system-reminder>Your operational mode has changed..."
```



#### 步骤 11：Build Agent 执行计划

现在 build agent 开始工作，它有完整的工具权限：

```
Session: session_01JPLAN001
消息历史:
├── ... (之前的消息) ...
│
├── Message[user] id=message_01PLAN003
│   │ agent: "build"
│   ├── TextPart: "开始执行计划"
│   └── TextPart (synthetic): "Your operational mode has changed..."
│
└── Message[assistant] id=message_01PLAN004
    │ agent: "build"
    ├── TextPart: "好的，我将按照计划开始实现用户认证系统..."
    ├── ToolPart: tool="read" (读取计划文件)
    ├── ToolPart: tool="write" (创建 src/auth/jwt.ts)
    ├── ToolPart: tool="write" (创建 src/auth/middleware.ts)
    ├── ToolPart: tool="bash" (运行测试)
    └── TextPart: "认证系统已实现完成！"
```





## Callbacks 的生命周期

`src/session/prompt.ts: 259-265`

这里后来的请求假如回调队列 callbacks 之后，callbacks 会在哪里被调用，在什么情况下会从 callbacks 队列中拿出回调执行？

### 1. **入队时机** [prompt.ts:259-265](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

```typescript
const abort = start(sessionID)
if (!abort) {
  // 循环已在运行，后续请求进入回调队列
  return new Promise<MessageV2.WithParts>((resolve, reject) => {
    const callbacks = state()[sessionID].callbacks
    callbacks.push({ resolve, reject })
  })
}
```



**当发生什么时入队：**

- 同一个 [sessionID](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 的**第二个及后续请求**调用 [loop()](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 时
- [start(sessionID)](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 返回 `undefined`（因为已有运行中的循环）
- 新请求被包装成 Promise，[resolve/reject](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 回调加入队列

------

### 2. **出队执行时机** [prompt.ts:625-633](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

循环执行完成，开始返回结果时被调用：

```typescript
// 循环体结束（while(true) 跳出）
SessionCompaction.prune({ sessionID })

// 逐个流式返回消息给所有回调
for await (const item of MessageV2.stream(sessionID)) {
  if (item.info.role === "user") continue
  
  // 关键：拿出队列中所有的回调
  const queued = state()[sessionID]?.callbacks ?? []
  
  // 对每个回调执行 resolve
  for (const q of queued) {
    q.resolve(item)  // ✅ 在这里调用
  }
  return item
}
```



------

### 3. **具体场景示例**

**场景：用户连续发送两个请求**

```text
时间轴：
┌─────────────────────────────────────────────────────┐
│ T0: Request 1 arrives → loop(sessionID)             │
│     ✓ start(sessionID) 创建新循环                    │
│     → 进入 while(true) 开始处理                      │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│ T1: Request 2 arrives → loop(sessionID) (同一sessionID)
│     ✗ start(sessionID) 返回 undefined             │
│     → 创建 Promise，回调入队：                      │
│        callbacks.push({                            │
│          resolve: req2.resolve,                   │
│          reject: req2.reject                      │
│        })                                          │
│     → return Promise (req2 等待)                    │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│ T2: Request 1 的 loop() 执行完成                     │
│     → 读取 MessageV2.stream(sessionID)              │
│     → 对每条消息流：                                │
│        const queued = state()[sessionID].callbacks  │
│        for (const q of queued) {                   │
│          q.resolve(item) ← req2 的 Promise resolve │
│        }                                           │
│     → req2 返回结果                                 │
└─────────────────────────────────────────────────────┘
```



------

### 4. **取消时机** [prompt.ts:247-252](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

如果循环被中止，回调也会被取消：

```typescript
export function cancel(sessionID: string) {
  const match = state()[sessionID]
  if (!match) return
  
  // 拒绝所有排队的回调
  for (const item of match.callbacks) {
    item.reject()  // ✅ 在取消时调用
  }
  
  delete s[sessionID]
}
```



**何时触发 cancel：**

- 用户点击停止/中止按钮
- [prompt.ts:277-279](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 执行（循环函数退出）
- 通过 [SessionPrompt.cancel(sessionID)](vscode-file://vscode-app/c:/Users/10147/AppData/Local/Programs/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) API

------

### 5. **关键设计要点**

| 阶段     | 操作                                 | 代码位置 |
| -------- | ------------------------------------ | -------- |
| **入队** | Request 检测循环已运行               | L262     |
| **等待** | Promise 挂起，等待 resolve           | L261     |
| **出队** | 循环完成，流式返回消息               | L630-631 |
| **执行** | 逐个调用 resolve，向每个请求返回结果 | L631     |
| **异常** | 循环中止时，调用 reject              | L250     |

**为什么设计成这样？**

- **序列化执行**：同一 sessionID 的多个请求不会并行运行循环
- **公平分配**：多个请求都能获取循环执行的结果
- **简化状态**：避免复杂的并发控制




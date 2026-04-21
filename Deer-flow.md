# Deer-Flow 项目详解

Deer-Flow 是字节开源的 Deep Research Agent 项目。目前打算基于这个项目进行二次开发。Deer-Flow 搭建的深度思考框架如下：

- 先理解用户需求，可以和用户多轮对话，引导用户阐述清楚自己的需求。
- 然后通过 Planner 将用户需求拆解为子任务
- 把子任务调度给不同的子Agent来具体执行（网页搜索、本地RAG、Coder编程）
- 最后汇总执行结果，生成研究报告

其中，Deer-Flow并没有完整实现子Agent部分，开发者可以自行扩展适合执行特定任务的子Agent，这也是该项目的扩展性所在。目前我打算将 `open code` 揉到里面，实现更强的能力。





## 源码深入解析

> 审计的源码版本， Commit ID: d4ab77de5c630855b13c735828c61dcc076294cd

Deer Flow 的核心是一个基于状态机（StateGraph）的 Agent 工作流。它定义了一组节点（Nodes）和它们之间的流转规则（Edges），通过共享的状态（State）来传递上下文。核心逻辑是基于 `LangGraph` 实现的。Agent 的 整体 Workflow 如下图所示：

```bash
这里是图 。。。。。
```

### 架构总览

**核心组件：**

- **入口 (Entry):** `src/workflow.py` - 负责初始化状态、处理澄清对话循环、启动图执行。
- **图构建 (Builder):** `src/graph/builder.py` - 定义了图的拓扑结构，包括节点注册和边连接。
- **节点逻辑 (Nodes):** `src/graph/nodes.py` - 包含每个 Agent 的具体业务逻辑和工具调用。
- **状态定义 (State):** `src/graph/types.py` - 贯穿整个生命周期的上下文数据结构。



### 核心执行逻辑分析

工作流的执行可以划分为五个主要阶段：**初始化与澄清** -> **背景调查** -> **规划** -> **执行循环** -> **报告生成**。

传入工作流的初始状态如下：

| 字段                            | 默认值      | 说明                                                         |
| ------------------------------- | ----------- | ------------------------------------------------------------ |
| locale                          | en-US       | 所属地区，确保 Agent 的输出语言与用户偏好一致，贯穿整个 Prompt 构建过程。(zh-CN) |
| research_topic                  | ""          | 用户最初输入的原始调研主题，如果为空，则为用户初始提问 or LLM 概括的主题 |
| clarified_research_topic        | ""          | 经过澄清阶段（Coordinator 与用户多轮对话）后确定的最终、完整的调研主题。 |
| observations                    | []          | 观察结果列表，存储了 Worker Agent（Researcher, Analyst, Coder）执行每个步骤后的具体输出结果。 |
| resources                       | []          | 用户上传的资源文件列表（用于 RAG）。提供给 Researcher Agent 进行本地文档检索。 |
| plan_iterations                 | 0           | 计划重生成的迭代次数。用于防止 Planner 在生成计划时陷入无限循环，达到最大次数后会强制进入报告生成阶段。 |
| current_plan                    | Plan \| str | 当前生成的执行计划。包含具体的步骤（Steps）以及每个步骤的类型（research/analysis/processing）和执行状态（step.execution_res） |
| final_report                    | ""          | 最终生成的报告内容。                                         |
| auto_accepted_plan              | False       | 是否自动接受计划。如果为 `True`，则跳过 `human_feedback` 节点的人工确认环节，直接开始执行。 |
| enable_background_investigation | False       | 是否启用背景调查。决定是否在规划前运行 `background_investigation_node`，进行额外的搜素 |
| enable_clarification            | False       | 是否启用澄清功能的总开关。决定 Coordinator 是直接通过，还是进入提问模式，与用户进行多轮对话，收集更清晰的研究需求 |
| clarification_rounds            | 0           | 当前已经进行的澄清对话轮次计数。用于与 `max_clarification_rounds` 比较，控制对话长度。 |
| clarification_history           | []          | 澄清阶段的对话历史记录。仅包含用户用于澄清需求的回复，不包含 LLM 的提问。 |
| max_clarification_rounds        | 3           | 允许的最大澄清轮次（默认为 3）。                             |



### 初始化与澄清阶段 (Coordinator)

**涉及节点：** `coordinator_node` **入口文件：** `src/workflow.py` & `src/graph/nodes.py`。 它的主要职责是与用户沟通，明确用户的需求，然后决定是继续向用户提问（澄清需求），还是将任务移交给规划器（Planner）开始研究，亦或是直接回复（如闲聊 或者 触发某些安全检查）。具体执行逻辑如下所示：

#### 未启用 Clarification

用户的需求很明确，不用再和用户讨论需求。这种情况会有以下两个处理方式：

1. 用户在闲谈，或者触发某些安全检查，直接结束。具体控制可以在`prompts`中体现。
2. 跳转到 `planner`

**相关代码**如下：

```python
# 先从 coordinato.zh_CN.md 中加载系统提示词模板
# 并把 state['messages'] 的历史对话附加到后面
messages = apply_prompt_template("coordinator", state, locale=state.get("locale", "en-US"))
messages.append(
    {
        "role": "system",
        "content": "Clarification is DISABLED. For research questions, use handoff_to_planner. For greetings or small talk, use direct_response. Do NOT ask clarifying questions.",
    }
)

# Bind both handoff_to_planner and direct_response tools
tools = [handoff_to_planner, direct_response]
response = (
    get_llm_by_type(AGENT_LLM_MAP["coordinator"])
    .bind_tools(tools)
    .invoke(messages)
)
```

这里会提供给 LLM 两个工具（`handoff_to_planner`, `direct_response`）。在 `coordinator.zh_CN.md` 的提示词模板中，明确规定 LLM 在`enable_clarification: false`的情况下必须要调用这两个工具之一。仔细分析这两个工具，发现是空壳函数，单纯是为了给 LLM 提供上下文。

```python
@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return

@tool
def direct_response(
    message: Annotated[str, "The response message to send directly to user."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Respond directly to user for greetings, small talk, or polite rejections. Do NOT use this for research questions - use handoff_to_planner instead."""
    return
```

【💡】值得注意的是，工具调用可以结构化 LLM 的响应，比如 LLM 调用了 `handoff_to_planner`，LLM 会根据上下文以及自己的理解填充工具的参数。我们就可以从 `response.tool_call` 中获取到 LLM 生成的结构化的参数（`locale`, `research_topic`）：

```json
"tool_calls": [
    {
        "args": {
            "locale": "zh-CN",
            "research_topic": "规划明天去北京的旅行计划"
        },
        "id": "chatcmpl-tool-af4e1661551e5286",
        "name": "handoff_to_planner",
        "type": "tool_call"
    }
]

for tool_call in response.tool_calls:
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

	if not enable_clarification and tool_args.get("research_topic"):
		research_topic = tool_args["research_topic"]
```

最终由 LLM 的工具调用情况决定后续的处理情况，如果调用 `handoff_to_planner` 则跳转到 `Planner` 节点，否则直接结束。



#### 启用 Clarification

让 LLM 判断当前用户的问题描述是否清晰，如果需要，通过与用户进行多轮对话来获取对问题的充分阐述。LLM 会将`用户`的进一步阐述保存在 `clarification_history` 中，并从中总结出 `clarified_research_topic` ，供后续 `Planner` 使用。会通过 `max_clarification_rounds` 来控制最大对话轮数。

**相关代码实现如下**：

首先根据用户对问题的进一步阐述（`clarification_history`）总结出 `clarified_research_topic`

- 如果`clarification_history`未空，则直接使用原始问题代替
- 如果 len(clarification_history) == 1，那么 topic 就选第一个元素
- 如果 > 1，那么 topic 就是将 clarification_history 中的所有元素揉在一起，具体组织方法如下面的代码

```python
head, *tail = clarification_history
clarified_string = f"{head} - {', '.join(tail)}"
```



然后构建 `system prompts`

- 从 `coordinator_zh_CN.md` 加载模板，并用 `state` 的参数填充模板中预制的变量。最后将历史对话（`state["messages"]`） 附加在后面

```python
messages = apply_prompt_template("coordinator", state, locale=state.get("locale", "en-US"))

# Add clarification status for first round
if clarification_rounds == 0:
    messages.append(
        {
            "role": "system",
            "content": "Clarification mode is ENABLED. Follow the 'Clarification Process' guidelines in your instructions.",
        }
    )
```

- 设置用户澄清上下文 `clarification_context`

```python
clarification_context = f"""Continuing clarification (round {clarification_rounds}/{max_clarification_rounds}):
            User's latest response: {current_response}
            Ask for remaining missing dimensions. Do NOT repeat questions or start new topics."""

messages.append({"role": "system", "content": clarification_context})
```



提供合适的工具，并让 LLM 引导用户进一步阐述自己要研究的问题，如果对话轮数已经达到上限，则直接移交给 `Planner`

```python
tools = [handoff_to_planner, handoff_after_clarification]
# Check if we've already reached max rounds
if clarification_rounds >= max_clarification_rounds:
    # Max rounds reached - force handoff by adding system instruction
    # Add system instruction to force handoff - let LLM choose the right tool
    messages.append(
        {
            "role": "system",
            "content": f"MAX ROUNDS REACHED. You MUST call handoff_after_clarification (not handoff_to_planner) with the appropriate locale based on the user's language and research_topic='{clarified_topic}'. Do not ask any more questions.",
        }
    )

response = (
    get_llm_by_type(AGENT_LLM_MAP["coordinator"])
    .bind_tools(tools)
    .invoke(messages)
)
```



让 LLM 判断用户对问题的描述是否足够充分：

- 如果不充分，将 LLM 的反馈（提问）追加到历史对话中，进一步引导用户补充说明清楚自己的问题。下面代码中的 `__interrupt__` 就是让 `coordinator` 节点抛出中断信号，表示需要进行人机交互，让用户根据 LLM 的回复进一步阐明自己的问题。

```python
# No tool calls - LLM is asking a clarifying question
if not response.tool_calls and response.content:
    # Continue clarification process
    clarification_rounds += 1
    
    # Deer-Flow 官方对于引导用户进一步阐明需求，支持的并不好，需要自己修改实现
    # 加一个类似于 human_feedback_node 的逻辑，引入人机交互
    return Command(
        update={
            "messages": [HumanMessage(content=response.content, name="coordinator")],
            "locale": locale,
            "research_topic": research_topic,
            "resources": configurable.resources,
            "clarification_rounds": clarification_rounds,
            "clarification_history": clarification_history,
            "clarified_research_topic": clarified_topic,
            "is_clarification_complete": False,
            "goto": goto,
            "__interrupt__": [("coordinator", response.content)],
        },
        goto=goto,
    )
```

- 如果充分，LLM 会计划调用 `handoff_after_clarification`，跳转到 `Planner` 节点。但如果配置了 `background_investigator`，就会先跳转到 `background_investigator` 节点进行初步的检索，以获取更丰富的信息。

```python
for tool_call in response.tool_calls:
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    
    if tool_name in ["handoff_to_planner", "handoff_after_clarification"]:
        goto = "planner"

# Apply background_investigation routing if enabled (unified logic)
if goto == "planner" and state.get("enable_background_investigation"):
    goto = "background_investigator"
    
return Command(
        update={
            "messages": messages,
            "locale": locale,
            "research_topic": research_topic,
            "clarified_research_topic": clarified_research_topic_value,
            "resources": configurable.resources,
            "clarification_rounds": clarification_rounds,
            "clarification_history": clarification_history,
            "is_clarification_complete": goto != "coordinator",
            "goto": goto,
        },
        goto=goto,
)
```



### 规划阶段 (Planner)

Planner 负责生成具体的任务步骤（Steps），是 **DeerFlow** 深度研究框架的大脑。它负责**理解用户需求**、**结合背景信息**、**制定或更新研究计划**，并决定接下来交给用户审查，还是确定是否收集到足够的信息支撑报告生成。

#### 上下文准备

主要处理用户输入、澄清后的主题（`clarified_research_topic`）以及背景调查结果 （`background_investigation`）。将这些上下文作为`system prompts`。

- 如果开启 `clarification` 功能，则使用用户澄清之后更详细的 topic 替代原始的用户输入，然后从 `planner_ch_CN.md` 中加载系统提示词模板，并使用 `modified_state` 中的参数填充预留的字段。

```python
# FIXME 这里存在 Bug，如果从 human_feedback_node 返回 planner，用户对计划的修改意见无法转达给 LLM
# FIXME LLM 依旧使用 clarified_research_topic 作为用户需求
if state.get("enable_clarification", False) and state.get(
        "clarified_research_topic"
    ):
        # Modify state to use clarified research topic instead of full conversation
        modified_state = state.copy()
        modified_state["messages"] = [
            {"role": "user", "content": state["clarified_research_topic"]}
        ]
        modified_state["research_topic"] = state["clarified_research_topic"]
        messages = apply_prompt_template("planner", modified_state, configurable, state.get("locale", "en-US"))
```

- 如果开启背景调查，则注入调查结果

```python
# 如果开启 background investigation 则注入调查结果
if state.get("enable_background_investigation") and state.get(
    "background_investigation_results"
):
    messages += [
        {
            "role": "user",
            "content": (
                "background investigation results of user query:\n"
                + state["background_investigation_results"]
                + "\n"
            ),
        }
    ]
```



#### LLM 调用

可以使用流式 `llm.stream(messages)` 或普通模式 `ll.involke(messages)` 与 LLM 进行交互。



#### 结果校验

`planner.zh_CN.md` 规定 LLM 必须输出 json 格式的数据，因此在收到 LLM 的响应之后，要对其内容和格式进行校验，主要校验以下内容：

- 验证 LLM 响应的 json 格式的有效性，如果无效，直接跳到 reporter
- 虽然 Prompt 中可能写了“如果信息不足请进行搜索”，但 LLM 有时会偷懒（Hallucination）或者过于自信。`validate_and_fix_plan` 函数会检查 steps 列表。
  - 如果有缺失的 `step_type` 字段，根据是否 `need_search` 来填充缺省值。如果需要搜索，则设置为`research`类型，反之为 `analysis` 类型。注意(Issue #677: not all processing needs code)
  - 如果配置要求 `enforce_web_search=True`，但 LLM 生成的计划中所有步骤都标记为 `need_search=False`，代码会强行修改第一个步骤，将其改为搜索步骤。这弥补了 Prompt 约束力的不足。

plan.step 字段描述如下：

```python
class StepType(str, Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    PROCESSING = "processing" # 其实就是 coding

class Step(BaseModel):
    need_search: bool = Field(..., description="Must be explicitly set for each step")
    title: str
    description: str = Field(..., description="Specify exactly what data to collect")
    step_type: StepType = Field(..., description="Indicates the nature of the step")
    execution_res: Optional[str] = Field(
        default=None, description="The Step execution result"
    )
```



#### 下一步跳转

在跳转前，会将 LLM Plan出来的详细内容追加到历史对话中，作为下一个节点的上下文。

如果计划已经完善（LLM 将 json 响应的 `has_enough_context` 字段设置为 True），那么就直接跳转到 `reporter` 节点

```python
if isinstance(curr_plan, dict) and curr_plan.get("has_enough_context"):
    logger.info("Planner response has enough context.")
    new_plan = Plan.model_validate(curr_plan)
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": new_plan,
            **preserve_state_meta_fields(state),
        },
        goto="reporter",
    )
```

其他情况，需要人工确认计划是否完善：

```python
return Command(
    update={
        "messages": [AIMessage(content=full_response, name="planner")],
        "current_plan": full_response,
        **preserve_state_meta_fields(state),
    },
    goto="human_feedback",
)
```



### 人类反馈 (Human_feedback)

主要是审查 LLM 给出的研究 Plan 是否合理，如果接受，则跳转到`research_team`节点；如果需要修改，则提出要求并返回 `Planner` 节点进一步处理。

```python
feedback = interrupt("Please Review the Plan.")

# if the feedback is not accepted, return the planner node
if feedback_normalized.startswith("[EDIT_PLAN]"):
    logger.info(f"Plan edit requested by user: {feedback}")
    return Command(
        update={
            "messages": [
                HumanMessage(content=feedback, name="feedback"),
            ],
            **preserve_state_meta_fields(state),
        },
        goto="planner",
    )
elif feedback_normalized.startswith("[ACCEPTED]"):
    logger.info("Plan is accepted by user.")
```



### 研究团队 (research team)

负责将具体的研究步骤分配给不同的 `subagent` 执行。具体调度函数如下所示。主要是根据 `step.step_type` 来将规划出来的子任务分配给不同的`subagent`执行。

```python
def continue_to_running_research_team(state: State):
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return "planner"

    if all(step.execution_res for step in current_plan.steps):
        return "planner"

    # Find first incomplete step
    incomplete_step = None
    for step in current_plan.steps:
        if not step.execution_res:
            incomplete_step = step
            break

    if not incomplete_step:
        return "planner"

    if incomplete_step.step_type == StepType.RESEARCH:
        return "researcher"
    if incomplete_step.step_type == StepType.ANALYSIS:
        return "analyst"
    if incomplete_step.step_type == StepType.PROCESSING:
        return "coder"
    return "planner"
```



### 具体执行任务的子Agent (researcher)

这里以负责执行网络搜索/爬虫的子Agent为例子。首先初始化搜索引擎/爬虫工具，然后调用 Langchain 接口创建 `ReAct` 范式的 Agent，最后将执行结果保在共享状态 `State` 中的 `observations` 字段。核心任务执行函数是 `_execute_agent_step`. 

#### 配置可用工具

首先会根据配置添加可用的 本地RAG/搜索引擎/爬虫工具：

```python
if configurable.enable_web_search:
        tools.extend([get_web_search_tool(configurable.max_search_results), crawl_tool])
    else:
        logger.info("[researcher_node] Web search is disabled, using only local RAG")

# Add retriever tool if resources are available (always add, higher priority)
retriever_tool = get_retriever_tool(state.get("resources", []))

# 优先考虑本地 RAG 工具
tools.insert(0, retriever_tool)
```



`Deer-Flow` 支持多种 本地RAG/搜索引擎/爬虫工具 的实现，大部分是通过继承 `langchain_core.tools.BaseTool` 类实现的，爬虫是通过装饰器`@tool`实现的。工具的名字分别如下：

- 本地RAG：`local_search_tool`

```python
class RetrieverTool(BaseTool):
    name: str = "local_search_tool"
    description: str = "Useful for retrieving information from the file with `rag://` uri prefix, it should be higher priority than the web search or writing code. Input should be a search keywords."
    args_schema: Type[BaseModel] = RetrieverInput

    retriever: Retriever = Field(default_factory=Retriever)
    resources: list[Resource] = Field(default_factory=list)
```

- 搜索引擎: `web_search`

```python
elif SELECTED_SEARCH_ENGINE == SearchEngine.INFOQUEST.value:
    time_range = search_config.get("time_range", -1)
    site = search_config.get("site", "")
    logger.info(
        f"InfoQuest search configuration loaded: time_range={time_range}, site={site}"
    )
    return LoggedInfoQuestSearch(
        name="web_search",
        time_range=time_range,
        site=site,
    )
```

- 爬虫: `crawl_tool`

```python
@tool
@log_io
def crawl_tool(
    url: Annotated[str, "The url to crawl."],
) -> str:
    """Use this to crawl a url and get a readable content in markdown format."""
```



#### 创建 ReAct 范式的 Agent

调用 langchain 的接口创建 ReAct 范式的 Agent：`langchain.agents.create_agent`，同时考虑 MCP 服务器、Agent 中间件。Agent 中间件其实就是在和 LLM 交互前后插桩，实现提示词的压缩以及对输出的处理。

- MCP 配置，【💡】这里后续可以替换为 Skills. 

```python
# Add MCP tools to loaded tools if MCP servers are configured
client = MultiServerMCPClient(mcp_servers)
all_tools = await client.get_tools()
for tool in all_tools:
    if tool.name in enabled_tools:
        tool.description = (
            f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
        )
        loaded_tools.append(tool)
```

- Agent 中间件定义（`langchain.agents.middleware.AgentMiddleware`），主要是继承 `AgentMiddleware` 类实现的，重写该类的 `before_model` 函数，对输入的上下文进行 `压缩`/`裁剪`。下面的`pre_model_hook` 函数负责压缩上下文：1. 控制对话历史不超过token上限；2. 仅保留前3轮对话。

```python
from langchain.agents.middleware import AgentMiddleware
from langchain.agents import create_agent as langchain_create_agent

llm_token_limit = get_llm_token_limit_by_type(AGENT_LLM_MAP[agent_type])
pre_model_hook = partial(ContextManager(llm_token_limit, 3).compress_messages)

class PreModelHookMiddleware(AgentMiddleware):
	"""Middleware to execute a pre-model hook before model invocation.
    This middleware wraps the legacy pre_model_hook callable and executes it
    as part of the middleware chain.
    """
    def __init__(self, pre_model_hook: Callable):
        self._pre_model_hook = pre_model_hook
    
    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Execute the pre-model hook."""
        if not self._pre_model_hook:
            return None
        
        try:
            result = self._pre_model_hook(state, runtime)
            return result
        except Exception as e:
            logger.error(
                f"Pre-model hook execution failed in before_model: {e}",
                exc_info=True
            )
            return None
        
# Langchain 默认会创建 ReAct 模式的 Agent，会循环调用工具，直至满足要求
# Langchain 支持为 Agent 自定义中间件 (middleware)，以便于在LLM调用前后进行插装，执行自定义的逻辑
# 比如做提示词预处理等
agent = langchain_create_agent(
    name=agent_name,
    model=get_llm_by_type(llm_type),
    tools=loaded_tools,
    middleware=middleware,
)
```



#### 子任务执行 (_execute_agent_step)

`_execute_agent_step` 是具体执行子任务的核心函数。负责驱动具体的智能体（如 Researcher, Coder, Analyst）执行计划中的单个步骤。它封装了以下核心功能：

- 上下文构建 / 压缩
- 智能体调用
- 错误处理
- 结果验证以
- LangGraph 状态更新



##### 上下文准备

将已经执行完毕的步骤的执行结果作为 上下文 输入给 LLM。上下文主要包含：

- Research Topic
- 已经完成的步骤结果 (step.title + step.execution_res)
- 当前待执行步骤的说明

```python
completed_steps_info = "# Completed Research Steps\n\n"
for i, step in enumerate(completed_steps):
    completed_steps_info += f"## Completed Step {i + 1}: {step.title}\n\n"
    completed_steps_info += f"<finding>\n{step.execution_res}\n</finding>\n\n"

agent_input = {
        "messages": [
            HumanMessage(
                content=f"# Research Topic\n\n{plan_title}
                \n\n{completed_steps_info}
                # Current Step\n\n
                ## Title\n\n{current_step.title}\n\n
                ## Description\n\n{current_step.description}\n\n
                ## Locale\n\n{state.get('locale', 'en-US')}"
            )
        ]
    }
```



如果配置了本地 RAG，则添加提示词来强制 LLM 先从本地 RAG 中获取信息。如果要执行网络搜索，还需要额外添加系统提示词来强制 LLM 调用搜索工具，避免产生幻觉，比如虚构 URL

```python
resources_info = "**The user mentioned the following resource files:**\n\n"
for resource in state.get("resources"):
    resources_info += f"- {resource.title} ({resource.description})\n"
    
agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )
```



##### 验证输入上下文

调用 `validate_message_content` 函数验证 LLM 的输入，验证的内容包含如下几个方面：

1. 所有消息均包含`content`字段
2. 除合法空响应外，消息内容不得为空或空字符串
3. 复杂对象（列表、字典）将转换为JSON字符串
4. 内容过长时将截断以防止令牌溢出，这里截断用的是上面提到的 `pre_model_hook`。其实在创建 Agent 时，已经将其作为 Middleware 加载。这里为了保险起见又显示执行了一遍截断，其实没有必要。

##### 执行

实际执行工具的 Agent 是通过 Langchain 的 create_agent 接口创建的，模式是 ReAct 模式。会基于 Observations 循环调用工具，直至能够解决问题或达到最大递归次数。Deer-Flow 定义 `recursion_limit` 来控制 LangGraph 中最大能游走的节点个数。

##### 错误处理

尽管 agent 是 Langchain 封装好的 ReAct 模式，但在执行过程中仍可能出现意想不到的问题。因此使用 try except 来捕获执行过程中可能的错误。当出现错误时，会将错误保存在两个地方，并跳转回 `research_team` 

- 新增一条 HumanMessage 来保存抛出的异常，并附加在历史对话后面。
- 追加到 State 的 observations 字段

```python
try:
    result = await agent.ainvoke(
        input=agent_input, config={"recursion_limit": recursion_limit}
    )
except Exception as e:
    import traceback

    detailed_error = f"[ERROR] {agent_name.capitalize()} Agent Error\n\nStep: {current_step.title}\n\nError Details:\n{str(e)}\n\nPlease check the logs for more information."
    current_step.execution_res = detailed_error
    
    return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=detailed_error,
                        name=agent_name,
                    )
                ],
                "observations": observations + [detailed_error],
                **preserve_state_meta_fields(state),
            },
            goto="research_team",
        )
```



##### 执行后的验证

如果赋予Agent 联网调用的功能，则在这里验证是否真正检索网页，而非虚构URL产生幻觉。检查函数为 `validate_web_search_usage`。Langchain 封装好的 Agent 会自动执行具体的工具调用，从而产生 `ToolMessage`，因此在 response 中会出现`HumanMessage`、`AIMessage`、`ToolMessage`。该函数的检查逻辑是遍历所有返回的历史消息，判断相关消息是否真正调用了搜索引擎`web_search`。如果没有搜索，就产生告警，放在 `observations` 中。

【💡】检查逻辑有点简单，只要有一条消息包含 `web_search`，就返回 True。没有深入校验是否所有该搜索的请求都搜索了。

```python
for message in messages:
        # Check for ToolMessage instances indicating web search was used
        if isinstance(message, ToolMessage) and message.name == "web_search":
            web_search_used = True
            logger.info(f"[VALIDATION] {agent_name} received ToolMessage from web_search tool")
            break
            
        # Check for AIMessage content that mentions tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get('name') == "web_search":
                    web_search_used = True
                    logger.info(f"[VALIDATION] {agent_name} called web_search tool")
                    break
            # break outer loop if web search was used
            if web_search_used:
                break
                    
        # Check for message name attribute
        if hasattr(message, 'name') and message.name == "web_search":
            web_search_used = True
            logger.info(f"[VALIDATION] {agent_name} used web_search tool")
            break
            
if not web_search_used:
    # Add validation information to observations
    validation_info = (
        "\n\n[WARNING] This research was completed without using the web_search tool. "
        "Please verify that the information provided is accurate and up-to-date."
        "\n\n[VALIDATION WARNING] Researcher did not use the web_search tool as recommended."
    )

```



##### 处理观察结果

首先将最新的调用情况（`result["messages"][-1]`）保存在 `current_step.execution_res` 和 `response_content` 中。

然后将工具的调用结果（主要是ToolMessages）追加到 State 的 observations 字段，并将所有的执行历史`result["messages"]`追加到全局对话历史中`state["messages"]。`但其实 `result["messages"]` 中已经包含了大部分 observations，除了 validation_info

```python
response_content = result["messages"][-1]

return Command(
        update={
            "messages": agent_messages,
            "observations": observations + [response_content + validation_info],
            **preserve_state_meta_fields(state),
        },
        goto="research_team",
    )
```



### 报告总结 (Reporter)

根据当前计划的描述（`current_plan`），以及具体的执行情况，即 `state["observations"]` ，给出详尽的报告。

#### 上下文准备

- `reporter.zh_CN.md` 提供的系统提示词模板
- 当前计划描述（`input_`）
- 所有执行步骤的具体执行情况（`observation_messages`）

```python
current_plan = state.get("current_plan")

input_ = {
    "messages": [
        HumanMessage(
            f"# Research Requirements\n\n
            ## Task\n\n{current_plan.title}
            \n\n
            ## Description\n\n{current_plan.thought}"
        )
    ],
    "locale": state.get("locale", "en-US"),
}

invoke_messages = apply_prompt_template("reporter", input_, configurable, input_.get("locale", "en-US"))

# 将所有执行步骤中的工具调用结果 (observations) 加入到提示词中
observation_messages = []
for observation in observations:
    observation_messages.append(
        HumanMessage(
            content=f"Below are some observations for the research task:\n\n{observation}",
            name="observation",
        )
    )
```



#### 观察结果压缩

对 `observation_messages` 进行压缩，和之前提到压缩方案一致。

最后将准备好的上下文输入给 LLM ，等待生成最终报告。





## 动态调试 Deer Flow

我通过动态调试一步一步理解此项目。主要分成两个部分：

- 核心 Agent 的 WorkFlow
- 原始代码逻辑

### Agent WorkFlow Graph 可视化

首先配置项目根目录的 `langgraph.json`，Langgraph-cli 可以通过这个配置文件在运行时追踪 Graph 节点之间的数据流和控制流。具体配置如下，更详细的说明在 [langgraph-cli](https://docs.langchain.com/oss/python/langgraph/studio) 

```json
{
  "dockerfile_lines": [],
  "graphs": {
    "deep_research": "./src/workflow.py:graph",
    "podcast_generation": "./src/podcast/graph/builder.py:workflow",
    "ppt_generation": "./src/ppt/graph/builder.py:workflow"
  },
  "python_version": "3.12",
  "env": "./.env",
  "dependencies": ["."]
}

```



注意在 `.env` 中配置 `LANGSMITH_API_KEY`，需要得到 LangSmith 的支持才能可视化执行流。然后通过 `langgraph dev` 可视化调试 LangGraph 框架：

```bash
#!/bin/bash
DEBUG_MODE=true uv run langgraph dev
```



### 原始代码逻辑	

通过 pydebugger 动态调试整个程序的执行逻辑。具体通过 `uv add --dev debugpy` 安装 python 调试器。

`uv` 可以很方便在项目根目录创建虚拟环境，并且避免了显示`activate` 或 `deactivate` 激活/退出 虚拟环境。

- `uv run xxx` 可以在当前根目录下的虚拟环境中执行 python 命令。
- `uv add --dev xxx` 可以在当前根目录的虚拟环境中安装指定的依赖，同时会更新根目录下的`pyproject.toml` 和 `uv.lock`，以便于通过`uv sync` 快速同步和迁移虚拟环境。
- `uv remove --dev xxx` 可以删除特定以来，同时更新 `pyproject.toml` 和 `uv.lock`
- `uv sync` 会根据`uv.lock` 或 `pyproject.toml` 中列出的依赖库来在当前目录下（.venv）同步虚拟环境



在主程序入口点 (`src.workflow`) 插入如下代码：

```python
import os
# 仅在环境变量设置了 DEBUG 时启用，避免生产环境卡死
if os.getenv("DEBUG_MODE"):
    import debugpy
    print("⏳ 等待调试器连接 (端口 5678)...")
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()
    print("✅ 调试器已连接！")
```



然后配置 vscode , 在左边选型卡选择 `Run and Debug` 并设置新的远程调试：（Run and Debug -> Remote Attach -> create launch.json）。然后 vscode 会引导连接到远程调试的IP和端口，并自动生成`.vscode/launch.json`，之后就可以愉快的调试了：

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "10.26.85.44",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        }
    ]
}
```











Locale 选择 en-US 或 zh-CN





# 新体会



工具调用有时候只是让 LLM 的响应根据有结构化，比如定义下面的空工具。工具本身并不会干实际的工作，但是 LLM 会根据上下文以及自己的理解填充工具的参数（`research_topic` 和 `locale`），这样我们就能拿到结构化的数据。

```python
@tool
def handoff_to_planner(
    research_topic: Annotated[str, "The topic of the research task to be handed off."],
    locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    return
```

除此之外，也可以在提示词中让 LLM 处理结构化变量。具体来说，在实现 Agent 的过程中，一般都是通过 API 调用 LLM，这个时候，LLM 的输入就会被结构化成 `json`，我们可以在 `system prompts` 中引用结构化字段（比如 `tools` 中定义的工具）。比如`coordinator.zh_CN.md`。注意前提是输入给 LLM 的 json 中有定义这个键值，或者在历史消息中出现过相关的定义。

```markdown
3. **澄清过程（当`Clarification mode is ENABLED`时）**

# 工具调用要求

**关键**：如需调用工具，你必须调用可用工具之一。这是强制性的：
- 对于问候或闲聊：使用`direct_response()`工具
- 对于礼貌拒绝：使用`direct_response()`工具
- 对于研究问题：使用`handoff_to_planner()`或`handoff_after_clarification()`工具
- 对于澄清过程，可以不调用工具，引导用户需要进一步澄清哪些维度。
```

- `Clarification mode is ENABLED` 在历史 system 消息中出现过：

```python
# Add clarification status for first round
if clarification_rounds == 0:
    messages.append(
        {
            "role": "system",
            "content": "Clarification mode is ENABLED. Follow the 'Clarification Process' guidelines in your instructions.",
        }
    )
```

- 而涉及的工具调用在输入的 json 中有定义（`tools`字段包含）




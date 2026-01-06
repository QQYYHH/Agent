# Agent

是由LLM驱动的智能体。LLM负责解析用户的意图，并将结果映射到外部APP的API上，实现LLM和外部APP的交互。有助于推动APP的自动化操作和执行。本质上，他是一个循环机制：思考、行动、观察结果、再次行动，直到达成目标。

Agent主要流程：



1. 用户提出需求
2. LLM进行规划（Plan），生成一步一步的解决方案（有点类似于CoT思维链），选择外部工具并生成API的调用动作（符合MCP格式）。其实，规划本身不一定基于LLM的内部知识，它也可以利用外部工具，例如借助知识图谱，做进一步的推理。
3. 交给外部环境执行，并收集执行/反馈结果，将结果交给LLM分析。同时记录执行的历史结果（直接记录上下文历史对话，形成短期记忆；或 通过 RAG 优化长期记忆）
4. LLM给出下一步动作（修改当前子任务，或记录中间结果然后执行下一个子任务），直至需求完成。因为下一步动作可能有多种，而且不一定都执行成功，所以通常借助思维树来维护整个状态，以便于 LLM 尝试不同的方案。如果当前所有方案都行不通，那么很可能是前面的方案出现了问题，及时进行回溯即可。



Agent 与 传统自动化工作流的区别？传统自动化工作流是靠规则驱动的，也就是说，我们要事先知道完成一个任务所需要的具体流程，整个过程是静态的，只能在规定好的范围内执行自动化的操作。而 Agent 是 LLM 驱动的，因此它可以让 LLM 根据任务目标自动推理出所需的工作流程，然后一步一步执行。并且可以根据具体的执行结果动态调整工作流程，有较强的动态适应性和泛化能力。



## MCP

**MCP（Model Context Protocol）** 是近年来大模型 Agent 系统中非常重要的协议式规范，专门用于统一 LLM 与外部工具通信的**“上下文交互格式”**，通常配合 Agent 使用，以实现**多轮推理、工具调度、结果整合**等任务。

简单来讲，每个外部应用都对应一个MCP服务端和客户端。客户端可以从服务端获取外部应用的介绍、可调用的接口列表、调用的输入输出格式等信息，然后发送给外部 LLM，让其推理出合适的调用方式。



## CoT

论文 ToT 中给出了 CoT 的形式化定义和解释，我觉得挺好的，通俗易懂。

CoT的核心思路是引入思考过程 $z_1, ..., z_n$ 作为输入x和输出 y 之间的桥梁，使得 LLM 更容易理解复杂问题，如数学问题，逻辑问题等。其中每个 $z_i$ 都是连贯的token序列，可以是短语、句子或者段落。

形式化定义如下：
$$
z_i \sim p_{\theta}(z_i | x, z_{1...i-1})
$$
其中，$x$ 表示`初始提示词 + 当前的部分解决方案`，可以看出，$z_i$ 是以初始提示词和前 i-1 个思维步骤作为输入，从 LLM 中 采样/生成 出来的。注意，在初始提示词中，可以添加对输出思维链的约束条件，指定第一步干什么，第二步干什么，也就是 x 包含对 CoT 的设计。



## Self-consistency with CoT (CoT-SC)

是一种集成（ensemble）方法， 从 LLM 中采样 K 个独立同分布的思维链，最终通过投票选择置信度最高的答案。



## ToT

> 源码仓库 https://github.com/princeton-nlp/tree-of-thought-llm
>
> 该仓库不仅有实现源码，还包含相应的 prompts

Tree of Thought. 现有推理框架受限于基于 token 的从左到右的线性决策流程，因此提出树状推理过程，更符合人类的思维方式。这种想法源自于论文`Tree of Thoughts: Deliberate Problem Solving with Large Language Models`. 

ToT维护的树中每一个节点都代表一个推理状态 $s = [x, z_{1...i}]$ ，其中，$x$ 表示`初始提示词 + 当前的部分解决方案`，$z_i$ 表示第 $i$ 个思维步骤。

ToT 的核心组件如下：

### Thought generator

根据当前状态 $s$ ，生成 K 个下一步的候选 Thoughts，具体方法有两种：

1. 类似 CoT，根据当前状态 s 只生成下一步的1个thougth，采样 K 次得到 K 个thoughts. 这种方法适合下一步计划语义空间比较丰富的情况，例如下一步计划是一整段文字。

$$
z^{(j)} \sim p_{\theta}^{CoT}(z_{i+1} | s_i), j=1...k
$$

2. 直接根据当前状态生成下一步的 K 个候选 thoughts，BFS思想。这种方法适合 thoughts space 有限的情况，比如每个 thought 是一个单词（填字游戏）或一行方程（24点游戏）。

$$
[z^{(1)}, ..., z^{(k)}] \sim p_{\theta}(z_{i + 1}^{(1..k)} | s_i)
$$



### State evaluator

让 LLM 评估树的节点状态 s 的可行性。分两种方案：

1. 独立评估：单独评估每个状态，生成一个数值或分类，表示当前状态对于解决问题的进展程度。这种方法适合通过明确规则或模拟推理来评估的状态，一般是简单状态，例如 5 + 5 + 14 可以达到目标24
2. 投票评估：对多个状态进行比较和投票，选出对于解决问题最有贡献的状态。



### Search algorithm

BFS or DFS，一般用 DFS，因为更符合人类的思维方式。





## ReAct

比较经典的 Agent 模型，由论文 `REACT: SYNERGIZING REASONING AND ACTING IN  LANGUAGE MODELS` 提出。ReAct 的执行轨迹大概有两种：

1. Thought - Action - Observation. 这类定义比较死，每一步都是 `思考-行动-观察` 的链式行为，适合思考密集型任务
2. 上下文 - 动作。这类将 `实际执行外部工具` 和 `LLM思考/推理` 都作为动作，让大模型自己选择合适的时机进行思考/推理。基于推理/外部工具的调用结果更新当前上下文。然后让 LLM 根据当前最新上下文决定下一步动作（思考 or 调用工具）

具体不同类型任务的 执行过程/行动轨迹 可以参考这篇论文的 附录C

### 核心思想

把 Action Space 即动作空间扩展到语言空间。也就是说，`实际执行外部工具` 和 `LLM思考/推理` 都属于 Action. 当行动是思考或者推理时，不产生与外部环境的交互结果。需要注意的点如下：

1. LLM 会根据当前上下文 $c_t$ 判断后续是基于 $c_t$ 做推理，还是调用外部工具，以获取更多的信息。

2. 在 action 执行完之后，会根据执行结果 $o_t$ 或者 思考/推理 $thoughts$ 更新上下文 $c_t$. 形式化理解就是：$c_{t+1} = (c_t, o_t | thoughts)$. 

3. 初始上下文就是用户输入的原始 prompt

### 思考/推理 类型的 action 分类

这类行动分为很多种，下面是一些实例：

1. 分解任务目标并创建行动计划

2. 关联解决任务相关的内置常识知识

3. 从外部工具的执行结果中提取关键信息

4. 追踪任务进度并转换行动计划

5. 处理异常并调整行动计划



### 提示词设计

基于 few-shot prompting，给出一些任务执行的具体轨迹（可以参考论文的附录C）。原始论文通过实验验证，仅通过 1-6 个 few-shot 的样例就能让 LLM 在特定领域中工作，更多的样例不会改进LLM的性能。

### 微调方法

对小模型进行微调，使用 3000 条带正确答案的任务执行轨迹。使 LLM 能够在输入任务的提示下，输出（解码出）正确的执行轨迹（Thoughts, Action, Observation）



## Self-Refine

这篇论文源于 `SELF-REFINE:  Iterative Refinement with Self-Feedback`，首次引入反思机制，其核心思路如下：

1. Feedback: 接收 LLM 的原始输出，提出改进和反馈意见。
2. Refine: 将反馈意见 和 原始输出一并输入给 Refine 模型，给出修改后的输出。

上述两个过程会一直迭代进行至指定的次数，最终获取高质量的输出。采用迭代框架进行自我优化，通过自我评估自主提升生成质量。



但存在局限性，`Reflexion` 在文章中指出 `Self-Refine` 虽然有效，但仅适用于 single-generation reasoning tasks (单次生成推理任务)。所谓 single-generation reasoning tasks 指的是 LLM 在一次生成（generation）过程中完成推理与输出任务，无需经过一系列的中间“思考”过程，例如无需经过复杂的思维链，直接给出问题的答案。

虽然 `Self-Refine` 能不断反思输出的答案，但是缺少的中间思考过程可能会限制该方法在复杂任务上的推理能力。

此外，由于`Self-Refine` 没有记忆能力，因此在单个任务上累积的经验无法迁移到其他任务中，限制了模型的长期学习能力。





## Reflexion

> Reflexion: Language Agents with Verbal Reinforcement Learning
>
> https://github.com/noahshinn024/reflexion 源码仓库

<img width="1486" height="634" alt="reflexion" src="https://github.com/user-attachments/assets/cf81d59b-7671-409d-bea4-adfc7198256b" />

相比于 `ReAct`，多了中间的反思步骤(reflection)，具体流程如下：

该文章给出的 `ReAct` 模板为：

```python
REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""
```

其中 `scratchpad`中是每轮 `Thought, Action, Observation` 的拼接，相关代码如下：可以看到，`scratchpad = (T1, A1, O1), (T2, A2, O2), ...`，相当于保存 ReAct 的执行上下文。可以把 `scratchpad` 理解为是 LLM 的短期记忆，对历史行为轨迹（trajectory）的维护。

```python
def step(self) -> None:
    # Think
    self.scratchpad += f'\nThought {self.curr_step}:'
    self.scratchpad += ' ' + self.prompt_agent() # 调用外部 LLM
    print(self.scratchpad.split('\n')[-1])

    # Act
    self.scratchpad += f'\nAction {self.curr_step}:'
    action = self.prompt_agent() # 调用外部 LLM
    self.scratchpad += ' ' + action
    print(self.scratchpad.split('\n')[-1])

    # Observe
    self.scratchpad += f'\nObservation {self.curr_step}: '
    observation, self.reward, self.terminated, self.truncated, self.curr_step = self.env.step(action) # 与实际外部环境交互
    self.scratchpad += observation
    print(self.scratchpad.split('\n')[-1])
```

给出的 Reflexion 模板如下：

```python
REACT_REFLECT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""
```

可以发现，相比于 `ReAct`, 多了中间的 {reflections} 反思。具体反思是如何获得的，相关的反思模板如下：

```python
REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""
```

可以看到，是把`初始提示词 + ReAct` 作为 `previous trial / trajectory`，即先前的执行轨迹（短期记忆），喂给 LLM，让其根据目前的问题解决状态给出相应的反思。在拿到反思之后，填充到 `Reflexion 模板中`，喂给 LLM 得到下一轮的`ReAct`结果，如此反复循环，直到能够根据 `Observation` 推断出当前的任务已经完成。 



### Memory

`Reflexion` 额外引入了长短期记忆，以增强 LLM 在跨任务场景下的经验迁移能力：

1. 短期记忆，维护初始问题 + ReAct 产生的执行轨迹（trajectory），其实就是上面源码里面的 `scratchpad` 维护的内容。短期记忆可以提供具体的场景和执行情况。
2. 长期记忆：将所有的反思结果作为长期记忆。如下代码所示，在反思的过程中，会将反思结果加入全局数组 `reflections` 中，作为长期记忆。长期记忆提供在不同场景/情况下的`经验教训`。

```python
def reflect(self) -> None:
    self.reflections.append(self.prompt_reflection())
```

`Reflextion` 的核心就是长短期记忆的协同工作：短期记忆（行为轨迹）提供具体的场景和执行情况，而长期记忆则提供在不同场景下的`经验教训`。更抽象一点，长期记忆是跨会话之间的核心要点。





## Agent 优化

ReAct 模型，其实就是不断循环 Thought-Action-Observation 这个过程，但是这种链式处理过程，一旦某一步出错且LLM没有意识到，那么后续的步骤就会不断放大这个错误，最终导致整个任务执行失败。

您提出的这一点非常深刻，确实是ReAct（以及更早的Chain-of-Thought）这类线性推理链模式的核心缺陷。学术界将这个问题称为**“误差累积” (Error Accumulation) 或 “雪崩效应” (Snowballing Effect)**。

很快研究者们就发现了这个局限性，并提出了一系列“超越链式” (beyond-chain) 的设计模式。这些新模式的核心思想大体一致：**打破单一的、线性的“思考-行动”链条，引入更复杂的控制流，例如树状搜索（TOT）、自我反思（Reflexion）/修正循环（self-Refine）、或者多智能体协作。** 或者通过 contrastive reasoning 学习。（AVATAR）

# Toy Agent
基于 `LangGraph` 框架搭建简单的 Agent，用于调用计算函数求解表达式的值。
## Prepare

1. 按照 `vllm`，根据官网介绍，从源码按照。需要提前构建好 CUDA，Torch不着急安装，可以直接从`vllm`的源码构建出来。注意 GPU 仅支持 `Capability >= 7` 的硬件。

2. 基于 `vllm` 搭建 `Qwen3-8B`，搭建脚本如下：

``` bash
#!/bin/bash
# export VLLM_LOGGING_LEVEL=DEBUG
# --uvicorn-log-level info \
vllm serve /home/qyh/.cache/modelscope/hub/models/Qwen/Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
    --reasoning-parser qwen3 --tensor-parallel-size 2 \
    --max-model-len 20480 --max-num-seqs 2 \
    --enable-log-requests \
    --enable-log-outputs \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

3. 基于 `LangGraph` 搭建 Agent，源码如`toy_agent.py`所示。可以借助 `LangSmith` 追踪执行流，包括工作流的执行过程、LLM的输入输出。值得注意的是，该观测软件闭源，可以使用开源的`LangFuse`代替。

## Implement

`toy_agent.py`给出了具体实现，主要参考官方的示例：`https://docs.langchain.com/oss/python/langgraph/quickstart`. 把这个例子理解透彻花了我很长时间。

与 LLM 交互的部分主要是基于 `LangChain` 框架实现，比如定义各种消息：`HumanMessage`、`AIMessage`(Response). 下面是定义 `openai` 格式的 Chat 接口：

```python
from langchain.chat_models import init_chat_model

# 1. init model
model = init_chat_model(
    model="Qwen/Qwen3-8B", # 对应 vLLM 的参数 --served-model-name
    model_provider="openai", # langchain-openai
    base_url="http://10.26.85.44:8000/v1",
    api_key="EMPTY",
    temperature=0.6
)

# 2. Define tools，LangChain 提供了 @tool 装饰器对工具函数进行封装
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

# 3. Augment the LLM with tools
tools = [multiply]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
```



WorkFlow 的搭建基于 `LangGraph` 框架实现。整体来看，`LangGraph` 将工作流抽象成一张有向图，图之间的边可以是条件边（`add_conditional_edges`），也可以无条件跳转。比如下面代码，`should_continue` 决定具体跳转到哪个目标节点（`tool_node` or `END`），源节点是`llm_call`

```python
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
```

在工作流执行的过程中，`LangGraph`会维护一个全局的状态，比如这个官方这个例子中给出的`MessagesState`。注意该状态是全局共享的，工作流中的每个节点都可能会更新该状态的某个字段。`LangGraph`对状态中的字段更新方式进行了封装，我们可以通过注解（`Annotated`）的方式来指明某个字段的更新方式。在下面的例子中，我们定义`messages`的更新方式为`operator.add`，也就是新消息会不断累加到`messages`中（构成完整的历史对话信息）。默认情况下，是对字段直接覆盖（`llm_calls`）。其实在正常情况下，注解的内容需要显式通过代码获取到，下面的注释中我已经详细做了说明，而`LangGraph`对这个过程做了封装，因此不需要我们进行显式的代码调用。

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

# shared message 全局共享
class MessagesState(TypedDict):
    # Annotated 只是单纯的注解，不会主动触发
    # 通过 typing.get_type_hints(class, include_extras=True) 获取类成员的附加信息
    # 然后需要显式获取额外附加信息
    # hints = get_type_hints(MessageState)
    # hints: {'messages': typing.Annotated[list[int], <built-in function add>], 'llm_calls': <class 'int'>}
    # hints['messages'].__metadata__ : (<built-in function add>,)
    # op_add = hints['messages'].__metadata__[0]

    messages: Annotated[list[AnyMessage], operator.add] 
    llm_calls: int
```



### 动态节点生成

在如下图所反映的 `Orchestrator-worker` 工作流程下，`worker` 节点会根据编排节点（`Orchestrator`）与LLM的交互结果动态生成，这种模式在 LLM 驱动下的工作流中很常见。`LangGraph`给出了针对这种动态场景的解决方案，具体是用 [Send](https://docs.langchain.com/oss/python/langgraph/workflows-agents#orchestrator-worker) API 来实现。

<img width="661" height="235" alt="worker" src="https://github.com/user-attachments/assets/b5f34b60-9d64-420c-87e9-7a9ae45322f1" />

`Send` API 允许动态创建`worker`节点并向其发送特定输入。每个 `worker` 拥有`独立`状态，但是 `worker` 的`输出`均会修改`共享`状态。

```python
from langgraph.types import Send


# Shared Graph state
class State(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report


# Independent Worker state
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]
    

# Nodes: 基于 LLM 驱动后续的工作编排
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    # Generate queries
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}

# Nodes: 动态工作节点
def llm_call(state: WorkerState):
    pass
    
# Conditional edge function to create llm_call workers that each write a section of the report
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [Send("llm_call", {"section": s}) for s in state["sections"]]



```



## LLM行为观测

由于`LangGraph`对执行流进行了封装，因此不好获取原始与LLM交互的输入输出，即使通过 `LangSmith`，也只能拿到一部分数据，没法拿到完整的，最原始的输入输出。因此我使用 `tcpdump` 抓取了 `vllm` 所监听的端口的流量。原始流量数据文件为 `port8000.pcap`. 

结合 `toy_agent.py` 给出的示例，对流量的具体分析如下。

### 询问LLM并提供可用工具

如文件 `traffic_toy_tools_query1.json`所示，主要包含以下几个部分：

1. messages[0]: system prompt
2. messages[1]: human query: `Calculate (3 + 5) * 8`
3. tools description (`tools` 字段)，包含三个工具：`add`, `multiply`, `divide`

```json
{"messages":[{"content":"You are a helpful assistant tasked with performing arithmetic on a set of inputs.","role":"system"},{"content":"Calculate (3 + 5) * 8","role":"user"}],"model":"Qwen/Qwen3-8B","stream":false,"temperature":0.6,"tools":[{"type":"function","function":{"name":"add","description":"Adds `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}}},{"type":"function","function":{"name":"multiply","description":"Multiply `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}}},{"type":"function","function":{"name":"divide","description":"Divide `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}}}]}
```

### LLM 第一次回复

如文件 `traffic_toy_tools_response1.json` 所示，主要包含以下几个部分：

1. choices[0].message.`content` 为空，说明大部分都是 LLM 的思考过程
2. choices[0].message.`tool_calls` 中包含对两个工具的调用 `add(3, 5)` 以及 `multiply(8, 8)`. 
3. choices[0].message.`reasoning` 和 `reasoning_content` 中是 LLM 的推理过程
4. choices[0].`finish_reason`: tool_calls，表示当前正处于工具调用阶段，整个工作流还没有结束。

```json
{"id":"chatcmpl-b6eafacdb212ae45","object":"chat.completion","created":1767593888,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"message":{"role":"assistant","content":null,"refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[{"id":"chatcmpl-tool-9cfff31470c8d39b","type":"function","function":{"name":"add","arguments":"{\"a\": 3, \"b\": 5}"}},{"id":"chatcmpl-tool-afe2dd0e7aedad5f","type":"function","function":{"name":"multiply","arguments":"{\"a\": 8, \"b\": 8}"}}],"reasoning":"\nOkay, let's see. The user wants me to calculate (3 + 5) * 8. Hmm, first I need to figure out the order of operations here. The expression inside the parentheses is 3 + 5, which should be done first. So 3 plus 5 equals 8. Then, I multiply that result by 8. Wait, so that's 8 multiplied by 8. But wait, maybe I should check if the functions available can handle this.\n\nLooking at the tools provided, there's add, multiply, and divide functions. The add function takes two integers, so I can use that for 3 and 5. Then, the multiply function would take the result of that addition and multiply by 8. So first, call add with a=3 and b=5. That gives 8. Then, call multiply with a=8 (the result from add) and b=8. That should give 64. \n\nAlternatively, maybe there's a way to do it in one step, but the functions don't have a combined operation. So I need to break it down into two steps. First add 3 and 5, then multiply the sum by 8. Yep, that's the correct approach. Let me make sure I'm using the right parameters for each function. The add function's parameters are a and b, both integers. Multiply also takes a and b as integers. So the first tool call is add with 3 and 5, then multiply with the result and 8. Alright, that should work.\n","reasoning_content":"\nOkay, let's see. The user wants me to calculate (3 + 5) * 8. Hmm, first I need to figure out the order of operations here. The expression inside the parentheses is 3 + 5, which should be done first. So 3 plus 5 equals 8. Then, I multiply that result by 8. Wait, so that's 8 multiplied by 8. But wait, maybe I should check if the functions available can handle this.\n\nLooking at the tools provided, there's add, multiply, and divide functions. The add function takes two integers, so I can use that for 3 and 5. Then, the multiply function would take the result of that addition and multiply by 8. So first, call add with a=3 and b=5. That gives 8. Then, call multiply with a=8 (the result from add) and b=8. That should give 64. \n\nAlternatively, maybe there's a way to do it in one step, but the functions don't have a combined operation. So I need to break it down into two steps. First add 3 and 5, then multiply the sum by 8. Yep, that's the correct approach. Let me make sure I'm using the right parameters for each function. The add function's parameters are a and b, both integers. Multiply also takes a and b as integers. So the first tool call is add with 3 and 5, then multiply with the result and 8. Alright, that should work.\n"},"logprobs":null,"finish_reason":"tool_calls","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":377,"total_tokens":755,"completion_tokens":378,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```

### Agent调用本地工具并回复LLM

如文件 `traffic_toy_tools_query2.json` 所示，主要包含以下几个部分：

1. messages[0]: system prompt
2. messages[1]: human query
3. messages[2]: LLM 的第一次回复：`content`为空、工具调用步骤（`tool_calls`）。相比于完整回复，删掉了 `reasoning` 的内容。这是由 Agent 的上层逻辑控制的，也可以选择附带具体的推理内容。
4. messages[3]: 具体的工具调用结果，`LangChain`会将其封装为 `ToolMessage` 消息。注意 `ToolMessage.tool_call_id` 要和 LLM 给出的工具调用步骤（`tool_calls`）中的 `id` 一一对应。
5. tools description (`tools` 字段)

```json
{"messages":[{"content":"You are a helpful assistant tasked with performing arithmetic on a set of inputs.","role":"system"},{"content":"Calculate (3 + 5) * 8","role":"user"},{"content":null,"role":"assistant","tool_calls":[{"type":"function","id":"chatcmpl-tool-9cfff31470c8d39b","function":{"name":"add","arguments":"{\"a\": 3, \"b\": 5}"}},{"type":"function","id":"chatcmpl-tool-afe2dd0e7aedad5f","function":{"name":"multiply","arguments":"{\"a\": 8, \"b\": 8}"}}]},{"content":"8","role":"tool","tool_call_id":"chatcmpl-tool-9cfff31470c8d39b"},{"content":"64","role":"tool","tool_call_id":"chatcmpl-tool-afe2dd0e7aedad5f"}],"model":"Qwen/Qwen3-8B","stream":false,"temperature":0.6,"tools":[{"type":"function","function":{"name":"add","description":"Adds `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}}},{"type":"function","function":{"name":"multiply","description":"Multiply `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}}},{"type":"function","function":{"name":"divide","description":"Divide `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int","parameters":{"properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"],"type":"object"}}}]}
```



### LLM第二次回复

如文件 `traffic_toy_tools_response2.json` 所示，主要包含以下几个部分：

1. choices[0].message.`content`: 具体的回复内容，这里 LLM 直接给出了最终的表达式结果。
2. choices[0].message.`reasoning` 和 `reasoning_content` 给出了具体的思考内容。
3. choices[0].`finish_reason`: stop，表示 LLM 认为目前的工作流已经可以结束了。

```json
{"id":"chatcmpl-9e2c07f9773531b6","object":"chat.completion","created":1767593894,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"message":{"role":"assistant","content":"\n\nThe result of (3 + 5) * 8 is 64.","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":"\nOkay, let's see. The user asked to calculate (3 + 5) * 8. First, I need to handle the addition inside the parentheses. So I called the add function with a=3 and b=5. That gives 8. Then, I need to multiply that result by 8. Wait, the user wrote (3 + 5) * 8, so the second operand is 8. So after getting 8 from the addition, I should multiply by 8. But in my previous tool calls, I added 3+5 first, which is correct. Then the next step is to take the result of that addition (which is 8) and multiply it by 8. So the multiply function should have a=8 and b=8. That would give 64. The user's final answer should be 64. Let me check if the tool responses were correct. The add function returned 8, then the multiply function with 8 and 8 gives 64. Yep, that's right. So the final answer is 64.\n","reasoning_content":"\nOkay, let's see. The user asked to calculate (3 + 5) * 8. First, I need to handle the addition inside the parentheses. So I called the add function with a=3 and b=5. That gives 8. Then, I need to multiply that result by 8. Wait, the user wrote (3 + 5) * 8, so the second operand is 8. So after getting 8 from the addition, I should multiply by 8. But in my previous tool calls, I added 3+5 first, which is correct. Then the next step is to take the result of that addition (which is 8) and multiply it by 8. So the multiply function should have a=8 and b=8. That would give 64. The user's final answer should be 64. Let me check if the tool responses were correct. The add function returned 8, then the multiply function with 8 and 8 gives 64. Yep, that's right. So the final answer is 64.\n"},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":448,"total_tokens":697,"completion_tokens":249,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}


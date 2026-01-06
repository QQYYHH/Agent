from langchain.tools import tool
from langchain.chat_models import init_chat_model

# 1. init model
model = init_chat_model(
    model="Qwen/Qwen3-8B", # 对应 vLLM 的参数 --served-model-name
    model_provider="openai", # langchain-openai
    base_url="http://10.26.85.44:8000/v1",
    api_key="EMPTY",
    temperature=0.6
)

# 2. Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b

# 3. Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)


# 4. Define state
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

# 5. Define model node
from langchain.messages import SystemMessage

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# 6. Define tool node
from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# 7. Define end logic
from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# 8. Build and compile the agent
# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent graph structure
from IPython.display import Image, display
# img = Image(agent.get_graph(xray=True).draw_mermaid_png())
img = Image(agent.get_graph(xray=True).draw_png())

with open("toy_tools.png", "wb") as f:
    f.write(img.data)

# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Calculate (3 + 5) * 8")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
    # print(m)

    # TODO 看一下具体发送和响应的 prompt 是怎么样的
    # 感觉 LangGraph 封装的还是有些上层，看一下具体 prompt 的设计。
    # 使用 LangSmith 进行观测（闭源，专门适配LangChain）
    # 使用 LangFuse 进行观测（开源）

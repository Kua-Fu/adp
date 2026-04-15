"""
mm_8: 内存管理（Memory Management）示例

本示例演示两种记忆能力：
1) 短期记忆（Short-term Memory）：依赖 checkpointer，按 thread_id 记住同一会话上下文。
2) 长期记忆（Long-term Memory）：依赖 store，按 user_id 跨会话保存用户事实。

模型选择策略：
- 优先 OpenAI-compatible
- 回退 Gemini
"""

import os
import uuid
from typing import Annotated, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedStore, ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


# ------------------------- 环境加载 -------------------------
# 固定从当前脚本目录读取 .env，避免在不同 cwd 下执行时找不到配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


def _init_llm() -> Tuple[Optional[object], str]:
    """
    初始化 LLM（OpenAI-compatible 优先，Gemini 回退）。

    返回：
        (llm, status)
        - llm: 可调用的聊天模型对象，失败时为 None
        - status: 状态文本，用于终端排错
    """
    # ---------- 1) OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None, "Detected OPENAI_API_KEY, but `langchain-openai` is not installed."

        kwargs = {
            "model": openai_model,
            "temperature": openai_temperature,
            "api_key": openai_api_key,
        }
        if openai_base_url:
            kwargs["base_url"] = openai_base_url

        return ChatOpenAI(**kwargs), f"Using OpenAI-compatible backend: {openai_model}"

    # ---------- 2) Gemini（回退） ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    google_temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0"))

    if google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            return None, "Detected GOOGLE_API_KEY, but `langchain-google-genai` is not installed."

        llm = ChatGoogleGenerativeAI(model=google_model, temperature=google_temperature)
        return llm, f"Using Gemini backend: {google_model}"

    return (
        None,
        "No valid model credentials found. "
        "Please set OPENAI_API_KEY (preferred) or GOOGLE_API_KEY in mm_8/.env.",
    )


class State(TypedDict):
    """
    LangGraph 状态对象。

    字段说明：
    - messages: 会话消息列表；通过 add_messages 自动累加。
    - recall_memories: 本轮检索到的长期记忆文本列表（用于注入系统提示）。
    """

    messages: Annotated[list[AnyMessage], add_messages]
    recall_memories: list[str]


@tool
def save_recall_memory(
    memory: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    将“值得长期记住”的用户信息写入长期记忆库。

    典型内容：
    - 用户偏好（口味、写作风格、语言习惯）
    - 用户背景（职位、兴趣、项目上下文）

    注意：
    - 这里按 user_id 做命名空间隔离，避免不同用户记忆串线。
    """
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    namespace = ("memories", user_id)
    memory_id = str(uuid.uuid4())

    # value 使用 dict，后续可扩展更多字段（创建时间、来源、权重等）。
    store.put(namespace, memory_id, {"memory": memory})
    return f"Memory stored for user '{user_id}': {memory}"


def _search_recall_memories(
    store: BaseStore,
    user_id: str,
    query: str,
    limit: int = 5,
) -> list[str]:
    """
    从长期记忆库检索与当前 query 相关的记忆。

    说明：
    - 为兼容不同 store 能力，这里先做一次通用 search；
    - 若 store 不支持语义 query，则退化成“取前 N 条 + 关键词过滤”。
    """
    namespace = ("memories", user_id)

    items = []
    try:
        items = store.search(namespace, query=query, limit=limit)
    except Exception:
        try:
            items = store.search(namespace, limit=50)
        except Exception:
            return []

    memories: list[str] = []
    lowered = query.lower()

    for item in items:
        value = getattr(item, "value", None) or {}
        text = value.get("memory")
        if not text:
            continue

        # 若是“无 query 语义检索”的退化路径，用简单关键词做一次过滤。
        if query and any(token in text.lower() for token in lowered.split() if token.strip()):
            memories.append(text)
        elif not query:
            memories.append(text)

    # 如果过滤后为空，但确实有历史记忆，则返回前几条作为回忆兜底。
    if not memories:
        for item in items[:limit]:
            value = getattr(item, "value", None) or {}
            text = value.get("memory")
            if text:
                memories.append(text)

    return memories[:limit]


def call_model(state: State, config: RunnableConfig, store: BaseStore):
    """
    图节点：执行模型调用。

    流程：
    1) 根据当前用户问题检索长期记忆；
    2) 将检索结果写入系统提示词；
    3) 调用绑定了“存储记忆工具”的模型；
    4) 返回新消息与 recall_memories。
    """
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    # 取最后一条用户输入作为检索 query。
    last_user_text = ""
    if state.get("messages"):
        last_user_text = str(state["messages"][-1].content)

    recall_memories = _search_recall_memories(
        store=store,
        user_id=user_id,
        query=last_user_text,
        limit=5,
    )

    # 将回忆到的长期记忆拼进系统提示，让模型“带着记忆回答”。
    memory_block = "\n".join(f"- {m}" for m in recall_memories) if recall_memories else "- (暂无长期记忆)"

    system_prompt = (
        "你是一个具备内存管理能力的助理。\n"
        "请结合历史记忆回答问题。\n"
        "当你识别到用户给出了可长期保存的稳定信息（偏好/背景/长期目标）时，"
        "应调用工具 `save_recall_memory` 存储。\n\n"
        "当前用户的可用长期记忆：\n"
        f"{memory_block}"
    )

    # 把工具绑定到模型，让模型可在需要时发起工具调用。
    llm_with_tools = llm.bind_tools([save_recall_memory])

    # 注意：系统提示放在最前面，然后拼接当前会话消息。
    response = llm_with_tools.invoke([SystemMessage(content=system_prompt), *state["messages"]])

    return {
        "messages": [response],
        "recall_memories": recall_memories,
    }


def _build_graph():
    """
    构建并编译 LangGraph。

    结构：
    START -> call_model -> (若有工具调用) tools -> call_model -> ...
                              -> (无工具调用) 结束
    """
    graph_builder = StateGraph(State)

    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("tools", ToolNode([save_recall_memory]))

    graph_builder.add_edge(START, "call_model")
    graph_builder.add_conditional_edges("call_model", tools_condition)
    graph_builder.add_edge("tools", "call_model")

    checkpointer = MemorySaver()  # 短期记忆：线程级会话状态
    store = InMemoryStore()  # 长期记忆：跨线程用户记忆

    return graph_builder.compile(checkpointer=checkpointer, store=store)


def _run_turn(graph, user_text: str, thread_id: str, user_id: str) -> None:
    """
    执行单轮对话并打印结果。

    参数：
    - thread_id: 控制短期记忆（同线程上下文连续）
    - user_id: 控制长期记忆（跨线程共享）
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config=config)

    final_msg = result["messages"][-1]
    print(f"\n[thread={thread_id} user={user_id}] 用户: {user_text}")
    print(f"助手: {final_msg.content}")

    recalled = result.get("recall_memories", [])
    if recalled:
        print("本轮命中的长期记忆:")
        for idx, mem in enumerate(recalled, start=1):
            print(f"  {idx}. {mem}")


def main() -> None:
    """
    脚本入口：演示“跨轮次 + 跨线程”的记忆能力。

    演示步骤：
    1) 在 thread-A 里告诉助手你的偏好（模型应调用工具保存）；
    2) 在同一线程继续提问（短期记忆 + 长期记忆都可生效）；
    3) 切换到 thread-B 再问偏好（验证长期记忆跨线程生效）。
    """
    print("## Running mm_8 memory management demo ##")

    global llm  # 让 call_model 可以复用初始化后的模型对象
    llm, llm_status = _init_llm()
    if not llm:
        print(f"Error initializing language model: {llm_status}")
        print("\nSkipping execution due to LLM initialization failure.")
        return

    print(f"Language model initialized. {llm_status}")

    graph = _build_graph()

    # 同一个 user_id，表示是同一位用户；thread_id 可以变化。
    demo_user_id = os.getenv("MM_USER_ID", "user-001")

    # 第一段会话（thread-A）：写入并使用记忆
    _run_turn(graph, "你好，我叫小王，我喜欢川菜和徒步。", thread_id="thread-A", user_id=demo_user_id)
    _run_turn(graph, "你记得我的饮食和爱好吗？", thread_id="thread-A", user_id=demo_user_id)

    # 第二段会话（thread-B）：验证跨线程还能回忆到同一用户的长期记忆
    _run_turn(graph, "我们换了一个新会话，你还记得我喜欢什么吗？", thread_id="thread-B", user_id=demo_user_id)


if __name__ == "__main__":
    main()

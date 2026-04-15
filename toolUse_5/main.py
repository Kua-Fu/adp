"""
Tool Use 示例：
演示如何使用 LangChain Tool Calling Agent 让模型按需调用工具。

核心特性：
1) 定义一个 `search_information` 工具函数；
2) 通过 `create_tool_calling_agent` 组装智能体；
3) 使用异步并发同时处理多个查询；
4) 模型后端选择：OpenAI-compatible 优先，Gemini 回退。
"""

import os
import asyncio
from typing import List, Optional, Tuple

import nest_asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool

try:
    # 你给的“原始写法”在部分 LangChain 版本可直接使用。
    from langchain.agents import create_openai_tools_agent, AgentExecutor  # type: ignore
    _USE_LEGACY_AGENT_API = True
except ImportError:
    # LangChain 新版（如 1.2.x）已移除上述符号，回退到 create_agent。
    from langchain.agents import create_agent
    _USE_LEGACY_AGENT_API = False

    def create_openai_tools_agent(llm, tools, prompt):
        """
        新版兼容层：
        - 保留旧函数名与调用方式
        - 内部映射到 create_agent
        """
        system_prompt = "You are a helpful assistant."
        try:
            msgs = getattr(prompt, "messages", [])
            for m in msgs:
                if getattr(m, "type", "") == "system":
                    system_prompt = getattr(m, "prompt").template
                    break
        except Exception:
            pass

        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            debug=False,
        )

    class AgentExecutor:
        """
        新版兼容层：
        - 对齐旧 AgentExecutor(agent=..., tools=..., verbose=...) 构造
        - 对齐旧 ainvoke({"input": ...}) 返回 {"output": ...}
        """
        def __init__(self, agent, tools=None, verbose=False):
            self.agent = agent
            self.tools = tools or []
            self.verbose = verbose

        async def ainvoke(self, payload):
            query = payload.get("input", "")
            response = await asyncio.to_thread(
                self.agent.invoke,
                {"messages": [{"role": "user", "content": query}]},
            )
            messages = response.get("messages", [])
            final_output = messages[-1].content if messages else str(response)
            return {"output": final_output}


# ------------------------- 环境加载 -------------------------
# 固定从当前脚本目录读取 .env，避免在不同 cwd 下执行时读取失败。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


def _init_llm() -> Tuple[Optional[object], str]:
    """
    初始化工具调用模型（OpenAI-compatible 优先，Gemini 回退）。

    Returns:
        (llm, status)
        - llm: 可用于 tool-calling 的聊天模型实例；失败时为 None
        - status: 初始化状态文本，用于调试排错
    """
    # ---------- OpenAI-compatible ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return (
                None,
                "Detected OPENAI_API_KEY, but `langchain-openai` is not installed.",
            )

        kwargs = {
            "model": openai_model,
            "temperature": openai_temperature,
            "api_key": openai_api_key,
        }
        if openai_base_url:
            kwargs["base_url"] = openai_base_url

        llm = ChatOpenAI(**kwargs)
        return llm, f"Using OpenAI-compatible backend: {openai_model}"

    # ---------- Gemini ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
    google_temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0"))
    if google_api_key:
        llm = ChatGoogleGenerativeAI(model=google_model, temperature=google_temperature)
        return llm, f"Using Gemini backend: {google_model}"

    return (
        None,
        "No valid model credentials found. "
        "Please set OPENAI_API_KEY (preferred) or GOOGLE_API_KEY in toolUse_5/.env.",
    )


# ------------------------- 工具定义 -------------------------
# 使用 @tool 装饰器把普通 Python 函数注册为 LangChain 可调用工具。
@langchain_tool
def search_information(query: str) -> str:
    """
    提供简单事实信息查询（演示用本地知识库）。
    当用户询问天气、国家首都、地球人口、最高山峰等事实类问题时，应优先调用本工具。
    推荐 query 使用简短英文小写短语，例如：
    - weather in london
    - capital of france
    - population of earth
    - tallest mountain
    """
    print(f"\n--- TOOL CALLED: search_information(query='{query}') ---")

    simulated_results = {
        "weather in london": "The weather in London is currently cloudy with a temperature of 15C.",
        "capital of france": "The capital of France is Paris.",
        "population of earth": "The estimated population of Earth is around 8 billion people.",
        "tallest mountain": "Mount Everest is the tallest mountain above sea level.",
        "default": f"Simulated search result for '{query}': No specific information found, but the topic seems interesting.",
    }

    # 查询时统一转小写，避免大小写导致命中失败。
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- TOOL RESULT: {result} ---")
    return result


tools = [search_information]


# ------------------------- 构建 Agent -------------------------
llm = None
llm_status = ""
agent_executor = None

try:
    llm, llm_status = _init_llm()
    if llm:
        print(f"Language model initialized. {llm_status}")

        # 该提示词模板包含 `agent_scratchpad` 占位符（旧版 Agent API 所需）。
        agent_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. "
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # 按你要的“原始结构”创建 agent + executor。
        agent = create_openai_tools_agent(llm, tools, agent_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    else:
        print(f"Error initializing language model: {llm_status}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None
    agent_executor = None


async def run_agent_with_tool(query: str) -> None:
    """
    执行单条查询：
    1) 调用 AgentExecutor
    2) 让 Agent 决定是否使用工具
    3) 打印最终回答
    """
    if not agent_executor:
        print(f"Skipping query due to initialization failure: {query}")
        return

    print(f"\n--- Running Agent with Query: '{query}' ---")
    try:
        response = await agent_executor.ainvoke({"input": query})
        final_output = response.get("output", str(response))
        print("\n--- Final Agent Response ---")
        print(final_output)
    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")


async def main() -> None:
    """
    并发执行多条查询，演示：
    - Agent 根据输入内容自动选择是否调用工具
    - 多请求并发处理
    """
    queries: List[str] = [
        "What is the capital of France?",
        "What's the weather like in London?",
        "Tell me something about dogs.",  # 可能触发默认结果或直接模型回答
    ]
    tasks = [run_agent_with_tool(q) for q in queries]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # 在 notebook / 交互环境里防止事件循环冲突；
    # 普通脚本执行也可安全调用。
    nest_asyncio.apply()
    asyncio.run(main())

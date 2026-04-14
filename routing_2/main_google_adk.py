# Copyright (c) 2025 Marco Fago
#
# This code is licensed under the MIT License.
# See the LICENSE file in the repository for the full license text.

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import InMemorySessionService, Runner
from google.adk.tools import FunctionTool
from google.genai.types import Content, Part

# ------------------------- 环境初始化 -------------------------
# 统一从当前脚本所在目录读取 .env，避免依赖“当前工作目录”导致的配置读取失败。
# 这样无论你在项目根目录还是 routing_2 目录执行脚本，都能读到同一个 .env。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ADK Runner 的应用名（可视为这次会话/应用的命名空间）。
APP_NAME = "routing_2_google_adk_app"


def _build_adk_model() -> tuple[Any | None, str]:
    """
    根据环境变量选择模型后端（优先 OpenAI-compatible，其次 Gemini）。

    支持两套变量：
    1) OpenAI-compatible:
       OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL(可选)
    2) Gemini:
       GOOGLE_API_KEY, GOOGLE_MODEL(可选)
    """
    # OpenAI-compatible 相关配置
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    # Gemini 相关配置
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

    # 优先使用 OpenAI-compatible 配置：
    # 这样可以兼容像 one-api / OpenRouter / 其他网关提供的模型。
    if openai_api_key and openai_model:
        try:
            from google.adk.models.lite_llm import LiteLlm
        except ImportError:
            return None, (
                "OpenAI-compatible mode requires LiteLLM. "
                "Please install: google-adk[extensions] (or litellm)."
            )

        # LiteLLM 常见模型格式：provider/model_name
        # 如果用户只写了模型名（如 qwen3-coder-plus），这里自动补成 openai/qwen3-coder-plus。
        litellm_model = (
            openai_model if "/" in openai_model else f"openai/{openai_model}"
        )

        # 传给 LiteLlm 的连接参数：
        # - api_key：鉴权
        # - api_base：可选，只有使用兼容网关时才需要指定
        kwargs: dict[str, Any] = {"api_key": openai_api_key}
        if openai_base_url:
            kwargs["api_base"] = openai_base_url

        return LiteLlm(model=litellm_model, **kwargs), (
            f"Using OpenAI-compatible backend: {litellm_model}"
        )

    # 兜底：如果没有 OpenAI-compatible 配置，则尝试原生 Gemini。
    # 这里返回字符串模型名即可，ADK 会走 Gemini 默认通道。
    if google_api_key:
        return google_model, f"Using Gemini backend: {google_model}"

    # 两套配置都不可用，返回错误提示给上层打印。
    return None, (
        "No valid model credentials found. "
        "Set OPENAI_API_KEY+OPENAI_MODEL (optionally OPENAI_BASE_URL), "
        "or set GOOGLE_API_KEY."
    )


# --- Define Tool Functions ---
def booking_tool(request: str) -> str:
    """处理预订类请求（机票/酒店）。"""
    return (
        f"Booking action for '{request}' has been simulated. "
        f"[booking_tool handled]"
    )


def info_tool(request: str) -> str:
    """处理通用信息问答请求。"""
    return (
        f"Information request for '{request}' was handled. "
        f"Result: Simulated information retrieval. [info_tool handled]"
    )


def unclear_tool(request: str) -> str:
    """处理无法明确分类的请求，给出澄清提示。"""
    return (
        f"Coordinator could not delegate request: '{request}'. "
        f"Please clarify. [unclear_tool handled]"
    )


# 根据环境变量动态得到“实际模型对象（或模型名）+ 状态说明”。
ADK_MODEL, MODEL_STATUS = _build_adk_model()

if ADK_MODEL is not None:
    # ------------------------- 子代理定义 -------------------------
    # 3 个子代理分别负责不同任务域，每个子代理都挂载一个 FunctionTool。
    # 这里的工具函数目前是“模拟实现”，后续可替换为真实 API/数据库逻辑。
    booking_agent = Agent(
        name="booking_agent",
        model=ADK_MODEL,
        description=(
            "A specialized agent that handles all flight and hotel booking tasks."
        ),
        instruction=(
            "You are a booking expert. If a request is about flights/hotels/reservations, "
            "use your tool and return the booking result."
        ),
        tools=[FunctionTool(booking_tool)],
    )

    info_agent = Agent(
        name="info_agent",
        model=ADK_MODEL,
        description="A specialized agent that provides general information.",
        instruction=(
            "You answer general information requests. Use your tool and return the result."
        ),
        tools=[FunctionTool(info_tool)],
    )

    unclear_agent = Agent(
        name="unclear_agent",
        model=ADK_MODEL,
        description="Fallback agent for ambiguous or unclear requests.",
        instruction=(
            "When a request is ambiguous or cannot be categorized clearly, "
            "use your fallback tool and ask for clarification."
        ),
        tools=[FunctionTool(unclear_tool)],
    )

    # ------------------------- 协调器代理 -------------------------
    # coordinator 只负责“路由与委派”，不直接执行具体任务。
    # 通过 instruction 明确约束路由规则，降低模型自由发挥导致的误分流。
    coordinator_agent = Agent(
        name="coordinator",
        model=ADK_MODEL,
        description=(
            "Main coordinator that routes user requests to the correct specialist."
        ),
        instruction="""
You are a delegation-only coordinator.
Analyze each user request and route it to one specialist agent.

Routing policy:
- Booking intent (flight/hotel/reservation/ticket/hotel room) -> booking_agent
- General factual questions -> info_agent
- Ambiguous/unclear requests -> unclear_agent

Rules:
- Do not answer directly if delegation is possible.
- Delegate to exactly one sub-agent.
""".strip(),
        sub_agents=[booking_agent, info_agent, unclear_agent],
    )
else:
    # 模型不可用时，把代理设为 None，主函数中统一做保护性退出。
    booking_agent = None
    info_agent = None
    unclear_agent = None
    coordinator_agent = None


# --- Runner Execution Logic ---
async def call_agent_async(
    query: str, user_id: str = "user_123", session_id: str = "session_abc"
) -> str:
    """
    处理单条请求：
    1) 创建 ADK 会话
    2) 用 Runner 驱动 coordinator + sub-agents
    3) 从事件流中提取最终文本输出
    """
    if coordinator_agent is None:
        raise RuntimeError(MODEL_STATUS)

    # 会话服务负责存储每次对话上下文。
    # 注意：create_session 在当前 ADK 版本是异步函数，必须 await。
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    # Runner 是 ADK 的执行入口：把“用户消息 + 代理树 + 会话状态”串起来执行。
    runner = Runner(
        app_name=APP_NAME,
        agent=coordinator_agent,
        session_service=session_service,
    )

    # 按 Google GenAI 协议封装用户输入。
    user_message = Content(role="user", parts=[Part(text=query)])
    final_response_text = "No final response generated."

    # run_async 返回事件流（token、工具调用、中间状态、最终响应等）。
    # 这里我们只提取“最终响应事件”里的文本部分。
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            maybe_text = getattr(event.content.parts[0], "text", None)
            if maybe_text:
                final_response_text = maybe_text
                break

    return final_response_text


def main_google_adk() -> None:
    """
    Note:
    This function is intentionally named `main_google_adk` (not `main`),
    as requested.
    """
    # 如果模型配置不可用，打印原因并安全退出。
    if coordinator_agent is None:
        print(f"Skipping execution. {MODEL_STATUS}")
        return

    print("=== Running Google ADK routing demo ===")
    print(MODEL_STATUS)

    # 三个示例请求，分别覆盖：预订、常识问答、开放问题。
    requests = [
        ("Result A", "Book me a flight to London."),
        ("Result B", "What is the capital of Italy?"),
        ("Result C", "Tell me about quantum physics."),
    ]

    # 逐条运行并打印结果；单条失败不会中断后续请求，便于调试。
    for label, query in requests:
        print(f"\n--- Query: {query}")
        try:
            result = asyncio.run(call_agent_async(query=query))
            print(f"{label}: {result}")
        except Exception as exc:
            print(f"{label}: Error while processing request: {exc}")


if __name__ == "__main__":
    main_google_adk()

"""
Parallelization_3（Google ADK 版）：
并行研究 + 串行综合 的多代理流水线示例

核心流程：
1) 三个研究子代理并行执行（可再生能源 / 电动车 / 碳捕集）
2) 一个综合子代理串行收口（融合并校验并行结果）

模型后端策略：
- 优先 OpenAI-compatible（OPENAI_API_KEY + OPENAI_MODEL）
- 回退 Gemini（GOOGLE_API_KEY + GOOGLE_MODEL）
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemorySessionService, Runner
from google.adk.tools import FunctionTool
from google.adk.tools import google_search
from google.genai.types import Content, Part


# ------------------------- 环境加载 -------------------------
# 固定从当前脚本目录读取 .env，避免因“执行目录不同”导致找不到配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ADK Runner 的应用名（会话隔离命名空间）
APP_NAME = "parallelization_3_google_adk_app"


def _build_adk_model() -> tuple[Any | None, str, str]:
    """
    根据环境变量选择模型后端：
    1) OpenAI-compatible 优先
    2) Gemini 回退

    Returns:
        (model_or_none, status_text, backend_type)
    """
    # ---------- OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    if openai_api_key and openai_model:
        # LiteLLM 常见模型格式是 provider/model。
        # 若用户只写模型名（如 qwen3.5-plus），这里自动补成 openai/qwen3.5-plus。
        litellm_model = (
            openai_model if "/" in openai_model else f"openai/{openai_model}"
        )

        kwargs: dict[str, Any] = {"api_key": openai_api_key}
        if openai_base_url:
            # 兼容 one-api / OpenRouter / 其他 OpenAI-compatible 网关
            kwargs["api_base"] = openai_base_url

        return (
            LiteLlm(model=litellm_model, **kwargs),
            f"Using OpenAI-compatible backend: {litellm_model}",
            "openai_compatible",
        )

    # ---------- Gemini（回退） ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    if google_api_key:
        # ADK 对 Gemini 可直接传字符串模型名。
        return google_model, f"Using Gemini backend: {google_model}", "gemini"

    return (
        None,
        "No valid model credentials found. "
        "Set OPENAI_API_KEY+OPENAI_MODEL (optionally OPENAI_BASE_URL), "
        "or set GOOGLE_API_KEY.",
        "none",
    )


ADK_MODEL, MODEL_STATUS, MODEL_BACKEND = _build_adk_model()


def renewable_search_tool(query: str) -> str:
    """
    OpenAI-compatible 回退工具（模拟研究检索）：
    当后端不是 Gemini 时，用该函数工具替代 google_search，避免工具不兼容报错。
    """
    return (
        "Renewable research notes (simulated): "
        f"query='{query}'. Focus areas include solar efficiency, "
        "grid-scale storage, policy incentives, and commercialization pace."
    )


def ev_search_tool(query: str) -> str:
    """
    OpenAI-compatible 回退工具（模拟 EV 检索）：
    与 renewable_search_tool 配对，保证并行代理在非 Gemini 后端仍可执行。
    """
    return (
        "EV research notes (simulated): "
        f"query='{query}'. Focus areas include battery chemistry, "
        "fast charging network growth, cost trends, and adoption constraints."
    )


def carbon_capture_search_tool(query: str) -> str:
    """
    OpenAI-compatible 回退工具（模拟碳捕集检索）：
    与前两个研究工具形成三路并行，保证结构与截图一致。
    """
    return (
        "Carbon capture research notes (simulated): "
        f"query='{query}'. Focus areas include DAC, post-combustion capture, "
        "cost per ton CO2, and pilot/commercial deployment status."
    )


if ADK_MODEL is not None:
    # google_search 是 Gemini 专属工具；在 OpenAI-compatible 模型上会抛错。
    # 因此这里按后端动态选择工具：
    # - Gemini: 真正使用 google_search
    # - OpenAI-compatible: 使用 FunctionTool 包装的本地回退函数
    if MODEL_BACKEND == "gemini":
        researcher_1_tools = [google_search]
        researcher_2_tools = [google_search]
        researcher_3_tools = [google_search]
    else:
        researcher_1_tools = [FunctionTool(renewable_search_tool)]
        researcher_2_tools = [FunctionTool(ev_search_tool)]
        researcher_3_tools = [FunctionTool(carbon_capture_search_tool)]

    # ------------------------- 并行研究子代理 1 -------------------------
    # 负责“可再生能源”方向的研究，使用 google_search 工具抓取信息。
    researcher_agent_1 = LlmAgent(
        name="RenewableEnergyResearcher",
        model=ADK_MODEL,
        description=(
            "Researches renewable energy sources."
        ),
        instruction=(
            "You are an AI Research Assistant specializing in energy.\n"
            "Research the latest advancements in 'renewable energy sources'.\n"
            "Use the available search tool provided.\n"
            "Summarize your key findings concisely (1-2 sentences).\n"
            "Output only the summary."
        ),
        tools=researcher_1_tools,
        # 把该子代理结果写入共享状态，供后续 merger/synthesis 使用。
        output_key="renewable_energy_result",
    )

    # ------------------------- 并行研究子代理 2 -------------------------
    # 负责“电动汽车与相关技术”方向，形成与代理 1 互补的信息面。
    researcher_agent_2 = LlmAgent(
        name="EVResearcher",
        model=ADK_MODEL,
        description=(
            "Researches electric vehicle technology."
        ),
        instruction=(
            "You are an AI Research Assistant specializing in transportation.\n"
            "Research the latest developments in 'electric vehicle technology'.\n"
            "Use the available search tool provided.\n"
            "Summarize your key findings concisely (1-2 sentences).\n"
            "Output only the summary."
        ),
        tools=researcher_2_tools,
        output_key="ev_technology_result",
    )

    # ------------------------- 并行研究子代理 3 -------------------------
    # 负责“碳捕集（Carbon Capture）”方向，补全截图中的第三个 researcher。
    researcher_agent_3 = LlmAgent(
        name="CarbonCaptureResearcher",
        model=ADK_MODEL,
        description=(
            "Researches carbon capture methods."
        ),
        instruction=(
            "You are an AI Research Assistant specializing in climate solutions.\n"
            "Research the current state of carbon capture methods.\n"
            "Use the available search tool provided.\n"
            "Summarize your key findings concisely (1-2 sentences).\n"
            "Output only the summary."
        ),
        tools=researcher_3_tools,
        output_key="carbon_capture_result",
    )

    # ------------------------- 并行节点 -------------------------
    # ParallelAgent 会并发执行三个研究子代理，缩短整体响应时间。
    parallel_research_agent = ParallelAgent(
        name="parallel_research_agent",
        description=(
            "Runs multiple specialized researchers in parallel."
        ),
        sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    )

    # ------------------------- 综合分析子代理 -------------------------
    # 该代理位于并行节点之后，负责“收敛”并行结果，输出最终结构化结论。
    synthesis_agent = LlmAgent(
        name="FinalSynthesisAgent",
        model=ADK_MODEL,
        description=(
            "Senior analyst that merges three research outputs into one final response."
        ),
        instruction=(
            "You are a senior analyst and final merger.\n"
            "Read and merge results from these state keys:\n"
            "- renewable_energy_result\n"
            "- ev_technology_result\n"
            "- carbon_capture_result\n"
            "Produce one integrated answer with this structure:\n"
            "1) Executive Summary\n"
            "2) Key Evidence\n"
            "3) Cross-domain Implications\n"
            "4) Open Questions / Uncertainties\n"
            "Requirements:\n"
            "- Remove duplicates and reconcile overlaps.\n"
            "- If findings conflict, state both and explain uncertainty.\n"
            "- Keep a clear, practical, and concise tone."
        )
    )

    # ------------------------- 根代理（串行编排） -------------------------
    # SequentialAgent 用于表达“先并行研究，再综合收口”的两阶段流程。
    root_agent = SequentialAgent(
        name="research_pipeline_agent",
        description=(
            "A pipeline: parallel research first, then synthesis."
        ),
        sub_agents=[parallel_research_agent, synthesis_agent],
    )
else:
    # 若模型未初始化成功，下面主函数会给出错误并安全退出。
    root_agent = None


async def call_agent_async(
    query: str, user_id: str = "user_123", session_id: str = "session_parallel_001"
) -> str:
    """
    异步调用 ADK 代理树并提取最终文本结果。

    执行步骤：
    1) 创建会话（保存上下文状态）
    2) 构造 Runner（连接根代理 + 会话服务）
    3) 发送用户消息并遍历事件流
    4) 从 final response 事件中抽取文本
    """
    if root_agent is None:
        raise RuntimeError(MODEL_STATUS)

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    user_message = Content(role="user", parts=[Part(text=query)])
    final_response_text = "No final response generated."

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            maybe_text = getattr(event.content.parts[0], "text", None)
            if maybe_text:
                final_response_text = maybe_text

    return final_response_text


def main_google_adk() -> None:
    """
    Google ADK 入口函数（按你的要求命名为 main_google_adk）。
    """
    if root_agent is None:
        print(f"Skipping execution. {MODEL_STATUS}")
        return

    print("=== Running Google ADK Parallelization Demo ===")
    print(MODEL_STATUS)

    # 示例问题：覆盖三个研究域（可再生能源 / EV / 碳捕集），再统一综合。
    test_queries = [
        (
            "Result A",
            "Provide a comprehensive analysis of sustainable energy advancements "
            "and EV technology trends in recent years, including carbon capture progress.",
        )
    ]

    for idx, (label, query) in enumerate(test_queries, start=1):
        print(f"\n--- Query {idx}: {query}")
        try:
            result = asyncio.run(
                call_agent_async(query=query, session_id=f"session_parallel_{idx}")
            )
            print(f"{label}: {result}")
        except Exception as exc:
            print(f"{label}: Error while processing request: {exc}")


if __name__ == "__main__":
    main_google_adk()

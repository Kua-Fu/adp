"""
hitl_13: Human-in-the-Loop（HITL）技术支持智能体示例

本示例基于截图中的结构实现：
1) 一个技术支持 Agent（可调用排障/建单/转人工工具）；
2) 一个 before_model_callback（给模型请求注入客户画像信息）；
3) OpenAI-compatible 优先、Gemini 回退的模型策略；
4) 最小可运行 Runner（交互输入问题并输出结果）。
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import InMemorySessionService, Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# ------------------------- 环境加载 -------------------------
# 固定读取当前模块目录下的 .env，避免在不同 cwd 运行时读取错环境变量。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# ADK Runner 的应用名（用于隔离会话命名空间）。
APP_NAME = "hitl_13_adk_app"


def _build_adk_model() -> tuple[Any | None, str]:
    """
    构建 ADK 模型后端（OpenAI-compatible 优先，Gemini 回退）。

    优先级说明：
    1) OpenAI-compatible:
       - OPENAI_API_KEY + OPENAI_MODEL
       - OPENAI_BASE_URL 可选（兼容网关场景）
    2) Gemini:
       - GOOGLE_API_KEY
       - GOOGLE_MODEL 可选，默认 gemini-2.5-flash
    """
    # ---------- 1) OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    if openai_api_key and openai_model:
        try:
            # LiteLlm 让 ADK 能通过 OpenAI-compatible 协议访问模型。
            from google.adk.models.lite_llm import LiteLlm
        except ImportError:
            return None, (
                "OpenAI-compatible 模式需要 LiteLlm。"
                "请安装 google-adk[extensions] 或 litellm。"
            )

        # 若模型名不含 provider 前缀，则自动补 openai/ 前缀。
        litellm_model = (
            openai_model if "/" in openai_model else f"openai/{openai_model}"
        )

        kwargs: dict[str, Any] = {"api_key": openai_api_key}
        if openai_base_url:
            kwargs["api_base"] = openai_base_url

        return LiteLlm(model=litellm_model, **kwargs), (
            f"模型后端：OpenAI-compatible（优先） -> {litellm_model}"
        )

    # ---------- 2) Gemini（回退） ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    if google_api_key:
        # ADK 原生 Gemini 支持直接传字符串模型名。
        return google_model, f"模型后端：Gemini（回退） -> {google_model}"

    # ---------- 3) 无可用模型 ----------
    return None, (
        "未检测到可用模型配置。请在 hitl_13/.env 配置："
        "OPENAI_API_KEY + OPENAI_MODEL（优先），"
        "或 GOOGLE_API_KEY（回退）。"
    )


# ------------------------- 工具函数（截图同款） -------------------------
# 占位工具：真实项目中可替换为诊断 API、工单系统、人工客服系统接口。
def troubleshoot_issue(issue: str) -> dict:
    """模拟排障工具：返回基础排障报告。"""
    return {
        "status": "success",
        "report": f"Troubleshooting steps for {issue}.",
    }


def create_ticket(issue_type: str, details: str) -> dict:
    """模拟建单工具：返回工单号。"""
    return {
        "status": "success",
        "ticket_id": "TICKET123",
        "issue_type": issue_type,
        "details": details,
    }


def escalate_to_human(issue_type: str, tool_context: ToolContext) -> dict:
    """
    模拟转人工工具。

    这里通过 ToolContext 写入状态位，体现 HITL 的“人机协同”落点：
    - state["needs_human"] = True 表示当前会话需要人工介入。
    """
    tool_context.state["needs_human"] = True
    return {
        "status": "success",
        "message": f"Escalated {issue_type} to a human specialist.",
    }


def personalization_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[types.Content]:
    """
    将客户画像信息注入模型请求（截图逻辑的当前 ADK 版本实现）。

    处理逻辑：
    1) 从会话状态读取 customer_info；
    2) 拼装个性化提示文本；
    3) 将该提示作为 system 内容插入到 llm_request 最前面；
    4) 返回 None，表示继续按修改后的请求正常执行。
    """
    customer_info = callback_context.state.get("customer_info")
    if not customer_info:
        return None

    customer_name = customer_info.get("name", "valued customer")
    customer_tier = customer_info.get("tier", "standard")
    recent_purchases = customer_info.get("recent_purchases", [])

    personalization_note = (
        f"\nIMPORTANT PERSONALIZATION:\n"
        f"Customer Name: {customer_name}\n"
        f"Customer Tier: {customer_tier}\n"
    )
    if recent_purchases:
        personalization_note += (
            f"Recent Purchases: {', '.join(recent_purchases)}\n"
        )

    if llm_request.contents:
        system_content = types.Content(
            role="system",
            parts=[types.Part(text=personalization_note)],
        )
        llm_request.contents.insert(0, system_content)

    # 返回 None：让 ADK 使用“已被修改的 llm_request”继续走后续流程。
    return None


# ------------------------- Agent 构建 -------------------------
ADK_MODEL, MODEL_STATUS = _build_adk_model()

if ADK_MODEL is not None:
    technical_support_agent = Agent(
        name="technical_support_specialist",
        model=ADK_MODEL,
        instruction="""
You are a technical support specialist for our electronics company.
FIRST, check if the user has a support history in state["customer_info"]["support_history"].
If they do, reference this history in your responses.

For technical issues:
1. Use the troubleshoot_issue tool to analyze the problem.
2. Guide the user through basic troubleshooting steps.
3. If the issue persists, use create_ticket to log the issue.

For complex issues beyond basic troubleshooting:
1. Use escalate_to_human to transfer to a human specialist.

Maintain a professional but empathetic tone. Acknowledge the frustration
technical issues can cause, while providing clear steps toward resolution.
""".strip(),
        tools=[troubleshoot_issue, create_ticket, escalate_to_human],
        # 在模型调用前注入客户画像信息，实现“个性化 + HITL”入口增强。
        before_model_callback=personalization_callback,
    )
else:
    technical_support_agent = None


async def run_once_async(
    user_issue: str,
    user_id: str = "hitl_user",
    session_id: str = "hitl_session",
) -> str:
    """
    最小可运行入口：发送一条用户问题给 technical_support_agent。
    """
    if technical_support_agent is None:
        raise RuntimeError(MODEL_STATUS)

    # 构造会话初始状态：包含客户信息，供 callback 与智能体指令读取。
    initial_state = {
        "customer_info": {
            "name": "Alex Chen",
            "tier": "premium",
            "recent_purchases": ["Smart TV X1", "Soundbar Pro"],
            "support_history": [
                "2026-03-02: TV 偶发重启，已指导升级固件",
                "2026-04-01: HDMI 无信号，用户更换线缆后恢复",
            ],
        },
        "needs_human": False,
    }

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=technical_support_agent,
        session_service=session_service,
    )

    user_message = types.Content(role="user", parts=[types.Part(text=user_issue)])
    final_response_text = "未收到最终响应。"

    # 读取事件流并提取最终响应；同时打印调用链，便于观察工具使用。
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        event_author = getattr(event, "author", None) or getattr(event, "agent_name", None)
        if event_author:
            print(f"[CALLBACK] 收到事件，author={event_author}")

        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    print(f"[CALLBACK] function_call -> name={getattr(fc, 'name', '')}, args={getattr(fc, 'args', {})}")
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    print(
                        "[CALLBACK] function_response -> "
                        f"name={getattr(fr, 'name', '')}, response={getattr(fr, 'response', {})}"
                    )

        if event.is_final_response() and event.content and event.content.parts:
            maybe_text = getattr(event.content.parts[0], "text", None)
            if maybe_text:
                final_response_text = maybe_text

    return final_response_text


def main() -> None:
    """
    命令行入口：
    - 打印当前模型后端状态；
    - 读取用户输入的问题；
    - 调用技术支持智能体并输出最终答复。
    """
    print(MODEL_STATUS)
    if technical_support_agent is None:
        print("智能体未初始化：请先完善 hitl_13/.env 后重试。")
        return

    issue = input("请输入你遇到的技术问题：").strip()
    if not issue:
        print("问题为空，程序已退出。")
        return

    try:
        result = asyncio.run(run_once_async(user_issue=issue))
        print("\n智能体输出：")
        print(result)
    except Exception as exc:
        print(f"执行失败：{exc}")


if __name__ == "__main__":
    main()

"""
recovery_12: 恢复（Recovery）模式的顺序代理示例

本文件按你的截图结构实现三个顺序子代理：
1) primary_handler：优先调用精确定位工具；
2) fallback_handler：当精确定位失败时执行兜底；
3) response_agent：统一整理并输出最终结果。

同时加入模型后端策略：
- OpenAI-compatible 优先；
- Gemini 回退。
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import InMemorySessionService, Runner
from google.genai.types import Content, Part

# ------------------------- 环境加载 -------------------------
# 固定从当前模块目录读取 .env，避免在不同 cwd 运行时读取到错误配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# ADK Runner 的应用名（用于区分会话命名空间）。
APP_NAME = "recovery_12_adk_app"


def _build_adk_model() -> tuple[Any | None, str]:
    """
    构建 ADK 模型对象（或模型名）。

    优先级：
    1) OpenAI-compatible（优先）
       - 需要 OPENAI_API_KEY + OPENAI_MODEL
       - OPENAI_BASE_URL 可选（用于兼容网关）
    2) Gemini（回退）
       - 需要 GOOGLE_API_KEY
       - GOOGLE_MODEL 可选，默认 gemini-2.5-flash
    """
    # ---------- 1) OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL")
    openai_base_url = os.getenv("OPENAI_BASE_URL")

    # 规则：
    # - 只要检测到 OpenAI-compatible 配置，就固定走 OpenAI-compatible；
    # - 即使同时存在 Gemini 配置，也不回退 Gemini（按你的要求）。
    if openai_api_key and openai_model:
        try:
            # LiteLlm 允许 ADK 通过 OpenAI-compatible 接口访问多种模型。
            from google.adk.models.lite_llm import LiteLlm
        except ImportError:
            return None, (
                "OpenAI-compatible 模式需要 LiteLlm。"
                "请安装 google-adk[extensions] 或 litellm。"
                "（已检测到 OpenAI-compatible 配置，因此不会回退 Gemini）"
            )

        # 如果用户只填了裸模型名（例如 qwen3-coder-plus），
        # 自动补成 openai/xxx，满足 LiteLLM 常见模型格式。
        litellm_model = (
            openai_model if "/" in openai_model else f"openai/{openai_model}"
        )

        kwargs: dict[str, Any] = {"api_key": openai_api_key}
        # 当使用 OpenAI 官方地址时可不填；接入兼容网关时建议显式配置。
        if openai_base_url:
            kwargs["api_base"] = openai_base_url

        return LiteLlm(model=litellm_model, **kwargs), (
            f"模型后端：OpenAI-compatible（固定） -> {litellm_model}"
        )

    # ---------- 2) Gemini（回退） ----------
    # ADK 原生 Gemini 通道可直接使用模型字符串；凭据由 GOOGLE_API_KEY 提供。
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

    if google_api_key:
        return google_model, f"模型后端：Gemini（回退） -> {google_model}"

    # ---------- 3) 两条路径都不可用 ----------
    return None, (
        "未检测到可用模型配置。请在 recovery_12/.env 中配置："
        "OPENAI_API_KEY + OPENAI_MODEL（优先），"
        "或 GOOGLE_API_KEY（回退）。"
    )


def get_precise_location_info(address: str) -> str:
    """
    精确定位工具（主路径）。

    说明：
    - 这里用可控的示例逻辑模拟“精确定位服务”；
    - 返回文本中包含 PRECISE_OK / PRECISE_FAIL 标记，
      方便后续代理在推理时判断是否需要回退。
    """
    normalized = address.strip()
    print(f"[TOOL] 调用 get_precise_location_info(address={normalized!r})")
    if not normalized:
        result = "[PRECISE_FAIL] 地址为空，无法执行精确定位。"
        print(f"[TOOL] get_precise_location_info 返回: {result}")
        return result

    # 简单启发式：包含门牌/楼栋等细粒度特征时，视为可命中精确地址。
    has_detail_hint = any(token in normalized for token in ("号", "弄", "楼", "室", "Road", "Street", "Ave"))
    if has_detail_hint:
        result = (
            "[PRECISE_OK] 已找到精确位置："
            f"{normalized}（示例坐标：31.2304, 121.4737）。"
        )
        print(f"[TOOL] get_precise_location_info 返回: {result}")
        return result

    result = (
        "[PRECISE_FAIL] 未检索到精确门牌级结果。"
        "建议提取城市信息后执行区域级检索。"
    )
    print(f"[TOOL] get_precise_location_info 返回: {result}")
    return result


def get_general_area_info(city: str) -> str:
    """
    区域级定位工具（回退路径）。

    说明：
    - 当精确地址失败时，至少返回城市级别的位置信息；
    - 这里同样是示例实现，后续可替换为真实地图/地理编码 API。
    """
    normalized_city = city.strip()
    print(f"[TOOL] 调用 get_general_area_info(city={normalized_city!r})")
    if not normalized_city:
        result = "未提供城市信息，无法回退到区域级定位。"
        print(f"[TOOL] get_general_area_info 返回: {result}")
        return result
    result = (
        f"已回退到区域级定位：{normalized_city}。"
        "当前返回该城市的中心区域与通用地理信息（示例结果）。"
    )
    print(f"[TOOL] get_general_area_info 返回: {result}")
    return result


def _build_location_agent(model: Any) -> SequentialAgent:
    """
    根据传入模型构建“主路径 + 回退 + 响应”顺序代理。
    """
    # Agent 1：优先走主工具，目标是尽可能拿到精确位置。
    primary_handler = Agent(
        name="primary_handler",
        model=model,
        instruction="""
你的职责是优先获取精确位置信息。
请调用 get_precise_location_info，输入用户提供的完整地址。

要求：
1. 如果工具返回 [PRECISE_OK]，将结果写入 state["location_result"]，
   并将 state["primary_location_failed"] 设为 False。
2. 如果工具返回 [PRECISE_FAIL]，将 state["primary_location_failed"] 设为 True，
   并尽量从用户原始输入中提取城市线索写入 state["city_hint"]。
""".strip(),
        tools=[get_precise_location_info],
    )

    # Agent 2：根据状态判断是否触发回退。
    fallback_handler = Agent(
        name="fallback_handler",
        model=model,
        instruction="""
请检查 state["primary_location_failed"] 决定是否执行回退：
- 若为 True：从 state["city_hint"] 或用户原始查询中提取城市名称，
  调用 get_general_area_info，并把结果写入 state["location_result"]。
- 若为 False：不执行任何额外动作，保持已有结果。
""".strip(),
        tools=[get_general_area_info],
    )

    # Agent 3：统一输出最终答复。
    response_agent = Agent(
        name="response_agent",
        model=model,
        instruction="""
请读取 state["location_result"] 并向用户清晰输出最终位置结果。
如果 state["location_result"] 不存在或为空，请礼貌说明暂时无法检索到位置。
""".strip(),
        tools=[],  # 该代理只负责读取状态并组织回复，不调用工具。
    )

    # 顺序代理：保证执行顺序固定为 主路径 -> 回退 -> 响应。
    return SequentialAgent(
        name="robust_location_agent",
        sub_agents=[primary_handler, fallback_handler, response_agent],
    )


# 统一构建模型，供所有子代理复用。
ADK_MODEL, MODEL_STATUS = _build_adk_model()
robust_location_agent = _build_location_agent(ADK_MODEL) if ADK_MODEL is not None else None


async def run_once_async(
    query: str,
    user_id: str = "recovery_user",
    session_id: str = "recovery_session",
    root_agent: SequentialAgent | None = None,
) -> str:
    """
    最小可运行 Runner 示例（单轮调用）。

    执行步骤：
    1) 创建内存会话服务；
    2) 构造 Runner 并挂载根代理；
    3) 发送一条用户消息；
    4) 从事件流中提取最终文本。
    """
    active_agent = root_agent or robust_location_agent
    if active_agent is None:
        raise RuntimeError(MODEL_STATUS)

    # 1) 初始化会话服务并创建会话（异步接口必须 await）。
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    # 2) 构建 Runner，把根代理与会话服务连接起来。
    runner = Runner(
        app_name=APP_NAME,
        agent=active_agent,
        session_service=session_service,
    )

    # 3) 按 Google GenAI 的消息协议封装输入。
    user_message = Content(role="user", parts=[Part(text=query)])
    final_response_text = "未收到最终响应。"

    def _safe_json(value: Any) -> str:
        """把工具参数/返回值稳妥转成可打印文本，避免日志阶段抛异常。"""
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    # 4) 遍历事件流，抓取最终响应文本。
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        event_author = getattr(event, "author", None) or getattr(event, "agent_name", None)
        if event_author:
            print(f"[CALLBACK] 收到事件，author={event_author}")

        # 打印函数调用/函数返回轨迹（即工具调用链路）。
        if getattr(event, "content", None) and getattr(event.content, "parts", None):
            for part in event.content.parts:
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    call_name = getattr(function_call, "name", "")
                    call_args = getattr(function_call, "args", None)
                    print(
                        "[CALLBACK] function_call -> "
                        f"name={call_name}, args={_safe_json(call_args)}"
                    )

                function_response = getattr(part, "function_response", None)
                if function_response is not None:
                    resp_name = getattr(function_response, "name", "")
                    resp_value = getattr(function_response, "response", None)
                    print(
                        "[CALLBACK] function_response -> "
                        f"name={resp_name}, response={_safe_json(resp_value)}"
                    )

        # 注意：不能在第一个 final_response 就 break。
        # 原因：顺序代理中前置子代理也可能先产出 final_response，
        # 若提前退出会截断 fallback_handler / response_agent 的后续执行。
        # 这里改为“全量消费事件流，并以最后一次 final_response 为准”。
        if event.is_final_response() and event.content and event.content.parts:
            maybe_text = getattr(event.content.parts[0], "text", None)
            if maybe_text:
                final_response_text = maybe_text

    return final_response_text


def main() -> None:
    """
    命令行最小入口：
    - 若模型未就绪：打印原因并退出；
    - 若模型可用：运行一次最小 Runner 示例并打印结果。
    """
    print(MODEL_STATUS)
    if robust_location_agent is None:
        print("代理未初始化：请先配置 .env 后再运行。")
        return

    # 运行后由用户输入地址，再由智能体执行定位流程。
    address = input("请输入要查询的地址：").strip()
    if not address:
        print("地址为空，已退出。")
        return

    query = f"请帮我查询这个地址的位置：{address}"
    print(f"已收到地址：{address}")

    try:
        result = asyncio.run(run_once_async(query=query))
        print("智能体输出：")
        print(result)
    except Exception as exc:
        err_text = str(exc)
        if "Connection error" in err_text or "OpenAIException" in err_text:
            print("说明：模型连接失败，代理流程未启动，因此不会出现 primary/fallback/response 的调用日志。")
        print(f"Runner 执行失败：{exc}")


if __name__ == "__main__":
    main()

"""
mcp_10: MCP 工具调用智能体示例

目标：
1) 智能体可连接并使用已创建的 MCP Server（fastmcp_server.py 暴露的 greet 工具）。
2) 模型选择策略与 mm_8 保持一致：OpenAI-compatible 优先，Gemini 回退。
3) 对关键实现步骤添加详细中文注释，便于学习与后续维护。

运行前准备：
- 先启动 MCP 服务端（另开一个终端）：
    python fastmcp_server.py
- 再运行本文件：
    python main.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from fastmcp import Client
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


# ------------------------- 环境加载 -------------------------
# 固定读取 mcp_10 目录下的 .env，避免在任意 cwd 运行时加载错配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


# ------------------------- 配置模型状态 -------------------------
@dataclass
class ModelBundle:
    """
    保存两个候选模型对象。

    字段说明：
    - openai_llm: OpenAI-compatible 模型对象（优先路径）
    - gemini_llm: Gemini 模型对象（回退路径）
    """

    openai_llm: Optional[object]
    gemini_llm: Optional[object]


def _init_llms() -> tuple[ModelBundle, str]:
    """
    初始化两类模型（OpenAI-compatible 优先、Gemini 回退）。

    与 mm_8 的差异：
    - mm_8 是“选一个可用模型直接返回”；
    - 本示例为了实现“失败后自动回退”，会把两种模型都尽量准备好，
      实际调用时先尝试 OpenAI-compatible，异常再回退 Gemini。

    返回：
    - (ModelBundle, status)
      - ModelBundle: 保存可用模型对象（可能只有一个可用）
      - status: 初始化说明文本（用于终端展示）
    """
    openai_llm: Optional[object] = None
    gemini_llm: Optional[object] = None

    # ---------- 1) OpenAI-compatible（优先） ----------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI

            kwargs = {
                "model": openai_model,
                "temperature": openai_temperature,
                "api_key": openai_api_key,
            }
            # 对 OpenAI-compatible 网关场景，允许自定义 base_url。
            if openai_base_url:
                kwargs["base_url"] = openai_base_url

            openai_llm = ChatOpenAI(**kwargs)
        except ImportError:
            # 不立即抛错，留给 Gemini 路径兜底。
            openai_llm = None

    # ---------- 2) Gemini（回退） ----------
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    google_temperature = float(os.getenv("GOOGLE_TEMPERATURE", "0"))

    if google_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            gemini_llm = ChatGoogleGenerativeAI(
                model=google_model,
                temperature=google_temperature,
                google_api_key=google_api_key,
            )
        except ImportError:
            gemini_llm = None

    if openai_llm and gemini_llm:
        return ModelBundle(openai_llm=openai_llm, gemini_llm=gemini_llm), (
            f"OpenAI-compatible 已就绪（优先）：{openai_model}；"
            f"Gemini 已就绪（回退）：{google_model}"
        )

    if openai_llm:
        return ModelBundle(openai_llm=openai_llm, gemini_llm=None), (
            f"仅 OpenAI-compatible 就绪：{openai_model}"
        )

    if gemini_llm:
        return ModelBundle(openai_llm=None, gemini_llm=gemini_llm), (
            f"OpenAI-compatible 不可用，已回退到 Gemini：{google_model}"
        )

    return (
        ModelBundle(openai_llm=None, gemini_llm=None),
        "未找到可用模型凭据。请在 mcp_10/.env 中配置 OPENAI_API_KEY（优先）或 GOOGLE_API_KEY（回退）。",
    )


class MCPAgent:
    """
    一个最小可用的“工具型智能体”。

    设计目标：
    - 让 LLM 自主决定何时调用工具 `mcp_greet`；
    - 工具执行由本地代码完成，最终结果再回填给模型组织自然语言答案；
    - 调用模型时优先 OpenAI-compatible，失败后自动回退 Gemini。
    """

    def __init__(self, model_bundle: ModelBundle, mcp_server_url: str) -> None:
        self.model_bundle = model_bundle
        self.mcp_server_url = mcp_server_url

        # 通过 @tool 定义可供模型调用的“函数工具”。
        # 该工具内部会转发到真实 MCP Server 的 greet 工具。
        @tool
        def mcp_greet(name: str) -> str:
            """
            通过 MCP Server 调用 greet 工具，返回问候结果。

            Args:
                name: 需要问候的姓名。

            Returns:
                MCP 工具执行后的文本结果。
            """
            return asyncio.run(self._call_mcp_tool("greet", {"name": name}))

        self.mcp_greet_tool = mcp_greet

    async def _call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
        """
        异步调用 MCP 工具。

        说明：
        - 这里使用 fastmcp.Client 连接已运行的 MCP Server；
        - 若服务器未启动、地址错误、工具不存在，都会抛出异常并在上层处理。
        """
        async with Client(self.mcp_server_url) as client:
            result = await client.call_tool(tool_name, arguments)

            # result.data 是最推荐读取的结构化输出。
            if result.data is not None:
                return str(result.data)

            # 兼容某些服务端只回 content blocks 的情况。
            text_chunks: list[str] = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text:
                    text_chunks.append(text)

            if text_chunks:
                return "\n".join(text_chunks)

            return "(工具已执行，但返回为空)"

    def _invoke_with_backend_fallback(self, messages, with_tools: bool):
        """
        调用模型并处理“OpenAI-compatible 优先、Gemini 回退”。

        策略：
        1) 若存在 OpenAI-compatible 模型，则先尝试；
        2) 若调用异常，且 Gemini 可用，则自动切换 Gemini 重试；
        3) 若都不可用，抛出异常给上层。
        """

        def _invoke(llm_obj, backend_name: str):
            # 是否给模型绑定工具由调用阶段决定：
            # - 第一轮：绑定工具，让模型有机会发起 tool call
            # - 第二轮：不绑定工具，让模型整合工具结果并输出最终答案
            llm_to_call = llm_obj.bind_tools([self.mcp_greet_tool]) if with_tools else llm_obj
            result = llm_to_call.invoke(messages)
            return result, backend_name

        openai_llm = self.model_bundle.openai_llm
        gemini_llm = self.model_bundle.gemini_llm

        last_error: Optional[Exception] = None

        if openai_llm is not None:
            try:
                return _invoke(openai_llm, "OpenAI-compatible")
            except Exception as exc:
                last_error = exc
                print(f"[WARN] OpenAI-compatible 调用失败，准备回退 Gemini。错误：{exc}")

        if gemini_llm is not None:
            try:
                return _invoke(gemini_llm, "Gemini")
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"所有可用模型均调用失败：{last_error}")

    def run_turn(self, user_text: str) -> str:
        """
        执行单轮智能体流程。

        流程拆解：
        1) 首轮推理（允许工具调用）；
        2) 若模型发起 tool call，则执行 MCP 工具并写回 ToolMessage；
        3) 二轮推理（整合工具结果，给出最终用户可读回复）。
        """
        system_prompt = (
            "你是一个可以使用 MCP 工具的中文助理。\n"
            "当用户要求问候某人、生成打招呼语时，请优先调用工具 `mcp_greet`。\n"
            "如果用户只是普通闲聊或问题不需要工具，可直接回答。"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]

        # 第一轮：允许模型决定是否调用工具。
        first_response, backend_used = self._invoke_with_backend_fallback(messages, with_tools=True)

        tool_calls = getattr(first_response, "tool_calls", None) or []
        if not tool_calls:
            # 无工具调用，直接返回模型首轮回答。
            print(f"[INFO] 后端：{backend_used}（未触发工具调用）")
            return str(first_response.content)

        # 执行模型请求的每一个工具调用。
        tool_messages = []
        for call in tool_calls:
            call_name = call.get("name")
            call_args = call.get("args", {})
            call_id = call.get("id")

            if call_name != self.mcp_greet_tool.name:
                # 兜底分支：若模型请求了未知工具，明确返回错误文本。
                tool_output = f"未知工具：{call_name}"
            else:
                try:
                    tool_output = self.mcp_greet_tool.invoke(call_args)
                except Exception as exc:
                    tool_output = f"MCP 工具调用失败：{exc}"

            tool_messages.append(ToolMessage(content=tool_output, tool_call_id=call_id))

        # 第二轮：把“模型工具请求 + 工具返回”都喂回去，让模型组织最终答复。
        second_messages = [*messages, first_response, *tool_messages]
        second_response, backend_used_second = self._invoke_with_backend_fallback(
            second_messages,
            with_tools=False,
        )
        print(f"[INFO] 后端：{backend_used_second}（已执行工具调用）")

        return str(second_response.content)


async def _probe_mcp_tools(mcp_server_url: str) -> list[str]:
    """
    连接 MCP Server 并列出工具名称。

    用途：
    - 在正式对话前做一次连通性与工具清单检查，
      帮助快速发现“服务没启动 / URL 错误 / 工具未注册”等问题。
    """
    async with Client(mcp_server_url) as client:
        tools = await client.list_tools()
        return [tool.name for tool in tools]


def main() -> None:
    """
    脚本入口。

    默认行为：
    1) 加载并初始化模型（OpenAI-compatible 优先）；
    2) 检查 MCP 服务可连通且包含 greet 工具；
    3) 启动交互式会话，持续接收用户输入并返回回答。

    你也可以通过环境变量覆写：
    - MCP_SERVER_URL: MCP Server 地址（默认 http://127.0.0.1:8000/mcp）
    """
    print("## Running mcp_10 interactive agent ##")

    bundle, status = _init_llms()
    if bundle.openai_llm is None and bundle.gemini_llm is None:
        print(f"模型初始化失败：{status}")
        return

    print(f"模型初始化状态：{status}")

    mcp_server_url = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
    print(f"MCP Server URL: {mcp_server_url}")

    # 先探测服务与工具列表，帮助用户快速判断服务端是否正确启动。
    try:
        tool_names = asyncio.run(_probe_mcp_tools(mcp_server_url))
        print(f"MCP 工具列表：{tool_names}")
    except Exception as exc:
        print(
            "无法连接 MCP Server，请先在另一个终端启动 fastmcp_server.py。\n"
            f"连接错误：{exc}"
        )
        return

    if "greet" not in tool_names:
        print("MCP Server 未发现 greet 工具，请检查 fastmcp_server.py 是否已正确注册 @mcp_server.tool")
        return

    agent = MCPAgent(model_bundle=bundle, mcp_server_url=mcp_server_url)
    print("\n已进入交互模式。")
    print("输入 `exit` / `quit` / `q` 可退出。")
    print("示例：请通过工具向 Alice 打个招呼，并用中文说明你做了什么。\n")

    while True:
        try:
            # 使用 input 进入阻塞式交互，适合本地终端逐轮对话。
            user_text = input("用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C 时优雅退出，避免抛出难读的堆栈信息。
            print("\n会话已结束。")
            break

        # 对空输入直接提示并继续，避免无意义模型调用。
        if not user_text:
            print("助手: 你还没有输入内容，请重新输入。")
            continue

        # 统一小写后判断退出命令，提升输入容错。
        if user_text.lower() in {"exit", "quit", "q"}:
            print("会话已结束。")
            break

        try:
            answer = agent.run_turn(user_text)
            print(f"助手: {answer}\n")
        except Exception as exc:
            # 单轮失败不退出主循环，方便用户继续下一轮排查/对话。
            print(f"助手: 本轮执行失败：{exc}\n")


if __name__ == "__main__":
    main()

"""
resource_16: Prompt 分类 + Google 搜索 + 回答生成示例

运行环境（按你的要求）：
    /Users/yz/work/env/adp/.venv/bin/python main.py

设计目标：
1) 先把用户问题分类为 simple / reasoning / internet_search；
2) 若需要联网检索，则调用 Google Custom Search；
3) 按分类选择更合适的模型回答；
4) 模型后端遵循「OpenAI-compatible 优先，Gemini 回退」。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI


# ------------------------- 环境加载 -------------------------
# 固定从当前模块目录加载 .env，避免从错误工作目录读取到其他模块配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


@dataclass
class RuntimeConfig:
    """
    保存运行期配置和客户端，避免在函数之间传很多零散参数。
    """

    openai_client: OpenAI | None
    openai_model: str
    openai_classifier_model: str
    openai_simple_model: str
    openai_reasoning_model: str
    openai_search_model: str
    google_api_key: str | None
    google_model: str
    cse_api_key: str | None
    cse_id: str | None


def _clean_text(value: str | None) -> str:
    """把 None 转为空字符串，并去掉两端空白。"""
    return (value or "").strip()


def init_runtime() -> RuntimeConfig:
    """
    初始化运行时配置：
    - OpenAI-compatible 优先（只要配置了 OPENAI_API_KEY 就启用）；
    - Gemini 仅作为回退（当 OpenAI-compatible 调用失败时使用）；
    - Google 搜索配置独立读取（用于 internet_search 分类）。
    """
    openai_api_key = _clean_text(os.getenv("OPENAI_API_KEY"))
    openai_base_url = _clean_text(os.getenv("OPENAI_BASE_URL"))
    openai_model = _clean_text(os.getenv("OPENAI_MODEL")) or "gpt-4o"
    # 默认全部使用 OPENAI_MODEL（避免写死不存在的模型）。
    # 如果你希望不同阶段用不同模型，可在 .env 里额外配置以下变量。
    openai_classifier_model = _clean_text(os.getenv("OPENAI_CLASSIFIER_MODEL")) or openai_model
    openai_simple_model = _clean_text(os.getenv("OPENAI_SIMPLE_MODEL")) or openai_model
    openai_reasoning_model = _clean_text(os.getenv("OPENAI_REASONING_MODEL")) or openai_model
    openai_search_model = _clean_text(os.getenv("OPENAI_SEARCH_MODEL")) or openai_model

    google_api_key = _clean_text(os.getenv("GOOGLE_API_KEY")) or None
    google_model = _clean_text(os.getenv("GOOGLE_MODEL")) or "gemini-2.0-flash"

    # Google CSE 配置：仅从 .env 的 CSE_API_KEY / CSE_ID 读取。
    cse_api_key = _clean_text(os.getenv("CSE_API_KEY")) or None
    cse_id = _clean_text(os.getenv("CSE_ID")) or None

    openai_client: OpenAI | None = None
    if openai_api_key:
        # OpenAI SDK 也可用于 OpenAI-compatible 网关；base_url 可选。
        client_kwargs: dict[str, Any] = {"api_key": openai_api_key}
        if openai_base_url:
            client_kwargs["base_url"] = openai_base_url
        openai_client = OpenAI(**client_kwargs)

    if not openai_client and not google_api_key:
        raise ValueError(
            "未检测到可用模型凭据。请在 resource_16/.env 至少配置 OPENAI_API_KEY（优先）"
            "或 GOOGLE_API_KEY（回退）。"
        )

    return RuntimeConfig(
        openai_client=openai_client,
        openai_model=openai_model,
        openai_classifier_model=openai_classifier_model,
        openai_simple_model=openai_simple_model,
        openai_reasoning_model=openai_reasoning_model,
        openai_search_model=openai_search_model,
        google_api_key=google_api_key,
        google_model=google_model,
        cse_api_key=cse_api_key,
        cse_id=cse_id,
    )


def _chat_with_gemini(prompt: str, model: str, google_api_key: str) -> str:
    """
    使用 Gemini 生成文本。

    这里采用延迟导入：
    - 只有真的需要 Gemini 回退时才要求安装 google-generativeai；
    - 若未安装，会抛出明确错误，便于快速定位问题。
    """
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI-compatible 调用失败，且 Gemini 回退需要安装 google-generativeai。"
        ) from exc

    genai.configure(api_key=google_api_key)
    gemini_model = genai.GenerativeModel(model_name=model)
    response = gemini_model.generate_content(prompt)
    return (response.text or "").strip()


def chat_with_fallback(
    runtime: RuntimeConfig,
    user_prompt: str,
    *,
    openai_model: str,
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """
    统一模型调用入口：
    1) 优先尝试 OpenAI-compatible；
    2) OpenAI-compatible 报错时，若存在 GOOGLE_API_KEY 则自动回退 Gemini；
    3) 返回 (文本内容, 实际使用后端标识)。
    """
    if runtime.openai_client is not None:
        try:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            response = runtime.openai_client.chat.completions.create(
                model=openai_model,
                messages=messages,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            return content.strip(), f"OpenAI-compatible/{openai_model}"
        except Exception as exc:
            if runtime.google_api_key:
                print(f"[WARN] OpenAI-compatible 调用失败，准备回退 Gemini。错误：{exc}")
            else:
                raise

    if runtime.google_api_key:
        combined_prompt = user_prompt
        if system_prompt:
            # Gemini 无需强制区分 system/user，这里做成单段提示词以兼容更多版本 SDK。
            combined_prompt = (
                f"你必须严格遵守以下系统指令：\n{system_prompt}\n\n"
                f"用户问题：\n{user_prompt}"
            )
        text = _chat_with_gemini(
            prompt=combined_prompt,
            model=runtime.google_model,
            google_api_key=runtime.google_api_key,
        )
        return text, f"Gemini/{runtime.google_model}"

    raise RuntimeError("OpenAI-compatible 与 Gemini 都不可用，无法继续生成回答。")


def _parse_classification(raw_text: str) -> dict[str, str]:
    """
    解析分类模型返回的 JSON。
    - 兼容 ```json ... ``` 代码块；
    - 解析失败时给出保守默认值，避免流程中断。
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"classification": "simple", "reason": "模型输出非 JSON，已降级到 simple。"}

    cls = _clean_text(data.get("classification")) or "simple"
    reason = _clean_text(data.get("reason")) or "未提供分类原因。"
    if cls not in {"simple", "reasoning", "internet_search"}:
        cls = "simple"
        reason = f"分类标签非法，已降级为 simple。原始原因：{reason}"
    return {"classification": cls, "reason": reason}


def classify_prompt(prompt: str, runtime: RuntimeConfig) -> tuple[dict[str, str], str]:
    """
    第一步：把用户问题分类。

    分类规则与截图一致：
    - simple：常识问答或无需多步推理的问题；
    - reasoning：需要逻辑推导、多步分析的问题；
    - internet_search：涉及时效信息或必须检索外部资料的问题。
    """
    system_message = (
        "You are a classifier that analyzes user prompts and returns one of three "
        "categories ONLY:\\n\\n"
        "- simple\\n"
        "- reasoning\\n"
        "- internet_search\\n\\n"
        "Rules:\\n"
        "- Use 'simple' for direct factual questions that need no current events.\\n"
        "- Use 'reasoning' for logic/math, or multi-step questions.\\n"
        "- Use 'internet_search' if the prompt refers to current events, recent data, "
        "or things not in your training data.\\n\\n"
        "Respond with JSON like:\\n"
        "{\"classification\": \"simple\", \"reason\": \"...\"}"
    )

    reply, backend = chat_with_fallback(
        runtime,
        prompt,
        openai_model=runtime.openai_classifier_model,
        system_prompt=system_message,
    )
    return _parse_classification(reply), backend


def google_search(query: str, runtime: RuntimeConfig, num_results: int = 1) -> list[dict[str, Any]]:
    """
    第二步（可选）：Google Custom Search 检索。

    返回统一结构的列表，每项包含 title/snippet/link。
    若配置缺失或请求失败，返回带 error 字段的列表，调用方可直接拼接到上下文中。
    """
    if not runtime.cse_api_key or not runtime.cse_id:
        return [{"error": "未配置 CSE_API_KEY 或 CSE_ID，无法联网检索。"}]

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": runtime.cse_api_key, "cx": runtime.cse_id, "q": query, "num": num_results}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        result_json = response.json()
    except requests.RequestException as exc:
        return [{"error": f"Google 搜索失败：{exc}"}]

    items = result_json.get("items") or []
    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        }
        for item in items
    ]


def generate_response(
    prompt: str,
    classification: str,
    runtime: RuntimeConfig,
    search_results: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """
    第三步：按分类生成最终答案。

    模型路由策略（优先参考 .env，避免模型不存在）：
    - simple -> OPENAI_SIMPLE_MODEL（默认回落到 OPENAI_MODEL）
    - reasoning -> OPENAI_REASONING_MODEL（默认回落到 OPENAI_MODEL）
    - internet_search -> OPENAI_SEARCH_MODEL（默认回落到 OPENAI_MODEL）
    """
    if classification == "simple":
        model = runtime.openai_simple_model
        full_prompt = prompt
    elif classification == "reasoning":
        model = runtime.openai_reasoning_model
        full_prompt = prompt
    elif classification == "internet_search":
        model = runtime.openai_search_model
        # 把搜索结果转成可读上下文字符串，便于模型引用与总结。
        rows: list[str] = []
        for idx, item in enumerate(search_results or [], start=1):
            if "error" in item:
                rows.append(f"[{idx}] Error: {item.get('error', '')}")
                continue
            rows.append(
                f"[{idx}] Title: {item.get('title', '')}\n"
                f"Snippet: {item.get('snippet', '')}\n"
                f"Link: {item.get('link', '')}"
            )
        search_context = "\n\n".join(rows) if rows else "无可用搜索结果。"
        full_prompt = (
            "Use the following web search results to answer the query.\n\n"
            f"Search Context:\n{search_context}\n\n"
            f"Query: {prompt}"
        )
    else:
        # 极端兜底：未知分类时按 simple 处理，保证程序健壮性。
        model = runtime.openai_simple_model
        full_prompt = prompt

    return chat_with_fallback(runtime, full_prompt, openai_model=model)


def main() -> None:
    """
    交互式入口：
    - 连续读取用户输入；
    - 输出分类、后端信息与最终回答；
    - 输入 exit/quit/q 可退出。
    """
    runtime = init_runtime()
    print("模型策略：OpenAI-compatible 优先，Gemini 回退")
    print("输入问题开始对话，输入 exit / quit / q 结束。")

    while True:
        prompt = input("\n请输入问题：").strip()
        if prompt.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            return
        if not prompt:
            print("输入为空，请重新输入。")
            continue

        try:
            classification_result, cls_backend = classify_prompt(prompt, runtime)
            classification = classification_result["classification"]
            print(f"[分类] {classification} | 后端: {cls_backend}")
            print(f"[分类原因] {classification_result['reason']}")

            search_results: list[dict[str, Any]] | None = None
            if classification == "internet_search":
                search_results = google_search(prompt, runtime)
                print(f"[搜索结果] {search_results}")

            answer, ans_backend = generate_response(
                prompt=prompt,
                classification=classification,
                runtime=runtime,
                search_results=search_results,
            )
            print(f"[回答后端] {ans_backend}")
            print(f"[回答]\n{answer}")
        except Exception as exc:
            print(f"[ERROR] 本轮处理失败：{exc}")
            print(
                "[提示] 请确认 resource_16/.env 中的 OPENAI_MODEL 是否为当前网关可用模型；"
                "也可配置 GOOGLE_API_KEY 以启用 Gemini 回退。"
            )


if __name__ == "__main__":
    main()

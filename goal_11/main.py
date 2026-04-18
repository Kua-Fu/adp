"""
goal_11: Goal-driven 目标驱动迭代智能体示例

本示例演示一个常见的“目标优化”工作流：
1) 先根据用户目标生成一个初稿；
2) 再让模型扮演评审，对初稿按目标进行打分与反馈；
3) 若未达标，就根据反馈继续改写；
4) 直到达标或达到最大迭代轮数。

模型策略：
- OpenAI-compatible 优先
- Gemini 回退

运行方式：
- 交互模式：直接运行后输入目标
- 单次模式：设置环境变量 GOAL_PROMPT 后运行
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage


# ------------------------- 环境加载 -------------------------
# 固定从当前模块目录读取 .env，避免在不同 cwd 下运行时读错配置。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)


@dataclass
class ModelBundle:
    """
    同时保存两个候选后端模型对象。

    字段：
    - openai_llm: OpenAI-compatible 模型（优先路径）
    - gemini_llm: Gemini 模型（回退路径）
    """

    openai_llm: Optional[object]
    gemini_llm: Optional[object]


@dataclass
class ReviewResult:
    """
    评审节点的结构化结果。

    字段：
    - passed: 是否达标
    - score: 分数（0-10）
    - strengths: 当前结果的优点摘要
    - weaknesses: 当前结果的不足摘要
    - suggestions: 下一轮改进建议
    """

    passed: bool
    score: int
    strengths: str
    weaknesses: str
    suggestions: str


@dataclass
class IterationResult:
    """
    一次完整目标优化流程的最终输出。

    字段：
    - final_output: 最终交付文本
    - rounds_used: 实际迭代轮数
    - reached_goal: 是否在限制轮数内达标
    - final_review: 最终评审结果
    """

    final_output: str
    rounds_used: int
    reached_goal: bool
    final_review: ReviewResult


def _init_llms() -> tuple[ModelBundle, str]:
    """
    初始化模型（OpenAI-compatible 优先，Gemini 回退）。

    设计说明：
    - 本函数并不会强制二选一，而是尽量把两种模型都准备好；
    - 真正调用时先尝试 OpenAI-compatible，失败再自动切到 Gemini。
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
            # OpenAI-compatible 网关场景下，允许自定义 base_url。
            if openai_base_url:
                kwargs["base_url"] = openai_base_url

            openai_llm = ChatOpenAI(**kwargs)
        except ImportError:
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
        "未找到可用模型凭据。请在 goal_11/.env 中配置 OPENAI_API_KEY（优先）或 GOOGLE_API_KEY（回退）。",
    )


class GoalDrivenAgent:
    """
    目标驱动迭代智能体。

    核心思路：
    - 生成器（Generator）负责按目标产出内容；
    - 评审器（Reviewer）负责按目标打分并给出改进建议；
    - 在“生成 -> 评审 -> 改写”循环中逐轮逼近目标。
    """

    def __init__(self, model_bundle: ModelBundle) -> None:
        self.model_bundle = model_bundle

    def _invoke_with_backend_fallback(self, messages):
        """
        统一模型调用入口：OpenAI-compatible 优先，Gemini 回退。

        返回：
        - (model_response, backend_name)
        """

        def _invoke(llm_obj, backend_name: str):
            result = llm_obj.invoke(messages)
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

    def _generate_initial_draft(self, goal: str) -> tuple[str, str]:
        """
        生成第一版初稿。

        这里不追求一步到位，而是先快速给出可评审版本，
        后续通过评审反馈逐步改进。
        """
        system_prompt = (
            "你是一名擅长目标导向写作的助理。"
            "请严格围绕用户目标，输出结构清晰、可执行、可落地的内容。"
        )

        user_prompt = (
            f"用户目标：{goal}\n\n"
            "请先给出一个较完整的初稿。"
        )

        response, backend = self._invoke_with_backend_fallback(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return str(response.content), backend

    def _review_draft(self, goal: str, draft: str, target_score: int) -> tuple[ReviewResult, str]:
        """
        评审当前草稿并输出结构化反馈。

        约束：
        - 必须输出 JSON（方便程序解析）；
        - 分数字段限定 0-10；
        - 当 score >= target_score 时，passed 应为 true。
        """
        system_prompt = (
            "你是一个严格的评审官。"
            "你需要判断候选内容是否真正满足用户目标，并给出可执行改进建议。"
        )

        user_prompt = (
            f"用户目标：{goal}\n\n"
            f"候选内容：\n{draft}\n\n"
            f"请按 0-10 打分，目标分数为 {target_score}。"
            "仅返回 JSON，不要返回其他文本。\n"
            "JSON schema:\n"
            "{\n"
            '  "passed": boolean,\n'
            '  "score": integer,\n'
            '  "strengths": string,\n'
            '  "weaknesses": string,\n'
            '  "suggestions": string\n'
            "}\n"
            "规则：当 score >= target_score 时，passed=true；否则 passed=false。"
        )

        response, backend = self._invoke_with_backend_fallback(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        review = self._safe_parse_review(str(response.content), target_score=target_score)
        return review, backend

    def _safe_parse_review(self, raw_text: str, target_score: int) -> ReviewResult:
        """
        安全解析评审 JSON。

        兼容策略：
        - 优先直接解析全文；
        - 若模型在 JSON 外多包了一层说明文本，则尝试截取最外层 `{...}` 再解析；
        - 若仍失败，返回一个保守的兜底评审结果。
        """
        payload: dict = {}

        try:
            payload = json.loads(raw_text)
        except Exception:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = raw_text[start : end + 1]
                try:
                    payload = json.loads(snippet)
                except Exception:
                    payload = {}

        if not payload:
            return ReviewResult(
                passed=False,
                score=0,
                strengths="解析失败，未能拿到结构化优点。",
                weaknesses="评审输出不是合法 JSON。",
                suggestions="请评审模型严格按 JSON schema 输出。",
            )

        score_raw = payload.get("score", 0)
        try:
            score = int(score_raw)
        except Exception:
            score = 0

        # 分数做边界收敛，避免模型给出非法范围。
        score = max(0, min(10, score))

        passed_raw = payload.get("passed", False)
        passed = bool(passed_raw)

        # 以业务规则为准：score 达标则 passed 必须为真。
        if score >= target_score:
            passed = True

        return ReviewResult(
            passed=passed,
            score=score,
            strengths=str(payload.get("strengths", "")),
            weaknesses=str(payload.get("weaknesses", "")),
            suggestions=str(payload.get("suggestions", "")),
        )

    def _improve_draft(self, goal: str, draft: str, review: ReviewResult) -> tuple[str, str]:
        """
        根据评审意见改写草稿。

        改写要求：
        - 保留当前版本中有价值的内容；
        - 重点修复 weaknesses 并落实 suggestions；
        - 产出一个“可再次评审”的完整新版本，而不是片段。
        """
        system_prompt = (
            "你是一个擅长根据反馈做高质量改写的助理。"
            "你需要在保持目标一致性的前提下，显著提升内容质量。"
        )

        user_prompt = (
            f"用户目标：{goal}\n\n"
            f"当前草稿：\n{draft}\n\n"
            "评审反馈：\n"
            f"- 优点：{review.strengths}\n"
            f"- 不足：{review.weaknesses}\n"
            f"- 改进建议：{review.suggestions}\n\n"
            "请给出一个完整改写版本。"
        )

        response, backend = self._invoke_with_backend_fallback(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        return str(response.content), backend

    def run_goal_loop(self, goal: str, max_rounds: int = 4, target_score: int = 8) -> IterationResult:
        """
        运行完整目标优化循环。

        参数：
        - goal: 用户目标
        - max_rounds: 最大迭代轮数（含初稿后的评审轮）
        - target_score: 达标分（0-10）

        返回：
        - IterationResult（最终文本 + 最终评审 + 是否达标）
        """
        # 第 0 步：生成初稿
        draft, backend = self._generate_initial_draft(goal)
        print(f"[INFO] 初稿生成后端：{backend}")

        final_review = ReviewResult(
            passed=False,
            score=0,
            strengths="",
            weaknesses="",
            suggestions="",
        )

        # 从第 1 轮开始“评审 ->（必要时）改写”
        for round_index in range(1, max_rounds + 1):
            print(f"\n===== 第 {round_index} 轮评审 =====")
            review, review_backend = self._review_draft(goal, draft, target_score=target_score)
            final_review = review

            print(
                f"[INFO] 评审后端：{review_backend} | "
                f"score={review.score}/10 | passed={review.passed}"
            )

            if review.passed:
                print("[INFO] 已达标，结束迭代。")
                return IterationResult(
                    final_output=draft,
                    rounds_used=round_index,
                    reached_goal=True,
                    final_review=review,
                )

            # 未达标且还有剩余轮次：执行改写
            if round_index < max_rounds:
                draft, improve_backend = self._improve_draft(goal, draft, review)
                print(f"[INFO] 改写后端：{improve_backend}")

        # 超过最大轮数仍未达标：返回最后版本与最后评审。
        return IterationResult(
            final_output=draft,
            rounds_used=max_rounds,
            reached_goal=False,
            final_review=final_review,
        )


def _run_single_goal(agent: GoalDrivenAgent, goal_text: str) -> None:
    """
    执行单个目标并打印结果。

    环境变量可选项：
    - GOAL_MAX_ROUNDS: 最大轮数（默认 4）
    - GOAL_TARGET_SCORE: 达标分（默认 8）
    """
    max_rounds = int(os.getenv("GOAL_MAX_ROUNDS", "4"))
    target_score = int(os.getenv("GOAL_TARGET_SCORE", "8"))

    print("\n================ 目标任务开始 ================")
    print(f"目标：{goal_text}")
    print(f"配置：max_rounds={max_rounds}, target_score={target_score}")

    result = agent.run_goal_loop(goal_text, max_rounds=max_rounds, target_score=target_score)

    print("\n================ 最终结果 ================")
    print(f"是否达标：{result.reached_goal}")
    print(f"使用轮数：{result.rounds_used}")
    print(f"最终分数：{result.final_review.score}/10")
    print(f"最终优点：{result.final_review.strengths}")
    print(f"最终不足：{result.final_review.weaknesses}")
    print(f"最终建议：{result.final_review.suggestions}")
    print("\n--- 最终输出文本 ---")
    print(result.final_output)
    print("==========================================\n")


def main() -> None:
    """
    脚本入口。

    运行模式：
    1) 若设置了 GOAL_PROMPT：执行单次任务并退出；
    2) 否则进入交互模式，持续输入目标，输入 exit/quit/q 退出。
    """
    print("## Running goal_11 goal-driven agent ##")

    bundle, status = _init_llms()
    if bundle.openai_llm is None and bundle.gemini_llm is None:
        print(f"模型初始化失败：{status}")
        return

    print(f"模型初始化状态：{status}")
    agent = GoalDrivenAgent(bundle)

    single_goal = os.getenv("GOAL_PROMPT", "").strip()
    if single_goal:
        _run_single_goal(agent, single_goal)
        return

    print("\n已进入交互模式：请输入你的目标。")
    print("输入 `exit` / `quit` / `q` 可退出。\n")

    while True:
        try:
            goal_text = input("目标: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话已结束。")
            break

        if not goal_text:
            print("请先输入一个明确目标。\n")
            continue

        if goal_text.lower() in {"exit", "quit", "q"}:
            print("会话已结束。")
            break

        try:
            _run_single_goal(agent, goal_text)
        except Exception as exc:
            # 单轮失败不终止整个会话，便于继续尝试其他目标。
            print(f"本轮执行失败：{exc}\n")


if __name__ == "__main__":
    main()

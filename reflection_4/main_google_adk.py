"""
说明：
1. 本文件按照截图内容进行抄录，核心代码保持原样。
2. 该片段用于展示 Google ADK 中的“先写后审”顺序代理编排思路。
3. 这是示例片段，不是可直接运行的完整程序（按你的要求不补全运行代码）。
"""

# 中文注释：导入两个核心代理类型
# - LlmAgent：单个大模型代理
# - SequentialAgent：顺序执行多个子代理
from google.adk.agents import SequentialAgent, LlmAgent

# 中文注释：第一个代理负责生成初稿（写作代理）
# 它会围绕用户主题产出一段简短说明文本，并写入状态键 draft_text。
# The first agent generates the initial draft.
generator = LlmAgent(
    name="DraftWriter",
    description="Generates initial draft content on a given subject.",
    instruction="Write a short, informative paragraph about the user's subject.",
    output_key="draft_text"  # The output is saved to this state key.
)

# 中文注释：第二个代理负责审查第一步生成的文本（事实核查代理）
# 它读取 draft_text，对事实正确性进行判断，并输出结构化结论。
# The second agent critiques the draft from the first agent.
reviewer = LlmAgent(
    name="FactChecker",
    description="Reviews a given text for factual accuracy and provides a structured critique.",
    instruction="""
You are a meticulous fact-checker.
1. Read the text provided in the state key 'draft_text'.
2. Carefully verify the factual accuracy of all claims.
3. Your final output must be a dictionary containing two keys:
   - "status": A string, either "ACCURATE" or "INACCURATE".
   - "reasoning": A string providing a clear explanation for your status, citing specific issues if any are found.
""",
    output_key="review_output"  # The structured dictionary is saved here.
)

# 中文注释：顺序代理管道
# 强制执行顺序为：先 generator，后 reviewer。
# The SequentialAgent ensures the generator runs before the reviewer.
review_pipeline = SequentialAgent(
    name="WriteAndReview_Pipeline",
    sub_agents=[generator, reviewer]
)

# 中文注释：执行流程说明
# 1) generator 运行后把草稿写入 state['draft_text']
# 2) reviewer 读取 draft_text 后输出审查字典到 state['review_output']
# Execution Flow:
# 1. generator runs -> saves its paragraph to state['draft_text'].
# 2. reviewer runs -> reads state['draft_text'] and saves its dictionary output to state['review_output'].

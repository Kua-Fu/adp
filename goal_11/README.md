# goal_11

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Goal%20Loop-1C3C3C?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)

这是一个“目标驱动迭代（Goal-driven Loop）”示例模块，演示：

- 根据用户目标先生成初稿。
- 对初稿进行结构化评审（评分、优点、不足、改进建议）。
- 未达标时根据反馈自动改写，形成闭环优化。
- 模型策略：OpenAI-compatible 优先，Gemini 回退。
- 支持单次模式与交互模式。

## 模块文件说明

- `main.py`：目标驱动智能体主程序（含生成、评审、改写循环）。
- `requirements.txt`：本模块 Python 依赖清单。
- `.env`：模型配置。

## 环境依赖

建议前置条件：

- Python：建议 `3.10 - 3.13`。
- 网络：可访问你配置的模型服务地址。
- 鉴权：至少配置一套可用凭据（OpenAI-compatible 或 Gemini）。

本项目当前指定环境：

```bash
/Users/yz/work/env/adp/.venv/bin/python -V
```

安装依赖：

```bash
/Users/yz/work/env/adp/.venv/bin/pip install -r /Users/yz/work/github/adp/goal_11/requirements.txt
```

## Python 依赖清单

本模块依赖如下（与 `requirements.txt` 保持一致）：

- `python-dotenv`：加载 `.env` 环境变量。
- `langchain-core`：消息对象与模型调用基础。
- `langchain-openai`：OpenAI-compatible 模型封装。
- `langchain-google-genai`：Gemini 模型封装（回退路径）。

## .env 变量说明

在 `goal_11/.env` 中可配置：

```env
# OpenAI-compatible（优先）
OPENAI_API_KEY=your_openai_or_compatible_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0

# Gemini（当 OpenAI-compatible 不可用时回退）
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_TEMPERATURE=0

# 可选：单次模式参数
GOAL_PROMPT=写一个 7 天学习 Python 的可执行计划。
GOAL_MAX_ROUNDS=2
GOAL_TARGET_SCORE=8
```

## 运行方式

### 方式 1：单次模式（推荐）

```bash
cd /Users/yz/work/github/adp/goal_11
GOAL_PROMPT='写一个 7 天学习 Python 的可执行计划。' \
GOAL_MAX_ROUNDS=2 \
GOAL_TARGET_SCORE=8 \
/Users/yz/work/env/adp/.venv/bin/python main.py
```

### 方式 2：交互模式

```bash
cd /Users/yz/work/github/adp/goal_11
/Users/yz/work/env/adp/.venv/bin/python main.py
```

进入后输入目标即可，输入 `exit` / `quit` / `q` 退出。

## 运行输出（本地实测）

运行结果（本地实测，2026-04-18）：

```text
## Running goal_11 goal-driven agent ##
模型初始化状态：仅 OpenAI-compatible 就绪：qwen3-coder-plus

================ 目标任务开始 ================
目标：写一个 7 天学习 Python 的可执行计划。
配置：max_rounds=2, target_score=8
[INFO] 初稿生成后端：OpenAI-compatible

===== 第 1 轮评审 =====
[INFO] 评审后端：OpenAI-compatible | score=9/10 | passed=True
[INFO] 已达标，结束迭代。

================ 最终结果 ================
是否达标：True
使用轮数：1
最终分数：9/10
最终优点：计划结构完整，涵盖Python核心知识点；每日目标明确且循序渐进；理论与实践结合，每天空余时间都安排了练习；学习原则清晰，强调动手实践；包含了具体的项目实战环节；提供了学习资源和检测标准
最终不足：时间分配可能过于紧凑，每天5-7小时高强度学习对初学者挑战较大；缺少具体的学习材料和教程链接；没有考虑个体差异和学习进度调整机制
最终建议：建议提供更灵活的时间安排选项（如周末集中学习版）；增加具体的学习资源链接和参考书籍；添加学习进度自检表；考虑加入在线社区或学习伙伴的建议；为不同背景的学习者提供差异化路径

--- 最终输出文本 ---
# 7天Python学习计划（初稿）

## 总体目标
在7天内掌握Python基础语法，能够编写简单程序，为后续深入学习打下坚实基础。

## 学习原则
- 理论与实践结合，每天至少50%时间用于编程练习
- 循序渐进，每天内容建立在前一天基础上
- 每日设置具体可衡量的学习成果

## 详细计划

### 第1天：Python环境搭建与基础语法
**学习目标**：熟悉Python开发环境，掌握基本数据类型和变量

**上午（2小时）**
- 安装Python 3.x和IDE（推荐PyCharm或VS Code）
- 运行第一个"Hello World"程序
- 了解Python基本语法规则（缩进、注释）

**下午（2小时）**
- 学习基本数据类型：整数、浮点数、字符串、布尔值
- 变量定义和赋值操作
- 基本运算符（算术、比较、逻辑）

**晚上（1小时）**
- 练习题：编写程序计算圆面积、温度转换等
- 复习当天知识点

**当日成果**：能独立编写包含变量和基本运算的小程序

### 第2天：控制流程
**学习目标**：掌握条件判断和循环结构

**上午（2小时）**
- if/elif/else条件语句
- 比较运算符和逻辑运算符应用

**下午（2小时）**
- for循环和while循环
- break和continue语句
- 循环嵌套

**晚上（1小时）**
- 练习题：猜数字游戏、九九乘法表、质数判断
- 复习巩固

**当日成果**：能使用条件和循环编写逻辑控制程序

### 第3天：数据结构基础
**学习目标**：掌握列表、元组、字典等常用数据结构

**上午（2小时）**
- 列表的创建、访问、修改
- 列表方法（append, remove, sort等）
- 列表切片操作

**下午（2小时）**
- 元组的基本操作
- 字典的创建和使用
- 集合的概念和操作

**晚上（1小时）**
- 练习题：学生成绩管理、购物清单、单词计数
- 综合运用各种数据结构

**当日成果**：熟练使用Python内置数据结构处理数据

### 第4天：函数编程
**学习目标**：理解函数概念，学会定义和调用函数

**上午（2小时）**
- 函数定义和调用
- 参数传递（位置参数、关键字参数）
- 返回值处理

**下午（2小时）**
- 局部变量和全局变量
- 递归函数概念
- Lambda表达式

**晚上（1小时）**
- 练习题：计算器程序、字符串处理函数、数学计算函数
- 重构前几天代码，使用函数优化

**当日成果**：能独立设计和实现功能模块化的函数

### 第5天：文件操作与异常处理
**学习目标**：掌握文件读写和错误处理机制

**上午（2小时）**
- 文件打开、读取、写入操作
- 不同文件模式（r, w, a等）
- with语句的使用

**下午（2小时）**
- 异常处理try/except/finally
- 常见异常类型
- 自定义异常

**晚上（1小时）**
- 练习题：文本文件处理、配置文件读写、日志记录程序
- 结合异常处理完善程序健壮性

**当日成果**：能安全地进行文件操作并处理可能出现的错误

### 第6天：面向对象编程入门
**学习目标**：理解类和对象概念，掌握基本OOP思想

**上午（2小时）**
- 类的定义和实例化
- 属性和方法
- 构造函数__init__

**下午（2小时）**
- 封装、继承、多态概念
- 类的特殊方法（__str__, __len__等）

**晚上（1小时）**
- 练习题：设计学生类、银行账户类、图书管理系统
- 综合运用面向对象思想重构项目

**当日成果**：能使用面向对象方式设计简单的类结构

### 第7天：综合项目实战
**学习目标**：整合前6天知识，完成一个完整小项目

**全天（5小时）**
- 项目选择：个人通讯录管理系统 或 简单的待办事项应用
- 项目规划和功能设计
- 分模块实现各个功能
- 测试和调试

**晚上（2小时）**
- 项目总结和代码优化
- 回顾7天学习历程
- 制定后续学习计划

**当日成果**：完成一个功能完整的Python应用程序

## 学习资源推荐
- 官方文档：python.org
- 在线教程：菜鸟教程Python
- 练习平台：LeetCode、HackerRank

## 检测标准
- 每日完成指定练习题
- 能独立解决编程问题
- 第7天项目运行正常
- 掌握基础语法和常用库使用

## 注意事项
- 每天学习时间不少于7小时
- 遇到问题及时查阅资料或寻求帮助
- 保持学习节奏，避免急于求成
- 重视代码规范和注释习惯
==========================================
```

## 常见问题

1. 提示“所有可用模型均调用失败”

- 优先检查 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL` 是否正确。
- 若希望回退 Gemini，请配置 `GOOGLE_API_KEY`。

2. 评审 JSON 解析失败

- 代码内已做兜底解析逻辑；若频繁发生，可降低模型温度并强化提示词约束。

3. 结果达不到目标分

- 适当提高 `GOAL_MAX_ROUNDS`（例如 4~6）。
- 细化目标描述，给出更明确的“成功标准”。

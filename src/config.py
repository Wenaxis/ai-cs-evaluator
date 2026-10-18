"""
全局配置。

规矩：所有路径、模型参数、评分权重都集中在这一个文件里，其他模块一律从这里 import。
不允许在别处硬编码路径或 key —— 工程习惯好不好，一眼看这里。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 读取同目录的 .env（如果存在）。.env 不提交到 Git。
load_dotenv()


# ============================================================
# 一、路径
# ============================================================

# __file__ 是 src/config.py，.parent 是 src/，再 .parent 才是项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # 原始表格，只读不改
PROCESSED_DIR = DATA_DIR / "processed"  # 清洗后的中间产物
KNOWLEDGE_DIR = DATA_DIR / "knowledge"  # 业务规则知识库（v3 新增）
OUTPUT_DIR = BASE_DIR / "output"        # 评测结果、导出的报告

RAW_XLSX = RAW_DIR / "客服实测记录.xlsx"
RECORDS_JSON = PROCESSED_DIR / "records.json"
KNOWLEDGE_JSON = KNOWLEDGE_DIR / "cs_rules.json"


# ============================================================
# 二、原始表的读取参数
# ============================================================

# 这张工作簿里有三个 sheet：统计 / 实测记录 / 操作说明。
# 默认读第一个会读到「统计」，所以必须显式指定 sheet 名。
SHEET_NAME = "实测记录"

# 表头不在第 1 行 —— 第 1 行是大标题，第 2 行是被测平台和测试日期，
# 真正的列名在第 3 行。pandas 的 header 参数是 0 起算的，所以填 2。
HEADER_ROW = 2

# 原始列名 → 统一字段名的映射
COLUMN_MAP = {
    "序号": "id",
    "场景": "scene",
    "我的提问": "question",
    "AI 的回答（摘要）": "answer",
    "结果": "human_label",       # 人工判定：答对 / 答偏
    "错因分类": "human_reason",   # 人工归类的原因
    "我的判断与更好的答法": "human_note",
}

# 合法取值 —— 用来做数据校验，防止脏数据流进下游
VALID_LABELS = {"答对", "答偏"}
VALID_REASONS = {"意图识别错", "语料未覆盖", "知识库缺失", "不适用"}


# ============================================================
# 三、大模型（DAY 3 才用得上）
# ============================================================

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 温度设 0：我们要的是可复现的评分，不是有创意的回答。
LLM_TEMPERATURE = 0.0

# 单条请求超时（秒）与失败重试次数
LLM_TIMEOUT = 60
LLM_MAX_RETRY = 3


# ============================================================
# 四、评分策略
# ============================================================

# 混合评分的权重。规则分偏客观、LLM 分偏语义，所以 LLM 给高一点。
# 这两个数不是拍脑袋定的 —— DAY 4 会用人工标注去校准它们。
WEIGHT_RULE = 0.4
WEIGHT_LLM = 0.6

# 判定阈值：final_score >= 3.5 判为「答对」，否则「答偏」。
# 依据评分量表：4–5 分是回答到位，3 分是「部分相关但答的不是那件事」，属答偏。
PASS_THRESHOLD = 3.5

# 评分量表的满分
MAX_SCORE = 5


# ============================================================
# 五、客服 Agent（v2 新增）
# ============================================================
# 从这里往下都是 v2 才有的配置。上面那些一个都没动 ——
# v1 的链路要能原样跑出 96.67% / Kappa 0.9231，这是所有对比的基准。

# Agent 单次会话最多跑几轮「模型 → 工具 → 模型」。
# 设上限是为了防死循环：模型偶尔会陷进「调工具 → 觉得不够 → 再调同一个工具」的圈子。
AGENT_MAX_ROUNDS = 6

# Agent 生成回复时的温度。比评审用的 0 高一点 ——
# 客服话术需要自然一些，不需要每个字都可复现；但也不能高，否则同一问题两次回答差太多。
AGENT_TEMPERATURE = 0.3

# 判例检索返回几条参考案例。3 条够了：再多会挤占上下文，
# 而且第 4 条往后的相关性已经很低，属于噪声。
PRECEDENT_TOP_K = 3

# 判定「规则类问题」的关键词。用来算一个过程指标：
# 问了「多久 / 能不能 / 怎么操作」却没调判例检索，说明 Agent 跳过了它最该做的动作。
RULE_QUESTION_RE_PATTERN = r"能不能|可不可以|多久|什么时候|多长时间|几天|怎么|如何|在哪里|在哪|要收"


# ============================================================
# 六、检索（v2 升级：从单路 BM25 到双路召回 + 重排）
# ============================================================

# 向量化用的 embedding 接口。**和对话模型分开配** ——
# DeepSeek 没有 embedding 接口，写代码的模型和做检索的模型本来就不该绑在一起。
# 智谱：EMBED_BASE_URL=https://open.bigmodel.cn/api/paas/v4 + EMBED_MODEL=embedding-3
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding-3")

# Chroma 的持久化目录。用本地持久化、不起服务 ——
# 21 条语料起一个向量数据库服务，属于给自己找麻烦。
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
CHROMA_COLLECTION = "cs_precedents"

# 双路召回各自取多少条候选。取多一点是为了给重排留出选择空间 ——
# 召回的活儿是「尽量别漏」，重排的活儿才是「排准」，两件事不该混在一起干。
RECALL_TOP_N = 8

# 重排后最终留给 Agent 几条。3 条够了，再多会挤占上下文还带进噪声。
RERANK_TOP_K = 3


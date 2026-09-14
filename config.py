"""RAG-Pro — 全局配置。

敏感值（端点、密钥）一律通过环境变量注入，杜绝硬编码。
零配置即可运行「纯召回实验 + --pure 构建」demo（本地 BGE embedding，无需任何 API key）。
"""
import os
from pathlib import Path

# ── 项目路径 ──
ROOT = Path(__file__).parent
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DATASETS_DIR = ROOT / "datasets"
OUTPUT_DIR = ROOT / "output"
COLLECTION_SEP = "__"

# ── Embedding 源选择: "local" | "ali" | "default" ──
# local  : 本地 sentence-transformers 模型，无密钥、离线可用（默认，demo 友好）
# ali    : 阿里云 DashScope OpenAI 兼容端点
# default: 自托管 vLLM OpenAI 兼容端点
EMBED_SOURCE = os.getenv("EMBED_SOURCE", "local")

EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL", "BAAI/bge-small-zh-v1.5")
EMBED_DIM_LOCAL = int(os.getenv("EMBED_DIM_LOCAL", "512"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")

EMBED_MODEL_ALI = os.getenv("EMBED_MODEL_ALI", "text-embedding-v4")
EMBED_DIM_ALI = int(os.getenv("EMBED_DIM_ALI", "1024"))
EMBED_BASE_URL_ALI = os.getenv("EMBED_BASE_URL_ALI", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBED_API_KEY_ALI = os.getenv("EMBED_API_KEY_ALI", "")

# ── LLM API（生成 + 评分；纯召回实验与 --pure 构建不需要）──
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")

# ── 入库 LLM（llm_index 策略 + 章节元数据生成，默认复用生成 LLM 配置）──
CHUNK_LLM_BASE_URL = os.getenv("CHUNK_LLM_BASE_URL", LLM_BASE_URL)
CHUNK_LLM_API_KEY = os.getenv("CHUNK_LLM_API_KEY", LLM_API_KEY)
CHUNK_LLM_MODEL = os.getenv("CHUNK_LLM_MODEL", "deepseek-v4-flash")

# ── 默认值 ──
DEFAULT_KB = os.getenv("DEFAULT_KB", "demo")
DEFAULT_TOP_K = 5
BUILD_WORKERS = int(os.getenv("BUILD_WORKERS", "3"))   # 并发入库构建
SEARCH_WORKERS = 5                                      # 召回实验并发检索数
META_WORKERS = 3                                        # 元数据生成并发 LLM 调用数
SCORE_WORKERS = 20                                      # 并发评分
GEN_WORKERS = 3                                         # 并发生成
EMBED_BATCH = 10                                        # Embedding 批次大小
UPSERT_BATCH = 800                                      # Qdrant upsert 批次大小

# ── RRF 融合参数 ──
RRF_K = 60

# ── 评分提示词 ──
PROMPTS = {
    "score_rubric": (
        "你是一个严格的评分专家。根据以下评分标准对 AI 回答进行打分（百分制）。\n\n"
        "## 评分标准\n"
        "1. 必答点（70分）：对照参考要点中的必答点逐一检查，每个必答点未回答扣相应分数。\n"
        "2. 选答点（20分）：选答点有回答则加分，不答不扣分。\n"
        "3. 参考文件（10分）：回答引用了正确的参考文件则得分。\n"
        "4. 多余内容扣分（最多30分）：回答中出现与问题无关的内容，每处扣5分。\n"
        "5. 冲突点：如果回答中存在与参考要点严重矛盾的表述，总分直接为0。\n\n"
        "## 输出格式\n"
        "```json\n"
        "{\n"
        '  "评分理由": [\n'
        '    "[必答]1: 得分/0分 - 原因",\n'
        '    "[选答]1: 得分/0分 - 原因",\n'
        '    "[参考文件]1: 得分/0分 - 原因",\n'
        '    "(surplus)"\n'
        "  ],\n"
        '  "总分": 85\n'
        "}\n"
        "```\n"
        '注意：如果回答中多次出现与参考要点无关的陈述，每个无关要点在评分理由中添加一个 "(surplus)"。'
    ),
    "score_rubric_strict": (
        "你是一个极其严格的评分专家。根据以下评分标准对 AI 回答进行打分（百分制）。\n\n"
        "## 评分标准（严格版）\n"
        "1. 必答点（70分）：必须精确匹配参考要点的核心含义（BERT语义相似度≥85%），结论词必须正确。\n"
        "2. 选答点（20分）：选答点有回答且语义相似度≥80%才得分。\n"
        "3. 参考文件（10分）：回答必须明确引用正确的参考文件名。\n"
        "4. 多余内容扣分（最多30分）：回答中出现与问题无关的内容，每处扣5分。\n"
        "5. 冲突点：如果回答中存在与参考要点严重矛盾的表述，总分直接为0。\n\n"
        "## 输出格式\n"
        "```json\n"
        "{\n"
        '  "评分理由": [\n'
        '    "[必答]1: 得分/0分 - 原因（相似度：XX%）",\n'
        '    "[选答]1: 得分/0分 - 原因（相似度：XX%）",\n'
        '    "[参考文件]1: 得分/0分 - 原因",\n'
        '    "(surplus)"\n'
        "  ],\n"
        '  "总分": 85\n'
        "}\n"
        "```"
    ),
}
"""
配置文件 - 集中管理所有参数
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any

# 项目根目录
BASE_DIR: Path = Path(__file__).parent
f = open(BASE_DIR / "config.json")
configs: Dict[str, Any] = json.load(f)


# 数据路径
RAW_DATA_DIR: Path = BASE_DIR / configs["raw"]
KEYWORDS_FILE: Path = BASE_DIR / configs["keyword"]

# 模型路径
TEACHER_MODEL_PATH: Path = BASE_DIR / configs["models"]["teacher"]
STUDENT_MODEL_PATH: Path = BASE_DIR / configs["models"]["student"]
DEPLOYMENT_MODEL_PATH: Path = BASE_DIR / configs["models"]["deploy"]
LABEL_MODEL_PATH: Path = BASE_DIR / configs["models"]["label"]

# 学科列表
SUBJECTS: List[str] = ["语文", "数学", "英语", "物理", "化学", "生物"]


# 训练配置
class TrainConfig:
    """训练阶段配置"""

    # 特征提取
    MAX_FEATURES = configs["train"]["max_features"]
    MIN_DF = configs["train"]["min_df"]
    MAX_DF = configs["train"]["max_df"]
    NGRAM_RANGE = (configs["train"]["ngram_min"], configs["train"]["ngram_max"])

    # 模型参数
    TEACHER_N_ESTIMATORS = configs["train"]["teacher_n_estimators"]
    TEACHER_MAX_DEPTH = configs["train"]["teacher_max_depth"]
    TEACHER_LEARNING_RATE = configs["train"]["teacher_learning_rate"]

    # 知识蒸馏
    DISTILLATION_TEMPERATURE = configs["train"]["temperature"]
    STUDENT_MAX_ITER = configs["train"]["student_max_iter"]
    STUDENT_C = configs["train"]["student_c"]

    # 特征选择
    SELECTED_FEATURES = configs["train"]["selected_features"]
    KEYWORDS_PER_SUBJECT = configs["train"]["keywords_per_subject"]


# 部署配置
class DeployConfig:
    """部署阶段配置"""

    # 性能限制
    MAX_MEMORY_MB = configs["deploy"]["max_memory_mb"]  # 最大内存使用限制
    MAX_CPU_PERCENT = configs["deploy"]["max_cpu_percent"]  # 最大CPU使用率
    MAX_PROCESSING_SECONDS = configs["deploy"][
        "max_processing_seconds"
    ]  # 单个文件最长处理时间

    # 缓存配置
    CACHE_SIZE = configs["deploy"]["cache_size"]  # 缓存最近处理的文件数
    ENABLE_CACHE = configs["deploy"]["enable_cache"]  # 是否启用缓存

    # 模型压缩
    COMPRESSION_LEVEL = configs["deploy"]["compression_level"]  # 0-9，9为最高压缩
    QUANTIZATION = configs["deploy"]["quantization"]  # 量化精度


# 关键词配置
class KeywordConfig:
    """关键词配置"""

    # 基础关键词（核心术语）
    BASE_KEYWORDS: Dict[str, List[str]] = configs["keywords"]["basic"]

    # 停用词
    STOPWORDS: Set[str] = configs["keywords"]["stopwords"]

    # 关键词提取参数
    MIN_WORD_LENGTH = configs["keywords"]["min_word_len"]
    MAX_WORD_LENGTH = configs["keywords"]["max_word_len"]
    TFIDF_TOP_N = configs["keywords"]["tfidf_top_n"]
    CHI2_TOP_N = configs["keywords"]["chi2_top_n"]


# 路径检查
def check_directories() -> None:
    """检查并创建所需目录"""
    directories: List[Path] = [
        BASE_DIR / "data",
        RAW_DATA_DIR,
        BASE_DIR / "utils",
        BASE_DIR / "scripts",
        BASE_DIR / "models",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ 目录已就绪: {directory}")


# 初始化检查
check_directories()

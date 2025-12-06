"""
配置文件 - 集中管理所有参数
"""

import os
from pathlib import Path
from typing import Dict, List, Set

# 项目根目录
BASE_DIR: Path = Path(__file__).parent

# 数据路径
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
KEYWORDS_FILE: Path = PROCESSED_DATA_DIR / "subject_keywords.json"

# 模型路径
MODELS_DIR: Path = BASE_DIR / "models"
TEACHER_MODEL_PATH: Path = MODELS_DIR / "teacher_model.joblib"
STUDENT_MODEL_PATH: Path = MODELS_DIR / "student_model.joblib"
DEPLOYMENT_MODEL_PATH: Path = MODELS_DIR / "deployment_model.joblib"

# 学科列表
SUBJECTS: List[str] = ["语文", "数学", "英语", "物理", "化学", "生物"]


# 训练配置
class TrainConfig:
    """训练阶段配置"""

    # 特征提取
    MAX_FEATURES = 1000
    MIN_DF = 2
    MAX_DF = 0.9
    NGRAM_RANGE = (1, 2)

    # 模型参数
    TEACHER_N_ESTIMATORS = 100
    TEACHER_MAX_DEPTH = 6
    TEACHER_LEARNING_RATE = 0.1

    # 知识蒸馏
    DISTILLATION_TEMPERATURE = 2.0
    STUDENT_MAX_ITER = 500
    STUDENT_C = 0.3

    # 特征选择
    SELECTED_FEATURES = 300
    KEYWORDS_PER_SUBJECT = 50


# 部署配置
class DeployConfig:
    """部署阶段配置"""

    # 性能限制
    MAX_MEMORY_MB = 1500  # 最大内存使用限制
    MAX_CPU_PERCENT = 80  # 最大CPU使用率
    MAX_PROCESSING_SECONDS = 10  # 单个文件最长处理时间

    # 缓存配置
    CACHE_SIZE = 100  # 缓存最近处理的文件数
    ENABLE_CACHE = True

    # 模型压缩
    COMPRESSION_LEVEL = 9  # 0-9，9为最高压缩
    QUANTIZATION = "float16"  # 量化精度


# 关键词配置
class KeywordConfig:
    """关键词配置"""

    # 基础关键词（核心术语）
    BASE_KEYWORDS: Dict[str, List[str]] = {
        "语文": ["诗歌", "散文", "文言文", "修辞", "作文", "标点", "鲁迅", "杜甫"],
        "数学": ["函数", "方程", "几何", "概率", "导数", "积分", "矩阵", "证明"],
        "英语": [
            "grammar",
            "vocabulary",
            "reading",
            "writing",
            "tense",
            "essay",
            "paragraph",
        ],
        "物理": ["力", "加速度", "电场", "磁场", "电路", "能量", "动量", "量子"],
        "化学": ["元素", "分子", "反应", "化学式", "溶液", "酸碱", "有机", "实验"],
        "生物": ["细胞", "基因", "DNA", "蛋白质", "生态系统", "进化", "遗传", "酶"],
    }

    # 停用词
    STOPWORDS: Set[str] = {
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
        "chapter",
        "page",
        "ppt",
        "slide",
    }

    # 关键词提取参数
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 10
    TFIDF_TOP_N = 50
    CHI2_TOP_N = 30


# 路径检查
def check_directories() -> None:
    """检查并创建所需目录"""
    directories: List[Path] = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        BASE_DIR / "utils",
        BASE_DIR / "scripts",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ 目录已就绪: {directory}")


# 初始化检查
check_directories()

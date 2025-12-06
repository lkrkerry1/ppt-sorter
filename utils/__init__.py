"""
PPT学科分类器 - 工具模块
"""

# 导入关键模块，方便外部使用
from .ppt_parser import PPTParser
from .text_processor import TextProcessor
from .keyword_manager import KeywordManager
from .data_loader import PPTDataset
from .evaluation import ModelEvaluator

__version__ = "1.0.0"
__all__ = [
    "PPTParser",
    "TextProcessor",
    "KeywordManager",
    "PPTDataset",
    "ModelEvaluator",
]

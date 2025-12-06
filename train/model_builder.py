"""模型构建器占位文件。
实现模型定义、训练、保存接口。
"""

from sklearn.base import BaseEstimator


def build_model(config: dict) -> BaseEstimator:
    """根据配置构建并返回模型（占位）。"""
    raise NotImplementedError("请实现 model 构建逻辑")

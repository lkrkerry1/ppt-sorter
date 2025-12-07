"""
模型构建器 - 构建和训练各种模型
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from typing import Any, Dict, Optional

from config import TrainConfig


class ModelBuilder:
    """模型构建器"""

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}

    def train_teacher_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        use_gpu: bool = False,
    ) -> Any:
        """
        训练教师模型（复杂模型）

        Args:
            X: 特征矩阵
            y: 标签（应为数值标签）
            n_estimators: 树的数量
            use_gpu: 是否使用GPU

        Returns:
            训练好的模型
        """
        print("   正在训练XGBoost教师模型...")

        # 确保y是整数类型
        if y.dtype.kind not in ("i", "u"):  # 如果不是整数类型
            y = y.astype(int)

        # XGBoost参数
        params = {
            "n_estimators": n_estimators,
            "max_depth": TrainConfig.TEACHER_MAX_DEPTH,
            "learning_rate": TrainConfig.TEACHER_LEARNING_RATE,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # GPU加速（如果可用）
        if use_gpu:
            try:
                # 关键修改点：移除 'gpu_id'，添加 'device' 和 'tree_method'
                params["device"] = "cuda"  # 或 "gpu"，新参数用于指定设备[citation:4]
                params["tree_method"] = (
                    "hist"  # 或 "gpu_hist"，指定使用GPU的算法[citation:9]
                )
                print("   使用GPU加速训练 (XGBoost >= 3.1)")
            except:
                print("   警告: GPU不可用，回退到CPU")
        # 如果 use_gpu 为 False，则无需指定 device，默认使用 CPU

        # 训练模型
        model = XGBClassifier(**params)
        model.fit(X, y)

        return model

    def train_student_model(
        self, X: np.ndarray, y: np.ndarray, model_type: str = "logistic"
    ) -> Any:
        """
        训练学生模型（轻量模型）

        Args:
            X: 特征矩阵
            y: 标签（应为数值标签）
            model_type: 模型类型 ('logistic', 'random_forest', 'lightgbm')

        Returns:
            训练好的模型
        """
        print(f"   正在训练{model_type}学生模型...")

        # 确保y是整数类型
        if y.dtype.kind not in ("i", "u"):  # 如果不是整数类型
            y = y.astype(int)

        if model_type == "logistic":
            model = LogisticRegression(
                C=TrainConfig.STUDENT_C,
                max_iter=TrainConfig.STUDENT_MAX_ITER,
                solver="liblinear",
                penalty="l1",
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
        elif model_type == "lightgbm":
            model = LGBMClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        model.fit(X, y)
        return model

    def save_model(self, model: Any, filepath: str):
        """保存模型到文件"""
        joblib.dump(model, filepath)
        print(f"   模型已保存到: {filepath}")

    def load_model(self, filepath: str) -> Any:
        """从文件加载模型"""
        return joblib.load(filepath)

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "type": type(model).__name__,
            "parameters": model.get_params() if hasattr(model, "get_params") else {},
        }

        # 添加模型特定信息
        if hasattr(model, "n_estimators"):
            info["n_estimators"] = model.n_estimators
        if hasattr(model, "n_features_in_"):
            info["n_features"] = model.n_features_in_
        if hasattr(model, "classes_"):
            info["n_classes"] = len(model.classes_)
            info["classes"] = model.classes_.tolist()

        return info

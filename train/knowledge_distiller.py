"""
知识蒸馏器 - 将复杂模型的知识转移到简单模型
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
from typing import Any, Tuple, Optional

from config import TrainConfig


class KnowledgeDistiller:
    """知识蒸馏器"""

    def __init__(self, teacher_model: Any) -> None:
        """
        初始化蒸馏器

        Args:
            teacher_model: 教师模型
        """
        self.teacher_model: Any = teacher_model
        self.student_model: Optional[Any] = None

    def distill(
        self,
        X: np.ndarray,
        y: np.ndarray,
        temperature: float = 2.0,
        student_type: str = "logistic",
    ) -> Any:
        """
        执行知识蒸馏

        Args:
            X: 训练特征
            y: 训练标签
            temperature: 蒸馏温度
            student_type: 学生模型类型

        Returns:
            蒸馏后的学生模型
        """
        print(f"   开始知识蒸馏 (温度={temperature})...")

        # 获取教师模型的软标签
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            teacher_probs = self.teacher_model.predict_proba(X)

        # 应用温度缩放
        if temperature != 1.0:
            teacher_probs = self._apply_temperature(teacher_probs, temperature)

        # 训练学生模型
        if student_type == "logistic":
            self.student_model = LogisticRegression(
                C=TrainConfig.STUDENT_C,
                max_iter=TrainConfig.STUDENT_MAX_ITER,
                solver="liblinear",
                multi_class="ovr",
                random_state=42,
            )
        elif student_type == "decision_tree":
            self.student_model = DecisionTreeClassifier(
                max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42
            )
        else:
            raise ValueError(f"未知的学生模型类型: {student_type}")

        # 使用软标签训练学生模型
        self.student_model.fit(X, teacher_probs.argmax(axis=1))

        print(f"   知识蒸馏完成")
        return self.student_model

    def _apply_temperature(
        self, probabilities: np.ndarray, temperature: float
    ) -> np.ndarray:
        """应用温度缩放"""
        # 避免数值不稳定
        probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)

        # 应用温度
        scaled = np.power(probabilities, 1.0 / temperature)

        # 重新归一化
        scaled = scaled / scaled.sum(axis=1, keepdims=True)

        return scaled

    def evaluate_distillation(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> Tuple[float, float]:
        """
        评估蒸馏效果

        Returns:
            (教师准确率, 学生准确率)
        """
        if self.student_model is None:
            raise ValueError("请先执行蒸馏")

        # 教师模型准确率
        teacher_preds = self.teacher_model.predict(X_val)
        teacher_acc = np.mean(teacher_preds == y_val)

        # 学生模型准确率
        student_preds = self.student_model.predict(X_val)
        student_acc = np.mean(student_preds == y_val)

        return teacher_acc, student_acc

    def get_agreement_rate(self, X_test: np.ndarray) -> float:
        """
        获取师生模型一致率

        Returns:
            一致率 (0-1)
        """
        if self.student_model is None:
            raise ValueError("请先执行蒸馏")
        teacher_preds = self.teacher_model.predict(X_test)
        student_preds = self.student_model.predict(X_test)

        agreement = np.mean(teacher_preds == student_preds)
        return agreement

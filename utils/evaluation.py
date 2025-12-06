"""
评估工具 - 评估模型性能
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Any, Tuple, List, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model: Any, class_names: List[str]):
        """
        初始化评估器

        Args:
            model: 要评估的模型
            class_names: 类别名称列表
        """
        self.model = model
        self.class_names = class_names

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """评估模型准确率"""
        y_pred = self.model.predict(X_test)
        accuracy: float = float(accuracy_score(y_test, y_pred))  # 显式转换为float
        return accuracy

    def detailed_evaluation(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """详细评估"""
        y_pred = self.model.predict(X_test)

        # 计算指标
        accuracy_val = accuracy_score(y_test, y_pred)
        precision_macro_val = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        recall_macro_val = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        f1_macro_val = f1_score(y_test, y_pred, average="macro", zero_division=0)

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_val),
            "precision_macro": float(precision_macro_val),
            "recall_macro": float(recall_macro_val),
            "f1_macro": float(f1_macro_val),
        }

        # 每个类别的指标，确保转换为numpy数组再转换为列表
        precision_per_class = precision_score(
            y_test, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        metrics["precision_per_class"] = precision_per_class.tolist()  # type: ignore
        metrics["recall_per_class"] = recall_per_class.tolist()  # type: ignore
        metrics["f1_per_class"] = f1_per_class.tolist()  # type: ignore

        # 混淆矩阵
        metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

        # 分类报告
        report_dict = classification_report(
            y_test, y_pred, target_names=self.class_names, output_dict=True
        )
        metrics["classification_report"] = report_dict

        return metrics

    def plot_confusion_matrix(
        self, X_test: np.ndarray, y_test: np.ndarray, save_path: Optional[str] = None
    ):
        """绘制混淆矩阵"""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"混淆矩阵已保存到: {save_path}")

        plt.show()

    def compare_models(
        self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """比较多个模型"""
        results: Dict[str, Dict[str, float]] = {}

        for name, model in models_dict.items():
            evaluator = ModelEvaluator(model, self.class_names)
            metrics = evaluator.detailed_evaluation(X_test, y_test)
            results[name] = {
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }

        return results

    def plot_model_comparison(
        self, models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray
    ):
        """绘制模型比较图"""
        results = self.compare_models(models_dict, X_test, y_test)

        models = list(results.keys())
        accuracies = [results[m]["accuracy"] for m in models]
        f1_scores = [results[m]["f1_macro"] for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy")
        rects2 = ax.bar(x + width / 2, f1_scores, width, label="F1 Score")

        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim((0.0, 1.0))  # 修复：改为浮点数元组

        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        autolabel(rects1)
        autolabel(rects2)

        plt.tight_layout()
        plt.show()

    def analyze_errors(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_test_raw: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """分析错误分类"""
        y_pred = self.model.predict(X_test)
        errors: List[Dict[str, Any]] = []

        for i in range(len(y_test)):
            if y_test[i] != y_pred[i]:
                error_info: Dict[str, Any] = {
                    "true_label": self.class_names[y_test[i]],
                    "predicted_label": self.class_names[y_pred[i]],
                    "true_prob": None,
                    "pred_prob": None,
                }

                # 如果有原始数据，添加文本片段
                if X_test_raw is not None and i < len(X_test_raw):
                    # 截取前100字符作为示例
                    sample_text = str(X_test_raw[i])
                    if len(sample_text) > 100:
                        sample_text = sample_text[:100] + "..."
                    error_info["sample"] = sample_text

                # 如果有概率预测
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X_test[i : i + 1])[0]
                    error_info["true_prob"] = float(probs[y_test[i]])
                    error_info["pred_prob"] = float(probs[y_pred[i]])

                errors.append(error_info)

        return errors

    def print_error_analysis(self, errors: List[Dict[str, Any]], top_n: int = 20):
        """打印错误分析"""
        if not errors:
            print("没有错误分类！")
            return

        print(f"\n错误分析 (前{min(top_n, len(errors))}个):")
        print("=" * 80)
        print(
            f"{'真实标签':<10} {'预测标签':<10} {'真实概率':<10} {'预测概率':<10} {'示例'}"
        )
        print("-" * 80)

        for i, error in enumerate(errors[:top_n]):
            true_prob = (
                f"{error['true_prob']:.3f}" if error["true_prob"] is not None else "N/A"
            )
            pred_prob = (
                f"{error['pred_prob']:.3f}" if error["pred_prob"] is not None else "N/A"
            )
            sample = error.get("sample", "")

            print(
                f"{error['true_label']:<10} {error['predicted_label']:<10} "
                f"{true_prob:<10} {pred_prob:<10} {sample}"
            )

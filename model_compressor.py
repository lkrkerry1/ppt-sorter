"""
ModelCompressor - 简化示例实现

此模块提供一个可运行的示例 `ModelCompressor`，用于把训练得到的模型与向量器
压缩为适合部署的 `joblib` 包。示例实现保守且无需 GPU 即可运行。
"""

import os
import joblib
import numpy as np


class ModelCompressor:
    def __init__(self, compression_level="high"):
        self.compression_level = compression_level
        self.label_mapping = None

    def compress_for_deployment(
        self, trained_model, vectorizer, X_sample, y_sample, output_path, top_k=200
    ):
        """将训练好的模型和向量器压缩为部署格式。

        参数说明：
        - trained_model: 训练好的 sklearn 风格模型
        - vectorizer: 原始文本向量器（CountVectorizer/TfidfVectorizer）
        - X_sample: 样本特征矩阵（稀疏或数组）用于统计信息
        - y_sample: 标签数组（用于构建关键词）
        - output_path: 输出 joblib 路径
        - top_k: 保留的特征数量
        """

        print("开始压缩模型...")

        # 1. 简单蒸馏：如果模型为树或复杂模型，我们尝试训练一个小的逻辑回归
        student = self._distill_model(trained_model, X_sample, y_sample)

        # 2. 选择重要特征
        important_idx = self._select_important_features(student, X_sample, top_k=top_k)

        # 3. 构建关键词匹配器（基于 vectorizer 的词汇与样本统计）
        keyword_matcher = self._build_keyword_matcher(vectorizer, X_sample, y_sample)

        # 4. 量化模型参数为 float16（若有）
        quantized = self._quantize_model(student)

        # 5. 组成部署包
        compressed = {
            "model": quantized,
            "vectorizer": vectorizer,
            "feature_mask": np.asarray(important_idx),
            "keyword_matcher": keyword_matcher,
            "precomputed": {
                # 标准化 label_mapping 为 index -> label 的列表
                "label_mapping": self._normalize_label_mapping(self.label_mapping)
            },
        }

        joblib.dump(compressed, output_path, compress=3)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"压缩完成: {output_path} ({size_mb:.2f} MB)")
        return compressed

    def _distill_model(self, teacher_model, X, y):
        # 如果 teacher_model 本身足够小则直接返回
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            return teacher_model

        try:
            # 训练逻辑回归作为学生模型（若 teacher 支持 predict_proba）
            if hasattr(teacher_model, "predict_proba"):
                teacher_preds = teacher_model.predict_proba(X)
                y_soft = teacher_preds.argmax(axis=1)
            else:
                y_soft = y

            student = LogisticRegression(
                C=0.3, max_iter=500, solver="liblinear", penalty="l1"
            )
            student.fit(X, y_soft)
            return student
        except Exception:
            return teacher_model

    def _select_important_features(self, model, X_sample, top_k=200):
        try:
            if hasattr(model, "coef_"):
                coefs = np.abs(model.coef_)
                if coefs.ndim > 1:
                    importances = coefs.max(axis=0)
                else:
                    importances = coefs
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                importances = np.asarray(X_sample).var(axis=0)
        except Exception:
            importances = np.asarray(X_sample).var(axis=0)

        idx = np.argsort(importances)[-min(top_k, int(len(importances))) :]
        return np.sort(idx)

    def _build_keyword_matcher(self, vectorizer, X_sample, y_sample, top_words=10):
        matcher = {}
        try:
            vocab = None
            if hasattr(vectorizer, "get_feature_names_out"):
                vocab = vectorizer.get_feature_names_out()
            elif hasattr(vectorizer, "vocabulary_"):
                inv = {v: k for k, v in vectorizer.vocabulary_.items()}
                vocab = [inv[i] for i in range(len(inv))]

            if vocab is None:
                return matcher

            labels = np.unique(y_sample)
            for lab in labels:
                idxs = np.where(y_sample == lab)[0]
                if len(idxs) == 0:
                    continue
                submat = X_sample[idxs]
                # 兼容稀疏矩阵
                try:
                    freqs = np.asarray(submat.sum(axis=0)).ravel()
                except Exception:
                    freqs = np.asarray(submat).sum(axis=0).ravel()

                top_idx = np.argsort(freqs)[-top_words:]
                keywords = [vocab[i] for i in top_idx if i < len(vocab)]
                matcher[str(lab)] = keywords
        except Exception:
            pass

        return matcher

    def _quantize_model(self, model):
        try:
            if hasattr(model, "coef_"):
                model.coef_ = model.coef_.astype(np.float16)
            if hasattr(model, "intercept_"):
                model.intercept_ = model.intercept_.astype(np.float16)
        except Exception:
            pass
        return model

    def _normalize_label_mapping(self, label_mapping):
        """把 label_mapping 标准化为 index->label 的列表。

        支持的输入格式：
        - None -> []
        - 列表（index->label）-> 直接返回
        - dict: 两种可能形式
            1) {index: label, ...} 或 {"0": "语文", ...}
            2) {label: index, ...} -> 会被反转为 index->label 列表
        """
        if label_mapping is None:
            return []

        # 如果已经是列表
        if isinstance(label_mapping, (list, tuple)):
            return list(label_mapping)

        if isinstance(label_mapping, dict):
            # 判断 dict 的值是否为 int（label->index 或 index->label）
            # 如果值为 int，则可能是 label->index，需要反转
            vals = list(label_mapping.values())
            keys = list(label_mapping.keys())

            if all(isinstance(v, int) for v in vals):
                # label -> index
                max_idx = max(vals) if vals else -1
                arr = [""] * (max_idx + 1)
                for k, v in label_mapping.items():
                    arr[int(v)] = str(k)
                return arr

            # 如果 keys 看起来像整数索引（string int 或 int), treat as index->label
            try:
                int_keys = [int(k) for k in keys]
                max_idx = max(int_keys)
                arr = [""] * (max_idx + 1)
                for k, v in label_mapping.items():
                    arr[int(k)] = str(v)
                return arr
            except Exception:
                pass

        # 兜底
        return []


if __name__ == "__main__":
    print("ModelCompressor module - 示例实现")

"""
ModelCompressor - 简化示例实现 (moved to src/)
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
        """将训练好的模型和向量器压缩为部署格式。"""

        print("开始压缩模型...")

        student = self._distill_model(trained_model, X_sample, y_sample)

        important_idx = self._select_important_features(student, X_sample, top_k=top_k)

        keyword_matcher = self._build_keyword_matcher(vectorizer, X_sample, y_sample)

        quantized = self._quantize_model(student)

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
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception:
            return teacher_model

        try:
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
        if label_mapping is None:
            return []

        if isinstance(label_mapping, (list, tuple)):
            return list(label_mapping)

        if isinstance(label_mapping, dict):
            vals = list(label_mapping.values())
            keys = list(label_mapping.keys())

            if all(isinstance(v, int) for v in vals):
                max_idx = max(vals) if vals else -1
                arr = [""] * (max_idx + 1)
                for k, v in label_mapping.items():
                    arr[int(v)] = str(k)
                return arr

            try:
                int_keys = [int(k) for k in keys]
                max_idx = max(int_keys)
                arr = [""] * (max_idx + 1)
                for k, v in label_mapping.items():
                    arr[int(k)] = str(v)
                return arr
            except Exception:
                pass

        return []


if __name__ == "__main__":
    print("ModelCompressor module - 示例实现")

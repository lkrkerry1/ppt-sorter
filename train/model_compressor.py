"""
模型压缩器 - 压缩模型以减小大小和提高推理速度
"""

import numpy as np
import joblib
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import csr_matrix
import gzip
import pickle
from typing import Any, Dict, List, Optional

from config import DeployConfig, KeywordConfig, TrainConfig


class ModelCompressor:
    """模型压缩器"""

    def __init__(self, compression_level: int = 9) -> None:
        """
        初始化压缩器

        Args:
            compression_level: 压缩级别 (0-9)
        """
        self.compression_level: int = compression_level

    def compress(
        self, model: Any, X_sample: np.ndarray, output_path: str
    ) -> Dict[str, Any]:
        """
        压缩模型为部署格式

        Args:
            model: 待压缩的模型
            X_sample: 样本数据（用于特征选择和量化）
            output_path: 输出路径

        Returns:
            压缩后的模型数据
        """
        print(f"   开始模型压缩 (级别={self.compression_level})...")

        # 1. 特征选择（仅保留重要特征）
        if hasattr(model, "coef_"):
            # 逻辑回归：根据权重选择
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_).max(axis=0)
            else:
                importances = np.abs(model.coef_)

            # 选择top_k特征
            k = min(TrainConfig.COMPRESS_DIM, X_sample.shape[1])
            top_indices = np.argsort(importances)[-k:]
        else:
            # 默认保留所有特征
            top_indices = np.arange(X_sample.shape[1])

        print(f"   特征选择: {X_sample.shape[1]} -> {len(top_indices)}")

        # 2. 量化模型参数
        quantized_model = self._quantize_model(model)

        # 3. 构建关键词快速匹配器
        keyword_matcher = self._build_keyword_matcher()

        # 4. 准备压缩数据
        compressed_data = {
            "model": quantized_model,
            "feature_indices": top_indices,
            "keyword_matcher": keyword_matcher,
            "compression_info": {
                "level": self.compression_level,
                "original_size": self._estimate_model_size(model),
                "feature_count": len(top_indices),
                "quantization": DeployConfig.QUANTIZATION,
            },
        }

        # 5. 保存压缩模型
        self._save_compressed(compressed_data, output_path)

        compressed_size = self._get_file_size(output_path)
        print(f"   压缩完成: {compressed_size:.2f} MB")

        return compressed_data

    def _quantize_model(self, model: Any) -> Any:
        """量化模型参数"""
        # 创建模型的副本
        import copy

        quantized_model = copy.deepcopy(model)

        # 量化逻辑回归权重
        if hasattr(quantized_model, "coef_"):
            if DeployConfig.QUANTIZATION == "float16":
                quantized_model.coef_ = quantized_model.coef_.astype(np.float16)
            elif DeployConfig.QUANTIZATION == "float32":
                quantized_model.coef_ = quantized_model.coef_.astype(np.float32)

        # 量化截距
        if hasattr(quantized_model, "intercept_"):
            if DeployConfig.QUANTIZATION == "float16":
                quantized_model.intercept_ = quantized_model.intercept_.astype(
                    np.float16
                )
            elif DeployConfig.QUANTIZATION == "float32":
                quantized_model.intercept_ = quantized_model.intercept_.astype(
                    np.float32
                )

        return quantized_model

    def _build_keyword_matcher(self) -> Dict[str, List[str]]:
        """构建关键词快速匹配器"""
        return KeywordConfig.BASE_KEYWORDS

    def _estimate_model_size(self, model: Any) -> float:
        """估算模型大小（MB）"""
        try:
            # 临时保存以估算大小
            temp_path = "temp_model.joblib"
            joblib.dump(model, temp_path)
            import os

            size = os.path.getsize(temp_path) / 1024 / 1024
            os.remove(temp_path)
            return size
        except:
            return 0.0

    def _save_compressed(self, data: Dict[str, Any], output_path: str):
        """保存压缩数据"""
        # 使用最高压缩级别
        if self.compression_level >= 8:
            # 使用gzip压缩
            with gzip.open(output_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            # 使用joblib压缩
            joblib.dump(data, output_path, compress=("lz4", self.compression_level))  # type: ignore

    def _get_file_size(self, filepath: str) -> float:
        """获取文件大小（MB）"""
        import os

        return os.path.getsize(filepath) / 1024 / 1024

    def create_ultra_light_model(self, model: Any, top_features: int = 100) -> Any:
        """
        创建超轻量模型（仅保留最重要的特征）

        Args:
            model: 原始模型
            top_features: 保留的特征数

        Returns:
            超轻量模型
        """
        print(f"   创建超轻量模型 (特征数={top_features})...")

        # 这里简化为返回原始模型
        # 实际实现需要根据特征重要性裁剪模型
        return model

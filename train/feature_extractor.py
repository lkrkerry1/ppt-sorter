"""
高级特征提取器 - 在强训练机上使用
支持文本、统计和深度学习特征
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter
import re
import jieba
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings

from config import TrainConfig, KeywordConfig
from utils.text_processor import TextProcessor
from scipy import sparse  # 添加sparse导入


class AdvancedFeatureExtractor:
    """高级特征提取器"""

    def __init__(self, use_gpu: bool = False):
        """
        初始化特征提取器

        Args:
            use_gpu: 是否使用GPU加速深度学习特征提取
        """
        self.use_gpu = use_gpu
        self.text_processor = TextProcessor()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.selector: Optional[SelectKBest] = None
        self.feature_names: Optional[np.ndarray] = None  # 修改为np.ndarray类型

        # 深度学习模型（可选）
        self.deep_model = None
        if use_gpu:
            self._init_deep_models()

    def _init_deep_models(self):
        """初始化深度学习模型（如果有GPU）"""
        try:
            import torch
            import torch.nn as nn
            from transformers import BertTokenizer, BertModel

            # 检查CUDA可用性
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("   GPU可用，初始化深度学习特征提取器")

                # 加载BERT模型
                self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
                self.bert_model = BertModel.from_pretrained("bert-base-chinese")
                self.bert_model.to(self.device)  # type: ignore
                self.bert_model.eval()
            else:
                self.device = torch.device("cpu")
                self.bert_model = None
        except ImportError:
            print("  警告: 未安装torch/transformers，跳过深度学习特征")
            self.bert_model = None

    def extract_features(self, ppt_files: List[str], mode: str = "train") -> np.ndarray:
        """
        提取特征

        Args:
            ppt_files: PPT文件路径列表
            mode: 'train'或'test'

        Returns:
            特征矩阵
        """
        # 提取文本
        print(f"   正在处理 {len(ppt_files)} 个PPT文件...")
        texts = [self._extract_text_from_ppt(file_path) for file_path in ppt_files]

        # 文本预处理
        processed_texts = [self.text_processor.process(text) for text in texts]

        # 提取传统特征
        if mode == "train":
            features = self._extract_traditional_features_train(processed_texts)
        else:
            features = self._extract_traditional_features_test(processed_texts)

        # 提取深度学习特征（如果可用）
        if self.bert_model is not None and self.use_gpu:
            deep_features = self._extract_deep_features(texts)
            features = np.hstack([features, deep_features])

        # 特征选择
        if mode == "train" and features.shape[1] > TrainConfig.SELECTED_FEATURES:
            features = self._select_features(features)

        return features

    def _extract_text_from_ppt(self, file_path: str) -> str:
        """从PPT提取文本"""
        try:
            from utils.ppt_parser import PPTParser

            parser = PPTParser()
            return parser.extract_text(file_path, max_slides=30)
        except Exception as e:
            print(f"  警告: 无法读取 {file_path}: {str(e)}")
            return ""

    def _extract_traditional_features_train(self, texts: List[str]) -> np.ndarray:
        """训练阶段提取传统特征"""
        # TF-IDF特征
        self.vectorizer = TfidfVectorizer(
            max_features=TrainConfig.MAX_FEATURES,
            min_df=TrainConfig.MIN_DF,
            max_df=TrainConfig.MAX_DF,
            ngram_range=TrainConfig.NGRAM_RANGE,
        )

        tfidf_features = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # 统计特征
        stat_features = self._extract_statistical_features(texts)

        # 关键词特征
        keyword_features = self._extract_keyword_features(texts)

        # 合并所有特征
        all_features = np.hstack(
            [tfidf_features.toarray(), stat_features, keyword_features]  # type: ignore
        )

        return all_features

    def _extract_traditional_features_test(self, texts: List[str]) -> np.ndarray:
        """测试阶段提取传统特征"""
        if self.vectorizer is None:
            raise ValueError("请先调用训练模式提取特征")

        # TF-IDF特征
        tfidf_features = self.vectorizer.transform(texts)

        # 统计特征
        stat_features = self._extract_statistical_features(texts)

        # 关键词特征
        keyword_features = self._extract_keyword_features(texts)

        # 合并所有特征
        all_features = np.hstack(
            [tfidf_features.toarray(), stat_features, keyword_features]  # type:ignore
        )

        return all_features

    def _extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """提取统计特征"""
        features = []

        for text in texts:
            # 基础统计
            text_length = len(text)
            word_count = len(text.split())
            avg_word_length = text_length / max(word_count, 1)

            # 特殊字符统计
            chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
            english_chars = len(re.findall(r"[a-zA-Z]", text))
            digit_chars = len(re.findall(r"\d", text))
            symbol_chars = len(re.findall(r"[^\w\s]", text))

            # 比例特征
            chinese_ratio = chinese_chars / max(text_length, 1)
            english_ratio = english_chars / max(text_length, 1)
            digit_ratio = digit_chars / max(text_length, 1)

            features.append(
                [
                    text_length,
                    word_count,
                    avg_word_length,
                    chinese_chars,
                    english_chars,
                    digit_chars,
                    symbol_chars,
                    chinese_ratio,
                    english_ratio,
                    digit_ratio,
                ]
            )

        return np.array(features)

    def _extract_keyword_features(self, texts: List[str]) -> np.ndarray:
        """提取关键词特征"""
        features = []

        for text in texts:
            text_lower = text.lower()
            subject_features = []

            for subject, keywords in KeywordConfig.BASE_KEYWORDS.items():
                # 统计关键词出现次数
                count = sum(1 for keyword in keywords if keyword in text_lower)
                subject_features.append(count)

            features.append(subject_features)

        return np.array(features)

    def _extract_deep_features(self, texts: List[str]) -> np.ndarray:
        """提取深度学习特征"""
        if self.bert_model is None:
            return np.zeros((len(texts), 0))

        import torch
        from transformers import BertTokenizer

        features = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # 移动到GPU（如果可用）
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 获取BERT输出
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # 使用[CLS] token的表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(cls_embeddings)

        return np.vstack(features)

    def _select_features(self, features: np.ndarray) -> np.ndarray:
        """特征选择"""
        # 使用方差选择法
        variances = np.var(features, axis=0)
        top_indices = np.argsort(variances)[-TrainConfig.SELECTED_FEATURES :]

        return features[:, top_indices]

    def get_feature_importance(self, model: Any) -> Dict[str, float]:
        """获取特征重要性"""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # 处理coef_可能是一维或二维的情况
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
        else:
            return {}

        # 映射特征名
        importance_dict: Dict[str, float] = {}
        if self.feature_names is not None and len(self.feature_names) == len(
            importances
        ):
            for name, importance in zip(self.feature_names, importances):
                importance_dict[str(name)] = float(importance)

        return importance_dict

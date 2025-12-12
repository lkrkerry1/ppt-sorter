"""
极简分类器 - 专为低配置设备设计
"""

import numpy as np
import re
import zipfile
import xml.etree.ElementTree as ET
import joblib
from typing import Tuple, Optional, Dict, List, Any
import gc

from config import DeployConfig, KeywordConfig, SUBJECTS, LABEL_MODEL_PATH
from utils.text_processor import TextProcessor


class UltraLightPPTClassifier:
    """
    极简PPT分类器
    内存占用<50MB，推理时间<0.5秒
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        初始化分类器

        Args:
            model_path: 模型文件路径，如果为None则使用内置关键词匹配
        """
        self.model_path: Optional[str] = model_path
        self.model: Optional[Any] = None
        self.keyword_matcher: Dict[str, List[str]] = KeywordConfig.BASE_KEYWORDS
        self.text_processor: TextProcessor = TextProcessor()
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.cache_size: int = DeployConfig.CACHE_SIZE
        self.label_encoder = joblib.load(LABEL_MODEL_PATH)

        # 加载模型（如果有）
        if model_path:
            self._load_model(model_path)

        # 内存监控
        self.memory_warning_issued: bool = False

    def _load_model(self, model_path: str) -> None:
        """加载模型"""
        try:
            print(f"加载模型: {model_path}")

            # 检查文件大小
            import os

            size_mb = os.path.getsize(model_path) / 1024 / 1024
            print(f"模型大小: {size_mb:.2f} MB")

            # 加载模型数据
            if model_path.endswith(".gz"):
                import gzip
                import pickle

                with gzip.open(model_path, "rb") as f:
                    model_data = pickle.load(f)
            else:
                model_data = joblib.load(model_path)

            # 提取模型组件
            try:
                self.model = model_data.get("model")
            except AttributeError:
                self.model = model_data
            self.feature_indices = model_data.get("feature_indices", None)
            self.keyword_matcher = model_data.get(
                "keyword_matcher", self.keyword_matcher
            )

            print("模型加载成功")

        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            print("将使用纯关键词匹配模式")
            self.model = None

    def predict(self, ppt_file: str) -> Tuple[str, float]:
        """
        预测PPT学科

        Args:
            ppt_file: PPT文件路径

        Returns:
            (学科, 置信度)
        """
        # 检查缓存
        file_hash = ""
        if DeployConfig.ENABLE_CACHE:
            file_hash = self._file_hash(ppt_file)
            if file_hash in self.cache:
                return self.cache[file_hash]

        # 内存检查
        if not self._check_memory():
            # 内存不足，使用纯关键词匹配
            result = self._keyword_only_predict(ppt_file)
        else:
            # 正常预测流程
            result = self._full_predict(ppt_file)

        # 更新缓存
        if DeployConfig.ENABLE_CACHE:
            self._update_cache(file_hash, result)

        return result

    def _full_predict(self, ppt_file: str) -> Tuple[str, float]:
        """完整预测流程"""
        try:
            # 1. 快速文本提取
            text = self._extract_text_fast(ppt_file)

            if not text:
                return "未知", 0.0

            # 2. 关键词快速匹配（第一层）
            keyword_result, keyword_conf = self._keyword_fast_match(text)
            if keyword_conf > 0.8:
                return keyword_result, keyword_conf

            # 3. 模型预测（第二层）
            if self.model is not None:
                model_result, model_conf = self._model_predict(text)

                # 综合置信度
                final_conf = max(model_conf, keyword_conf * 0.3)
                final_subject = (
                    model_result if model_conf >= keyword_conf else keyword_result
                )
                # print(f"testing {ppt_file}: {final_subject}")
                return final_subject, min(final_conf, 0.99)
            else:
                # 没有模型，返回关键词匹配结果
                return keyword_result, keyword_conf

        except Exception as e:
            print(f"预测过程中出错: {str(e)}")
            return "未知", 0.0

    def _keyword_only_predict(self, ppt_file: str) -> Tuple[str, float]:
        """仅使用关键词预测（内存不足时使用）"""
        try:
            text = self._extract_text_fast(ppt_file, max_slides=10)  # 限制页数
            if not text:
                return "未知", 0.0

            return self._keyword_fast_match(text)
        except:
            return "未知", 0.0

    def _extract_text_fast(self, file_path: str, max_slides: int = 30) -> str:
        """快速文本提取"""
        # 方法1: 直接解析pptx XML（最快）
        text = self._extract_via_xml(file_path, max_slides)
        if text:
            return text

        # 方法2: 使用python-pptx（备用）
        return self._extract_via_pptx(file_path, max_slides)

    def _extract_via_xml(self, file_path: str, max_slides: int) -> str:
        """通过直接解析XML提取文本"""
        if not file_path.endswith(".pptx"):
            return ""

        try:
            texts = []
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                # 获取所有幻灯片
                slide_files = []
                for name in zip_ref.namelist():
                    if name.startswith("ppt/slides/slide") and name.endswith(".xml"):
                        slide_number = int(name.split("slide")[1].split(".")[0])
                        slide_files.append((slide_number, name))

                # 按顺序排序并限制数量
                slide_files.sort()
                slide_files = slide_files[:max_slides]

                for _, slide_file in slide_files:
                    try:
                        content = zip_ref.read(slide_file).decode(
                            "utf-8", errors="ignore"
                        )
                        # 简单提取文本
                        slide_texts = re.findall(r"<a:t[^>]*>([^<]+)</a:t>", content)
                        if slide_texts:
                            texts.extend(slide_texts)
                    except:
                        continue

            return " ".join(texts)

        except Exception as e:
            # print(f"XML提取失败: {str(e)}")
            return ""

    def _extract_via_pptx(self, file_path: str, max_slides: int) -> str:
        """通过python-pptx提取文本"""
        try:
            from utils.ppt_parser import PPTParser

            parser = PPTParser()
            return parser.extract_text(file_path, max_slides=max_slides)
        except Exception as e:
            # print(f"pptx提取失败: {str(e)}")
            return ""

    def _keyword_fast_match(self, text: str) -> Tuple[str, float]:
        """关键词快速匹配"""
        text_lower = text.lower()
        scores = {}

        for subject, keywords in self.keyword_matcher.items():
            # 统计关键词命中数
            count = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    count += 1

            if count > 0:
                # 计算得分（考虑关键词权重）
                score = count / len(keywords)

                # 根据文本长度调整
                if len(text) > 1000:
                    score *= 1.2  # 长文本给予更高权重

                scores[subject] = min(score, 1.0)

        if not scores:
            return "未知", 0.0

        # 找到最高分
        best_subject = max(scores.items(), key=lambda x: x[1])
        return best_subject[0], best_subject[1]

    def _model_predict(self, text: str) -> Tuple[str, float]:
        """模型预测"""
        if self.model is None:
            return "未知", 0.0

        try:
            # 特征提取（简化的文本特征）
            features = self._extract_light_features(text)
            if features is None:
                return "未知", 0.0

            # 预测
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)[0]
                pred_idx = np.argmax(proba)
                confidence = float(proba[pred_idx])

                # 获取学科名称
                subject = self.label_encoder.inverse_transform([pred_idx])[0]
            else:
                prediction = self.model.predict(features)[0]
                subject = str(prediction)
                confidence = 0.7

            return subject, confidence

        except Exception as e:
            print(f"模型预测失败: {str(e)}")
            return "未知", 0.0

    def _extract_light_features(self, text: str) -> Optional[np.ndarray]:
        """提取轻量特征"""
        # 简化特征：只使用词频
        if not hasattr(self, "feature_indices"):
            return None

        # 文本处理
        processed = self.text_processor.process(text)
        words = processed.split()

        # 简单词频统计
        feature_vector = np.zeros(
            len(self.feature_indices) if self.feature_indices is not None else 100
        )

        # 这里应该根据实际特征提取逻辑实现
        # 简化：统计关键词出现
        for i, subject in enumerate(SUBJECTS):
            if subject in self.keyword_matcher:
                keywords = self.keyword_matcher[subject]
                count = sum(1 for kw in keywords if kw in text.lower())
                if i < len(feature_vector):
                    feature_vector[i] = count

        return feature_vector.reshape(1, -1)

    def _check_memory(self) -> bool:
        """检查内存使用情况"""
        try:
            import psutil

            memory = psutil.virtual_memory()

            if memory.percent > 90:
                if not self.memory_warning_issued:
                    print("⚠️  内存使用超过90%，切换到简化模式")
                    self.memory_warning_issued = True
                return False

            return True
        except:
            return True  # 如果无法检查内存，假设内存充足

    def _file_hash(self, filepath: str) -> str:
        """计算文件哈希（简化版）"""
        import os

        stat = os.stat(filepath)
        return f"{filepath}_{stat.st_size}_{stat.st_mtime}"

    def _update_cache(self, key: str, value: Tuple[str, float]):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            # 移除最旧的项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[key] = value

    def batch_predict(
        self, ppt_files: List[str], show_progress: bool = True
    ) -> List[Tuple[str, float]]:
        """批量预测"""
        results = []

        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(ppt_files, desc="处理PPT文件")
            except ImportError:
                iterator = ppt_files
        else:
            iterator = ppt_files

        for ppt_file in iterator:
            result = self.predict(ppt_file)
            results.append(result)

            # 定期清理内存
            if len(results) % 10 == 0:
                gc.collect()

        return results

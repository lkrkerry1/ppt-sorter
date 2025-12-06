"""
文本处理器 - 文本清洗、分词、标准化
"""

import re
import jieba
from typing import List, Set, Optional
from config import KeywordConfig


class TextProcessor:
    """文本处理器"""

    def __init__(self, custom_stopwords: Optional[Set[str]] = None) -> None:
        """
        初始化文本处理器

        Args:
            custom_stopwords: 自定义停用词
        """
        # 初始化jieba
        jieba.initialize()

        # 加载停用词
        self.stopwords: Set[str] = set(KeywordConfig.STOPWORDS)
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

        # 添加自定义词典（学科术语）
        self._add_custom_dict()

    def _add_custom_dict(self) -> None:
        """添加自定义词典"""
        # 添加学科关键词到词典
        for subject, keywords in KeywordConfig.BASE_KEYWORDS.items():
            for keyword in keywords:
                if len(keyword) >= 2 and not keyword.isascii():
                    # 只添加中文关键词
                    jieba.add_word(keyword, freq=1000)

    def process(self, text: str, keep_english: bool = True) -> str:
        """
        处理文本

        Args:
            text: 原始文本
            keep_english: 是否保留英文单词

        Returns:
            处理后的文本
        """
        if not text:
            return ""

        # 1. 基本清洗
        text = self._basic_clean(text)

        # 2. 中文分词
        chinese_part = self._extract_chinese(text)
        if chinese_part:
            chinese_words = jieba.lcut(chinese_part)
            chinese_words = [
                w for w in chinese_words if w not in self.stopwords and len(w) >= 2
            ]
            chinese_processed = " ".join(chinese_words)
        else:
            chinese_processed = ""

        # 3. 英文处理
        english_processed = ""
        if keep_english:
            english_part = self._extract_english(text)
            if english_part:
                english_words = self._process_english(english_part)
                english_processed = " ".join(english_words)

        # 4. 合并结果
        result = f"{chinese_processed} {english_processed}".strip()

        return result if result else text

    def _basic_clean(self, text: str) -> str:
        """基本文本清洗"""
        # 转换为小写（保留中文字符）
        text = text.lower()

        # 移除多余空白
        text = re.sub(r"\s+", " ", text)

        # 移除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r"[^\w\s\u4e00-\u9fff.,!?;:\-()]", " ", text)

        # 移除连续标点
        text = re.sub(r"[.,!?;:\-()]{2,}", " ", text)

        return text.strip()

    def _extract_chinese(self, text: str) -> str:
        """提取中文字符"""
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        return " ".join(chinese_chars)

    def _extract_english(self, text: str) -> str:
        """提取英文字符"""
        # 提取英文单词（至少2个字母）
        english_words = re.findall(r"[a-zA-Z]{2,}", text)
        return " ".join(english_words)

    def _process_english(self, english_text: str) -> List[str]:
        """处理英文文本"""
        words = english_text.split()

        # 过滤停用词和短词
        english_stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
        }

        filtered_words = [
            word
            for word in words
            if (
                word.lower() not in english_stopwords
                and len(word) >= 3
                and not word.isdigit()
            )
        ]

        return filtered_words

    def extract_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """
        提取关键词

        Args:
            text: 文本
            top_n: 返回的关键词数量

        Returns:
            关键词列表
        """
        # 简单实现：基于词频
        words = self.process(text).split()

        # 统计词频
        from collections import Counter

        word_counts = Counter(words)

        # 排除停用词
        filtered_counts = {
            word: count
            for word, count in word_counts.items()
            if word not in self.stopwords and len(word) >= 2
        }

        # 按词频排序
        sorted_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)

        return [word for word, count in sorted_words[:top_n]]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度（简化的Jaccard相似度）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度 (0-1)
        """
        words1 = set(self.process(text1).split())
        words2 = set(self.process(text2).split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

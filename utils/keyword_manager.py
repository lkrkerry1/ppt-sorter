"""
关键词管理器 - 管理学科关键词
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter, defaultdict
from config import KEYWORDS_FILE, SUBJECTS, KeywordConfig
import numpy as np


class KeywordManager:
    """关键词管理器"""

    def __init__(self):
        """初始化关键词管理器"""
        self.keywords = self.load_keywords()

    def load_keywords(self) -> Dict[str, List[str]]:
        """加载关键词"""
        if os.path.exists(KEYWORDS_FILE):
            try:
                with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                print(f"无法加载关键词文件: {KEYWORDS_FILE}")

        # 如果文件不存在，使用基础关键词
        return KeywordConfig.BASE_KEYWORDS.copy()

    def save_keywords(self):
        """保存关键词"""
        # 确保目录存在
        os.makedirs(os.path.dirname(KEYWORDS_FILE), exist_ok=True)

        with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.keywords, f, ensure_ascii=False, indent=2)

        print(f"关键词已保存到: {KEYWORDS_FILE}")

    def extract_from_texts(
        self, texts_by_subject: Dict[str, List[str]], top_n: int = 30
    ) -> Dict[str, List[str]]:
        """
        从文本中提取关键词

        Args:
            texts_by_subject: 按学科分类的文本列表
            top_n: 每科提取的关键词数量

        Returns:
            按学科分类的关键词
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        all_keywords: Dict[str, List[str]] = {}

        for subject, texts in texts_by_subject.items():
            if not texts:
                all_keywords[subject] = []
                continue

            # 合并所有文本
            combined_text = " ".join(texts)

            # 使用TF-IDF提取关键词
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=list(KeywordConfig.STOPWORDS),
                token_pattern=r"(?u)\b\w+\b",
            )

            # 拟合和转换
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()

            # 获取TF-IDF分数 - 修复spmatrix的toarray问题
            # 使用np.asarray()替代toarray()
            scores = np.asarray(tfidf_matrix.todense()).flatten()

            # 选择top_n关键词
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]

            all_keywords[subject] = keywords

        return all_keywords

    def merge_keywords(
        self, new_keywords: Dict[str, List[str]], keep_existing: bool = True
    ) -> Dict[str, List[str]]:
        """
        合并新旧关键词

        Args:
            new_keywords: 新提取的关键词
            keep_existing: 是否保留现有关键词

        Returns:
            合并后的关键词
        """
        merged: Dict[str, List[str]] = {}

        for subject in SUBJECTS:
            existing = set(self.keywords.get(subject, []))
            new = set(new_keywords.get(subject, []))

            if keep_existing:
                merged_set = existing.union(new)
            else:
                merged_set = new

            # 按长度和特定性排序
            sorted_keywords = sorted(
                merged_set,
                key=lambda x: (
                    len(x),
                    x in KeywordConfig.BASE_KEYWORDS.get(subject, []),
                ),
                reverse=True,
            )

            merged[subject] = list(sorted_keywords)[:50]  # 限制数量

        return merged

    def validate_keywords(self, test_texts: Dict[str, List[str]]) -> Dict[str, float]:
        """
        验证关键词效果

        Args:
            test_texts: 测试文本

        Returns:
            每科的关键词命中率
        """
        hit_rates: Dict[str, float] = {}

        for subject, texts in test_texts.items():
            if not texts:
                hit_rates[subject] = 0.0
                continue

            keywords = set(self.keywords.get(subject, []))
            total_hits = 0
            total_words = 0

            for text in texts:
                text_lower = text.lower()
                hits = sum(1 for kw in keywords if kw in text_lower)
                total_hits += hits
                total_words += len(text_lower.split())

            if total_words > 0:
                hit_rate = total_hits / total_words
            else:
                hit_rate = 0.0

            hit_rates[subject] = float(hit_rate)

        return hit_rates

    def get_keyword_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取关键词统计信息"""
        stats: Dict[str, Dict[str, Any]] = {}

        for subject, keywords in self.keywords.items():
            # 检查是否为str类型（避免类型错误）
            if not isinstance(keywords, list):
                continue

            # 计算中文关键词数量
            chinese_count = sum(
                1 for kw in keywords if re.search(r"[\u4e00-\u9fff]", str(kw))
            )
            # 计算英文关键词数量（纯英文，不包含中文）
            english_count = sum(
                1
                for kw in keywords
                if re.search(r"[a-zA-Z]", str(kw))
                and not re.search(r"[\u4e00-\u9fff]", str(kw))
            )

            stats[subject] = {
                "count": len(keywords),
                "avg_length": sum(len(str(kw)) for kw in keywords)
                / max(len(keywords), 1),
                "chinese_count": chinese_count,
                "english_count": english_count,
                "examples": keywords[:5],  # 前5个示例
            }

        return stats

    def find_ambiguous_keywords(self) -> List[Tuple[str, List[str]]]:
        """查找模糊关键词（出现在多个学科中）"""
        keyword_to_subjects: Dict[str, List[str]] = {}

        for subject, keywords in self.keywords.items():
            for keyword in keywords:
                if keyword not in keyword_to_subjects:
                    keyword_to_subjects[keyword] = []
                keyword_to_subjects[keyword].append(subject)

        ambiguous: List[Tuple[str, List[str]]] = []
        for keyword, subjects in keyword_to_subjects.items():
            if len(subjects) > 1:
                ambiguous.append((keyword, subjects))

        # 按模糊程度排序
        ambiguous.sort(key=lambda x: len(x[1]), reverse=True)

        return ambiguous

    def export_for_ai(self) -> Dict[str, List[str]]:
        """导出为AI友好的格式"""
        # 为每个学科添加描述
        enhanced: Dict[str, List[str]] = {}

        subject_descriptions = {
            "语文": "语言文字、文学、修辞、写作",
            "数学": "数学公式、几何、代数、计算",
            "英语": "英语语法、词汇、阅读、写作",
            "物理": "物理定律、力学、电磁学、光学",
            "化学": "化学元素、反应、实验、分子",
            "生物": "生物学、细胞、遗传、生态系统",
        }

        for subject, keywords in self.keywords.items():
            enhanced_keywords = keywords.copy()

            # 添加学科描述
            if subject in subject_descriptions:
                enhanced_keywords.insert(0, subject_descriptions[subject])

            enhanced[subject] = enhanced_keywords

        return enhanced

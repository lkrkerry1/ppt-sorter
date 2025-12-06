"""
数据加载器 - 加载和组织PPT数据集
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import random
from pathlib import Path

from config import SUBJECTS, RAW_DATA_DIR
from utils.ppt_parser import PPTParser


class PPTDataset:
    """PPT数据集类"""

    def __init__(
        self, data_dir: Optional[str] = None, subjects: Optional[List[str]] = None
    ):
        """
        初始化数据集

        Args:
            data_dir: PPT数据目录，结构为: data_dir/语文/*.pptx, data_dir/数学/*.pptx 等
            subjects: 学科列表，默认为配置中的SUBJECTS
        """
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self.subjects = subjects or SUBJECTS
        self.ppt_parser = PPTParser()

        # 缓存
        self._file_cache: Dict[str, List[str]] = {}
        self._label_cache: Dict[str, List[str]] = {}
        self._text_cache: Dict[str, str] = {}

    def load_data(
        self, limit_per_subject: Optional[int] = None, shuffle: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        加载数据集

        Args:
            limit_per_subject: 每个学科限制的文件数（None表示不限制）
            shuffle: 是否打乱数据

        Returns:
            (文件路径列表, 标签列表)
        """
        ppt_files: List[str] = []
        labels: List[str] = []

        print(f"正在从 {self.data_dir} 加载数据...")

        for subject in self.subjects:
            subject_dir = self.data_dir / subject

            if not subject_dir.exists():
                print(f"  警告: 学科目录不存在 - {subject_dir}")
                continue

            # 查找PPT文件
            subject_files: List[Path] = []
            for ext in [".pptx", ".ppt", ".pptm"]:
                subject_files.extend(list(subject_dir.glob(f"*{ext}")))

            # 限制数量
            if limit_per_subject and len(subject_files) > limit_per_subject:
                subject_files = random.sample(subject_files, limit_per_subject)

            if not subject_files:
                print(f"  警告: 学科 {subject} 没有找到PPT文件")
                continue

            # 添加到数据列表
            ppt_files.extend([str(f) for f in subject_files])
            labels.extend([subject] * len(subject_files))

            print(f"  {subject}: {len(subject_files)} 个文件")

        # 转换为数组用于打乱
        ppt_files_array = np.array(ppt_files)
        labels_array = np.array(labels)

        # 打乱数据
        if shuffle:
            indices = np.arange(len(ppt_files_array))
            np.random.shuffle(indices)
            ppt_files_array = ppt_files_array[indices]
            labels_array = labels_array[indices]

        # 转换回列表
        ppt_files_list = ppt_files_array.tolist()
        labels_list = labels_array.tolist()

        print(f"总共加载: {len(ppt_files_list)} 个PPT文件")

        # 缓存
        self._file_cache["all"] = ppt_files_list
        self._label_cache["all"] = labels_list

        return ppt_files_list, labels_list

    def load_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        加载并分割训练集和测试集

        Args:
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        if "all" not in self._file_cache:
            self.load_data()

        from sklearn.model_selection import train_test_split

        X = self._file_cache["all"]
        y = self._label_cache["all"]

        # 分层分割，确保每个学科在训练集和测试集中都有代表性
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"训练集: {len(X_train)} 个样本")
        print(f"测试集: {len(X_test)} 个样本")

        return X_train, X_test, y_train, y_test

    def load_subject_data(self, subject: str, limit: Optional[int] = None) -> List[str]:
        """
        加载指定学科的数据

        Args:
            subject: 学科名称
            limit: 限制文件数

        Returns:
            文件路径列表
        """
        if subject not in self.subjects:
            raise ValueError(f"无效的学科: {subject}")

        subject_dir = self.data_dir / subject
        if not subject_dir.exists():
            return []

        # 查找PPT文件
        ppt_files: List[Path] = []
        for ext in [".pptx", ".ppt", ".pptm"]:
            ppt_files.extend(list(subject_dir.glob(f"*{ext}")))

        # 限制数量
        if limit and len(ppt_files) > limit:
            ppt_files = random.sample(ppt_files, limit)

        return [str(f) for f in ppt_files]

    def extract_texts(self, ppt_files: List[str], cache: bool = True) -> List[str]:
        """
        从PPT文件提取文本

        Args:
            ppt_files: PPT文件路径列表
            cache: 是否缓存提取的文本

        Returns:
            文本列表
        """
        texts: List[str] = []

        for i, file_path in enumerate(ppt_files):
            # 检查缓存
            if cache and file_path in self._text_cache:
                texts.append(self._text_cache[file_path])
                continue

            # 提取文本
            try:
                text = self.ppt_parser.extract_text(file_path, max_slides=30)
                texts.append(text)

                # 缓存
                if cache:
                    self._text_cache[file_path] = text
            except Exception as e:
                print(f"  警告: 无法提取 {file_path} 的文本: {str(e)}")
                texts.append("")

            # 进度显示
            if (i + 1) % 10 == 0:
                print(f"  已提取 {i + 1}/{len(ppt_files)} 个文件")

        return texts

    def get_class_distribution(self) -> Dict[str, int]:
        """获取类别分布"""
        if "all" not in self._label_cache:
            self.load_data(shuffle=False)

        from collections import Counter

        return dict(Counter(self._label_cache["all"]))

    def analyze_dataset(self) -> Dict[str, int]:
        """分析数据集"""
        print("数据集分析:")
        print("-" * 40)

        # 获取类别分布
        distribution = self.get_class_distribution()

        # 统计每个学科的文件数
        total_files = sum(distribution.values())
        print(f"总文件数: {total_files}")
        print()

        print("各学科分布:")
        for subject in self.subjects:
            count = distribution.get(subject, 0)
            percentage = count / total_files * 100 if total_files > 0 else 0
            print(f"  {subject}: {count:4d} 个文件 ({percentage:5.1f}%)")

        # 检查数据平衡性
        if total_files > 0:
            max_count = max(distribution.values())
            min_count = min(distribution.values())
            imbalance_ratio = max_count / max(min_count, 1)

            print()
            print(f"最大/最小类别比例: {imbalance_ratio:.2f}")

            if imbalance_ratio > 3:
                print("⚠️  数据集不平衡，建议收集更多少数类别的数据")
            else:
                print("✓ 数据集相对平衡")

        return distribution

    def create_mini_dataset(
        self, samples_per_class: int = 10, save_dir: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        创建小型数据集（用于快速测试）

        Args:
            samples_per_class: 每个学科的样本数
            save_dir: 保存目录（如果为None则不保存）

        Returns:
            (文件路径列表, 标签列表)
        """
        ppt_files: List[str] = []
        labels: List[str] = []

        for subject in self.subjects:
            subject_files = self.load_subject_data(subject, limit=samples_per_class)
            ppt_files.extend(subject_files)
            labels.extend([subject] * len(subject_files))

        # 保存到文件（如果需要）
        if save_dir:
            import pandas as pd

            save_path = Path(save_dir) / "mini_dataset.csv"
            df = pd.DataFrame({"file_path": ppt_files, "label": labels})
            df.to_csv(save_path, index=False, encoding="utf-8")
            print(f"小型数据集已保存到: {save_path}")

        return ppt_files, labels

    def save_dataset_info(
        self, output_path: str = "dataset_info.json"
    ) -> Dict[str, Any]:
        """保存数据集信息"""
        import json
        from collections import defaultdict

        info: Dict[str, Any] = {
            "subjects": self.subjects,
            "data_dir": str(self.data_dir),
            "total_files": 0,
            "distribution": {},
            "file_examples": defaultdict(list),
        }

        # 收集信息
        for subject in self.subjects:
            subject_files = self.load_subject_data(subject)
            info["distribution"][subject] = len(subject_files)
            info["total_files"] += len(subject_files)

            # 每个学科保存几个示例文件名
            if subject_files:
                examples = [os.path.basename(f) for f in subject_files[:5]]
                info["file_examples"][subject] = examples

        # 保存
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"数据集信息已保存到: {output_path}")
        return info

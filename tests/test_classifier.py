"""
分类器测试
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
from typing import Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from deploy.ultra_light_classifier import UltraLightPPTClassifier
from utils.text_processor import TextProcessor
from utils.keyword_manager import KeywordManager


class TestTextProcessor(unittest.TestCase):
    """文本处理器测试"""

    def setUp(self) -> None:
        self.processor: TextProcessor = TextProcessor()

    def test_basic_clean(self) -> None:
        """测试基本清洗"""
        text: str = "这是一个测试文本！包含标点，和123数字。"
        cleaned: str = self.processor._basic_clean(text)
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("！", cleaned)  # 中文感叹号应该被移除

    def test_chinese_extraction(self) -> None:
        """测试中文提取"""
        text: str = "Hello 世界 123"
        chinese: str = self.processor._extract_chinese(text)
        self.assertEqual(chinese, "世界")

    def test_english_extraction(self) -> None:
        """测试英文提取"""
        text: str = "Hello 世界 testing 123"
        english: str = self.processor._extract_english(text)
        self.assertIn("Hello", english)
        self.assertIn("testing", english)
        self.assertNotIn("123", english)  # 数字不应被提取

    def test_process(self) -> None:
        """测试完整处理"""
        text: str = "这是一个测试文档，包含English words。"
        processed: str = self.processor.process(text)
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)


class TestKeywordManager(unittest.TestCase):
    """关键词管理器测试"""

    def setUp(self) -> None:
        self.manager: KeywordManager = KeywordManager()

    def test_load_keywords(self):
        """测试加载关键词"""
        keywords = self.manager.keywords
        self.assertIsInstance(keywords, dict)
        self.assertIn("语文", keywords)
        self.assertIn("数学", keywords)

    def test_keyword_stats(self):
        """测试关键词统计"""
        stats = self.manager.get_keyword_stats()
        self.assertIsInstance(stats, dict)
        for subject in ["语文", "数学", "英语", "物理", "化学", "生物"]:
            if subject in stats:
                self.assertIn("count", stats[subject])
                self.assertIn("examples", stats[subject])


class TestClassifier(unittest.TestCase):
    """分类器测试"""

    def setUp(self):
        # 使用纯关键词模式测试
        self.classifier = UltraLightPPTClassifier(model_path=None)

    def test_keyword_match(self):
        """测试关键词匹配"""
        # 测试数学内容
        math_text = "函数与方程是数学中的重要概念"
        subject, confidence = self.classifier._keyword_fast_match(math_text)
        self.assertIsInstance(subject, str)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_empty_text(self):
        """测试空文本"""
        subject, confidence = self.classifier._keyword_fast_match("")
        self.assertEqual(subject, "未知")
        self.assertEqual(confidence, 0.0)

    def test_unknown_text(self):
        """测试无关文本"""
        text = "今天的天气很好，我们去公园玩吧"
        subject, confidence = self.classifier._keyword_fast_match(text)
        # 可能匹配到某个学科，但置信度应该很低
        if subject != "未知":
            self.assertLess(confidence, 0.3)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_end_to_end(self):
        """端到端测试"""
        # 创建一个模拟PPT文件
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("二次函数是数学中的重要内容\n")
            f.write("y = ax^2 + bx + c\n")
            temp_file = f.name

        try:
            # 使用纯关键词模式
            classifier = UltraLightPPTClassifier(model_path=None)

            # 模拟预测
            subject, confidence = classifier._keyword_fast_match("二次函数 数学")

            # 验证结果
            self.assertIsInstance(subject, str)
            self.assertIsInstance(confidence, float)

            # 数学内容应该匹配到数学学科
            if "数学" in subject or "函数" in "二次函数 数学":
                self.assertGreater(confidence, 0.0)

        finally:
            # 清理临时文件
            os.unlink(temp_file)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    # 添加测试类
    suite.addTest(loader.loadTestsFromTestCase(TestTextProcessor))
    suite.addTest(loader.loadTestsFromTestCase(TestKeywordManager))
    suite.addTest(loader.loadTestsFromTestCase(TestClassifier))
    suite.addTest(loader.loadTestsFromTestCase(TestIntegration))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

"""本地快速自检脚本

步骤：
- 用小语料训练 CountVectorizer + LogisticRegression
- 使用 `ModelCompressor` 压缩为 `deployment_model.joblib`
- 生成一个最小的伪 `test.pptx`（包含 <a:t> 标签文本）
- 使用 `UltraLightPPTClassifier` 加载并预测
"""

import os
import sys
import zipfile
import joblib
import numpy as np

# Ensure project root is on sys.path so top-level modules can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from src.model_compressor import ModelCompressor
from src.ultralight_classifier import UltraLightPPTClassifier


def make_fake_pptx(path, text="数学 科学"):
    # 最小 slide xml，含若干 <a:t> 文本
    slide_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:spTree xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <a:t>{text}</a:t>
</p:spTree>"""

    with zipfile.ZipFile(path, "w") as zf:
        # 写一个 slide 文件，UltraLightPPTClassifier 只查找 ppt/slides/slide*
        zf.writestr("ppt/slides/slide1.xml", slide_xml)


def main():
    print("自检：准备训练小模型...")

    texts = [
        "这是一份数学课件",
        "英语课件内容",
        "物理实验报告",
        "化学方程式示例",
        "生物细胞结构描述",
        "语文古诗朗诵材料",
    ]
    # 标签 0..5 对应学科
    y = np.array([1, 2, 3, 4, 5, 0])

    vect = CountVectorizer()
    X = vect.fit_transform(texts)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    comp = ModelCompressor()
    # 使用 index->label 列表，确保压缩后能被正确解析
    comp.label_mapping = ["语文", "数学", "英语", "物理", "化学", "生物"]

    out_dir = os.path.join("tests", "tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "deployment_model.joblib")
    comp.compress_for_deployment(clf, vect, X, y, out_path, top_k=50)

    print("生成伪 PPTX 并运行预测...")
    pptx_path = os.path.join(out_dir, "test.pptx")
    # 使用与训练样本中一致的短语以触发关键词匹配（中文示例）
    make_fake_pptx(pptx_path, text="这是一份数学课件")

    clf2 = UltraLightPPTClassifier(out_path)
    result = clf2.predict_fast(pptx_path)
    print("预测结果:", result)

    # 清理临时文件（可选）
    # os.remove(out_path)
    # os.remove(pptx_path)


if __name__ == "__main__":
    main()

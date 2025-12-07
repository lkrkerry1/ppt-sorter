#!/usr/bin/env python3
"""
生成验证用PPT - 不同于训练集的PPT
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from scripts.generate_training_ppts import PPTGenerator


def generate_validation_set():
    """生成验证集"""
    print("生成验证用PPT...")
    print("=" * 50)

    DATA_DIR = Path("data/validation")
    generator = PPTGenerator()

    validation_topics = {
        "语文": [
            "修辞手法详解",
            "古诗词默写训练",
            "议论文写作",
            "文言文翻译技巧",
            "文学流派分析",
        ],
        "数学": ["不等式求解", "三角函数应用", "空间几何", "数学归纳法", "复数运算"],
        "英语": ["时态语态", "非谓语动词", "阅读理解策略", "写作模板", "口语练习"],
        "物理": ["电磁学原理", "热力学定律", "波动光学", "近代物理", "实验数据处理"],
        "化学": ["化学计算", "物质结构", "化学反应速率", "化学平衡移动", "环境化学"],
        "生物": ["人体生理学", "植物学基础", "微生物应用", "生态保护", "生物进化证据"],
    }

    for subject, topics in validation_topics.items():
        subject_dir = DATA_DIR / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        print(f"生成 {subject} 验证PPT...")
        for i, topic in enumerate(topics, 1):
            # 生成内容大纲
            content = [
                f"{topic}概述",
                "重点内容讲解",
                "典型例题分析",
                "易错点提示",
                "练习题目",
            ]

            # 生成PPT
            filename = f"验证_{subject}_{i:02d}_{topic}.pptx"
            output_path = subject_dir / filename

            try:
                generator.generate_subject_ppt(
                    subject=subject,
                    topic=topic,
                    content=content,
                    output_path=str(output_path),
                )
            except Exception as e:
                print(f"  生成失败 {filename}: {str(e)}")

    print("\n验证集生成完成!")
    print(f"保存在: {DATA_DIR}")


if __name__ == "__main__":
    generate_validation_set()

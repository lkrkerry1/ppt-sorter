#!/usr/bin/env python3
"""
主训练脚本 - 在强训练机上运行
使用: python train_main.py [--data_dir DATA_DIR] [--epochs EPOCHS]
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import TrainConfig, DeployConfig, SUBJECTS, MODELS_DIR
from train.feature_extractor import AdvancedFeatureExtractor
from train.model_builder import ModelBuilder
from train.knowledge_distiller import KnowledgeDistiller
from train.model_compressor import ModelCompressor
from utils.data_loader import PPTDataset
from utils.evaluation import ModelEvaluator


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练PPT学科分类模型")
    parser.add_argument(
        "--data_dir", type=str, default="data/raw", help="原始PPT数据目录"
    )
    parser.add_argument("--output_dir", type=str, default="models", help="模型输出目录")
    parser.add_argument(
        "--epochs", type=int, default=10, help="训练轮数（用于深度学习模型）"
    )
    parser.add_argument("--use_gpu", action="store_true", help="使用GPU加速训练")
    parser.add_argument("--compress", action="store_true", help="训练后压缩模型")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集比例")
    return parser.parse_args()


def main() -> int:
    """主训练流程"""
    args = parse_args()

    print("=" * 60)
    print("PPT学科分类器 - 训练阶段")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用GPU: {args.use_gpu}")
    print(f"测试集比例: {args.test_size}")
    print()

    # 第1步：加载数据
    print("1. 加载数据...")
    dataset = PPTDataset(args.data_dir, subjects=SUBJECTS)
    X_train, X_test, y_train, y_test = dataset.load_train_test_split(
        test_size=args.test_size, random_state=42
    )

    # 转换为numpy数组
    import numpy as np

    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    print(f"   训练集: {len(X_train)} 个样本")
    print(f"   测试集: {len(X_test)} 个样本")
    print(f"   学科分布: {dict(zip(*np.unique(y_train_np, return_counts=True)))}")

    # 第2步：特征提取
    print("\n2. 特征提取...")
    extractor = AdvancedFeatureExtractor(use_gpu=args.use_gpu)

    print("   提取文本特征...")
    X_train_features = extractor.extract_features(X_train, mode="train")
    X_test_features = extractor.extract_features(X_test, mode="test")

    print(f"   特征维度: {X_train_features.shape[1]}")

    # 第3步：训练教师模型
    print("\n3. 训练教师模型...")
    builder = ModelBuilder()
    teacher_model = builder.train_teacher_model(
        X_train_features,
        y_train_np,  # 使用numpy数组
        n_estimators=TrainConfig.TEACHER_N_ESTIMATORS,
        use_gpu=args.use_gpu,
    )

    # 评估教师模型
    evaluator = ModelEvaluator(teacher_model, SUBJECTS)
    teacher_acc = evaluator.evaluate(X_test_features, y_test_np)  # 使用numpy数组
    print(f"   教师模型准确率: {teacher_acc:.4f}")

    # 保存教师模型
    builder.save_model(teacher_model, str(MODELS_DIR / "teacher_model.joblib"))

    # 第4步：知识蒸馏
    print("\n4. 知识蒸馏...")
    distiller = KnowledgeDistiller(teacher_model)
    student_model = distiller.distill(
        X_train_features,
        y_train_np,
        temperature=TrainConfig.DISTILLATION_TEMPERATURE,  # 使用numpy数组
    )

    # 评估学生模型
    student_evaluator = ModelEvaluator(student_model, SUBJECTS)
    student_acc = student_evaluator.evaluate(
        X_test_features, y_test_np
    )  # 使用numpy数组
    print(f"   学生模型准确率: {student_acc:.4f}")
    print(f"   准确率下降: {(teacher_acc - student_acc):.4f}")

    # 保存学生模型
    builder.save_model(student_model, str(MODELS_DIR / "student_model.joblib"))

    # 第5步：模型压缩（可选）
    compressed_acc: Optional[float] = None
    model_size: Optional[float] = None
    if args.compress:
        print("\n5. 模型压缩...")
        compressor = ModelCompressor()
        compressed_model = compressor.compress(
            student_model,
            X_train_features[:100],  # 使用少量样本进行压缩优化
            output_path=str(MODELS_DIR / "deployment_model.joblib"),
        )

        # 测试压缩模型
        compressed_evaluator = ModelEvaluator(compressed_model, SUBJECTS)
        compressed_acc = compressed_evaluator.evaluate(
            X_test_features, y_test_np
        )  # 使用numpy数组
        print(f"   压缩模型准确率: {compressed_acc:.4f}")

        # 检查模型大小
        model_size = (
            os.path.getsize(MODELS_DIR / "deployment_model.joblib") / 1024 / 1024
        )
        print(f"   模型大小: {model_size:.2f} MB")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 输出模型性能对比
    print("\n模型性能对比:")
    print(f"{'模型':<15} {'准确率':<10} {'大小(MB)':<10}")
    print("-" * 40)
    print(f"{'教师模型':<15} {teacher_acc:.4f}      -")
    print(f"{'学生模型':<15} {student_acc:.4f}      -")
    if args.compress:
        print(f"{'部署模型':<15} {compressed_acc:.4f}      {model_size:.2f}")

    return 0


if __name__ == "__main__":
    try:
        import numpy as np

        sys.exit(main())
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

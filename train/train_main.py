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

from config import (
    TrainConfig,
    SUBJECTS,
    TEACHER_MODEL_PATH,
    STUDENT_MODEL_PATH,
    DEPLOYMENT_MODEL_PATH,
    LABEL_MODEL_PATH,
)
from train.feature_extractor import AdvancedFeatureExtractor
from train.model_builder import ModelBuilder
from train.knowledge_distiller import KnowledgeDistiller
from train.model_compressor import ModelCompressor
from utils.data_loader import PPTDataset
from utils.evaluation import ModelEvaluator
from sklearn.preprocessing import LabelEncoder  # 添加导入

# 设置代理（如果需要）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


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

    # 创建标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(SUBJECTS)  # 使用SUBJECTS顺序编码

    # 转换为数值标签
    y_train_np = label_encoder.transform(y_train)
    y_test_np = label_encoder.transform(y_test)

    print(f"   训练集: {len(X_train)} 个样本")
    print(f"   测试集: {len(X_test)} 个样本")

    # 显示标签映射关系
    print(
        f"   标签映射关系: {dict(zip(label_encoder.classes_, 
                               label_encoder.transform(label_encoder.classes_)))}"  # type:ignore
    )

    import numpy as np

    unique_labels, counts = np.unique(y_train_np, return_counts=True)
    label_names = label_encoder.inverse_transform(unique_labels)
    distribution = dict(zip(label_names, counts))
    print(f"   学科分布: {distribution}")

    # 保存标签编码器
    import joblib

    joblib.dump(label_encoder, str(LABEL_MODEL_PATH))
    print(f"   标签编码器已保存到: {LABEL_MODEL_PATH}")

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
        y_train_np,  # type:ignore # 使用数值标签 
        n_estimators=TrainConfig.TEACHER_N_ESTIMATORS,
        use_gpu=args.use_gpu,
    )

    # 评估教师模型 - 使用数值标签
    evaluator = ModelEvaluator(
        teacher_model, SUBJECTS
    )  # 在 train_main.py 的评估部分前添加：

    print(f"\n特征维度检查:")
    print(f"  训练特征维度: {X_train_features.shape}")
    print(f"  测试特征维度: {X_test_features.shape}")

    # 如果维度不匹配，尝试截断或填充
    if X_train_features.shape[1] != X_test_features.shape[1]:
        print(f"  警告: 特征维度不匹配!")

        # 取最小维度
        min_dim = min(X_train_features.shape[1], X_test_features.shape[1])

        # 如果测试特征维度太大，截断
        if X_test_features.shape[1] > min_dim:
            X_test_features = X_test_features[:, :min_dim]
            print(f"  截断测试特征到 {min_dim} 维")

        # 如果测试特征维度太小，填充零
        elif X_test_features.shape[1] < min_dim:
            padding = np.zeros(
                (X_test_features.shape[0], min_dim - X_test_features.shape[1])
            )
            X_test_features = np.hstack([X_test_features, padding])
            print(f"  填充测试特征到 {min_dim} 维")

    # 现在评估模型
    teacher_acc = evaluator.evaluate(X_test_features, y_test_np) # type:ignore
    print(f"   教师模型准确率: {teacher_acc:.4f}")

    # 保存教师模型
    builder.save_model(teacher_model, str(TEACHER_MODEL_PATH))

    # 第4步：知识蒸馏
    print("\n4. 知识蒸馏...")
    distiller = KnowledgeDistiller(teacher_model)
    student_model = distiller.distill(
        X_train_features,
        y_train_np,  # 使用数值标签 # type:ignore
        temperature=TrainConfig.DISTILLATION_TEMPERATURE,
    )

    # 评估学生模型
    student_evaluator = ModelEvaluator(student_model, SUBJECTS)
    student_acc = student_evaluator.evaluate(X_test_features, y_test_np) # type:ignore
    print(f"   学生模型准确率: {student_acc:.4f}")
    print(f"   准确率下降: {(teacher_acc - student_acc):.4f}")

    # 保存学生模型
    builder.save_model(student_model, str(STUDENT_MODEL_PATH))

    # 第5步：模型压缩（可选）
    compressed_acc: Optional[float] = None
    model_size: Optional[float] = None
    if args.compress:
        print("\n5. 模型压缩...")
        compressor = ModelCompressor()
        # 1. 调用compress，得到的是包含模型的字典
        compressed_data = (
            compressor.compress(  # 变量名改为compressed_data，表明它是字典
                student_model,
                X_train_features[
                    :100
                ],  # 注意：这里使用了原特征，但压缩器内部可能选择了部分特征
                output_path=str(DEPLOYMENT_MODEL_PATH),
            )
        )

        # 2. 从字典中取出真正的模型对象
        # 根据你的 model_compressor.py，模型在键 "model" 下
        compressed_model = compressed_data[
            "model"
        ]  # 这是具有 .predict() 方法的模型对象

        # 3. 使用取出的模型进行评估
        compressed_evaluator = ModelEvaluator(compressed_model, SUBJECTS)
        compressed_acc = compressed_evaluator.evaluate(X_test_features, y_test_np) # type:ignore
        model_size = os.path.getsize(DEPLOYMENT_MODEL_PATH) / 1024 / 1024
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
        sys.exit(main())
    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
部署主脚本 - 在低配置设备上运行
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from config import DEPLOYMENT_MODEL_PATH
from deploy.ultra_light_classifier import UltraLightPPTClassifier
from deploy.memory_guard import MemoryGuard
from deploy.check_environment import check_system_compatibility


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="PPT学科分类器 - 部署版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单个文件分类
  python deploy_main.py path/to/your.pptx
  
  # 批量分类
  python deploy_main.py --input folder/with/ppts --output results.csv
  
  # 使用纯关键词模式（最低内存）
  python deploy_main.py --keyword-only your.pptx
        """,
    )

    parser.add_argument("input", nargs="?", help="PPT文件路径或文件夹路径")
    parser.add_argument("--input", dest="input_dir", help="包含PPT文件的文件夹")
    parser.add_argument("--output", help="输出CSV文件路径（批量处理时）")
    parser.add_argument(
        "--model", default=str(DEPLOYMENT_MODEL_PATH), help="模型文件路径"
    )
    parser.add_argument(
        "--keyword-only", action="store_true", help="仅使用关键词匹配（最低内存占用）"
    )
    parser.add_argument("--no-cache", action="store_true", help="禁用缓存")
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="详细输出级别"
    )

    return parser


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """解析命令行参数"""
    return parser.parse_args()


def find_ppt_files(directory: str) -> List[str]:
    """查找目录中的所有PPT文件"""
    ppt_extensions = [".pptx", ".ppt"]
    ppt_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in ppt_extensions):
                ppt_files.append(os.path.join(root, file))

    return ppt_files


def print_result(
    filename: str, subject: str, confidence: float, processing_time: float = 0.0
):
    """美化输出结果"""
    # 置信度颜色
    if confidence > 0.8:
        conf_color = "\033[92m"  # 绿色
    elif confidence > 0.6:
        conf_color = "\033[93m"  # 黄色
    else:
        conf_color = "\033[91m"  # 红色

    reset_color = "\033[0m"

    # 学科颜色
    subject_colors = {
        "语文": "\033[94m",  # 蓝色
        "数学": "\033[95m",  # 紫色
        "英语": "\033[96m",  # 青色
        "物理": "\033[92m",  # 绿色
        "化学": "\033[93m",  # 黄色
        "生物": "\033[91m",  # 红色
    }

    subject_color = subject_colors.get(subject, "\033[97m")

    # 格式化输出
    base_name = os.path.basename(filename)
    subject_str = f"{subject_color}{subject:8s}{reset_color}"
    conf_str = f"{conf_color}{confidence:.2%}{reset_color}"

    if processing_time:
        time_str = f" ({processing_time:.2f}s)"
    else:
        time_str = ""

    print(f"  {base_name:40s} -> {subject_str} {conf_str}{time_str}")


def single_file_mode(args, classifier):
    """单文件模式"""
    input_path = args.input or args.input_dir

    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 - {input_path}")
        return 1

    print(f"\n处理文件: {input_path}")

    # 预测
    start_time = time.time()
    subject, confidence = classifier.predict(input_path)
    processing_time = time.time() - start_time

    # 输出结果
    print("\n" + "=" * 60)
    print_result(input_path, subject, confidence, processing_time)
    print("=" * 60)

    return 0


def batch_mode(args, classifier):
    """批量处理模式"""
    input_dir = args.input or args.input_dir

    if not os.path.isdir(input_dir):
        print(f"错误: 不是有效的目录 - {input_dir}")
        return 1

    # 查找PPT文件
    print(f"正在扫描目录: {input_dir}")
    ppt_files = find_ppt_files(input_dir)

    if not ppt_files:
        print("未找到PPT文件")
        return 1

    print(f"找到 {len(ppt_files)} 个PPT文件")

    # 批量预测
    results = []
    print("\n开始分类...")

    for ppt_file in ppt_files:
        start_time = time.time()
        subject, confidence = classifier.predict(ppt_file)
        processing_time = time.time() - start_time

        results.append(
            {
                "file": ppt_file,
                "subject": subject,
                "confidence": confidence,
                "time": processing_time,
            }
        )

        if args.verbose > 0:
            print_result(ppt_file, subject, confidence, processing_time)

    # 保存结果
    if args.output:
        import csv

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "subject", "confidence", "time"]
            )
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"\n结果已保存到: {args.output}")

    # 统计信息
    print("\n" + "=" * 60)
    print("批量处理完成!")
    print("=" * 60)

    # 按学科统计
    from collections import Counter

    subject_counts = Counter(r["subject"] for r in results)

    print("\n学科分布:")
    for subject in ["语文", "数学", "英语", "物理", "化学", "生物", "未知"]:
        count = subject_counts.get(subject, 0)
        if count > 0:
            percentage = count / len(results) * 100
            print(f"  {subject:8s}: {count:3d} 个 ({percentage:.1f}%)")

    # 平均置信度
    valid_results = [r for r in results if r["subject"] != "未知"]
    if valid_results:
        avg_confidence = sum(r["confidence"] for r in valid_results) / len(
            valid_results
        )
        print(f"\n平均置信度: {avg_confidence:.2%}")

    # 平均处理时间
    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"平均处理时间: {avg_time:.2f} 秒/文件")


def main():
    """主函数"""
    parser = create_parser()
    args = parse_args(parser)

    print("=" * 60)
    print("PPT学科分类器 - 部署版")
    print("=" * 60)

    # 检查环境
    print("检查运行环境...")
    env_check = check_system_compatibility()
    if env_check == "unsupported":
        print("⚠️  环境不满足最低要求，可能无法正常运行")
    elif env_check == "light":
        print("✓  环境检测: 轻量模式")
    else:
        print("✓  环境检测: 正常模式")

    # 检查输入
    if not args.input and not args.input_dir:
        print("\n错误: 需要指定输入文件或目录")
        parser.print_help()
        return 1

    # 初始化内存守护
    memory_guard = MemoryGuard()
    if not memory_guard.check():
        print("内存不足，程序退出")
        return 1

    # 加载分类器
    print("加载分类器...")
    try:
        if args.keyword_only:
            classifier = UltraLightPPTClassifier(model_path=None)
            print("模式: 纯关键词匹配 (最低内存)")
        else:
            model_path = args.model if os.path.exists(args.model) else None
            classifier = UltraLightPPTClassifier(model_path=model_path)
            if classifier.model is None:
                print("模式: 纯关键词匹配 (模型加载失败)")
            else:
                print("模式: 完整模型")
    except Exception as e:
        print(f"分类器初始化失败: {str(e)}")
        return 1

    # 判断模式
    input_path = args.input or args.input_dir
    is_directory = os.path.isdir(input_path)

    if is_directory or args.output:
        # 批量模式
        return batch_mode(args, classifier)
    else:
        # 单文件模式
        return single_file_mode(args, classifier)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
        import traceback

        # 重新解析参数以获取 verbose
        parser = create_parser()
        args = parse_args(parser)

        if args.verbose > 1:
            traceback.print_exc()
        sys.exit(1)

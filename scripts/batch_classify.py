#!/usr/bin/env python3
"""
批量分类脚本
"""

import argparse
import sys
import os
import csv
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from deploy.ultra_light_classifier import UltraLightPPTClassifier
from config import DEPLOYMENT_MODEL_PATH
from deploy.memory_guard import MemoryGuard


def find_ppt_files(directory: str, recursive: bool = True) -> List[str]:
    """查找PPT文件"""
    extensions = [".pptx", ".ppt"]
    ppt_files = []

    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    ppt_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            filepath = os.path.join(directory, file)
            if os.path.isfile(filepath) and any(
                file.lower().endswith(ext) for ext in extensions
            ):
                ppt_files.append(filepath)

    return sorted(ppt_files)


def classify_single_file(
    classifier: UltraLightPPTClassifier, filepath: str
) -> Tuple[str, str, float, float]:
    """分类单个文件"""
    start_time = time.time()
    subject, confidence = classifier.predict(filepath)
    processing_time = time.time() - start_time

    return filepath, subject, confidence, processing_time


def batch_classify(
    input_dir: str,
    output_csv: str,
    model_path: str | None = None,
    max_workers: int = 2,
    recursive: bool = True,
    show_progress: bool = True,
):
    """批量分类"""

    print(f"扫描目录: {input_dir}")
    ppt_files = find_ppt_files(input_dir, recursive)

    if not ppt_files:
        print("未找到PPT文件")
        return

    print(f"找到 {len(ppt_files)} 个PPT文件")

    # 初始化分类器
    print("初始化分类器...")
    classifier = UltraLightPPTClassifier(model_path)

    # 初始化内存守护
    memory_guard = MemoryGuard()

    results = []

    # 进度显示
    if show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(total=len(ppt_files), desc="处理进度")
        except ImportError:
            progress_bar = None
            print("处理文件:")
    else:
        progress_bar = None

    # 使用线程池（限制线程数以控制内存）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_file = {
            executor.submit(classify_single_file, classifier, filepath): filepath
            for filepath in ppt_files
        }

        # 收集结果
        for future in as_completed(future_to_file):
            filepath = future_to_file[future]

            try:
                result = future.result()
                results.append(result)

                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        {"subject": result[1], "conf": f"{result[2]:.2f}"}
                    )
                else:
                    print(
                        f"  {os.path.basename(filepath):30s} -> {result[1]:8s} ({result[2]:.2f})"
                    )

            except Exception as e:
                print(f"处理失败 {os.path.basename(filepath)}: {str(e)}")
                results.append((filepath, "错误", 0.0, 0.0))

            # 定期检查内存
            if len(results) % 10 == 0:
                if not memory_guard.check():
                    print("内存不足，停止处理")
                    break

    if progress_bar:
        progress_bar.close()

    # 保存结果
    if output_csv:
        save_results(results, output_csv)

    # 打印统计信息
    print_statistics(results)


def save_results(results: List[Tuple], output_csv: str):
    """保存结果到CSV"""
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件路径", "学科", "置信度", "处理时间(秒)"])

        for row in results:
            writer.writerow(row)

    print(f"\n结果已保存到: {output_csv}")


def print_statistics(results: List[Tuple]):
    """打印统计信息"""
    if not results:
        return

    print("\n" + "=" * 60)
    print("批量处理统计")
    print("=" * 60)

    # 按学科统计
    from collections import Counter

    subject_counts = Counter(r[1] for r in results)

    print("\n学科分布:")
    subjects = ["语文", "数学", "英语", "物理", "化学", "生物", "未知", "错误"]
    for subject in subjects:
        count = subject_counts.get(subject, 0)
        if count > 0:
            percentage = count / len(results) * 100
            print(f"  {subject:8s}: {count:4d} 个 ({percentage:6.1f}%)")

    # 置信度统计
    valid_results = [r for r in results if r[2] > 0 and r[1] not in ["未知", "错误"]]
    if valid_results:
        confidences = [r[2] for r in valid_results]
        avg_confidence = sum(confidences) / len(confidences)

        print(f"\n置信度分析:")
        print(f"  平均置信度: {avg_confidence:.2%}")
        print(f"  最高置信度: {max(confidences):.2%}")
        print(f"  最低置信度: {min(confidences):.2%}")

        # 置信度分布
        conf_bins = [0, 0.5, 0.7, 0.8, 0.9, 1.0]
        conf_labels = ["<50%", "50-70%", "70-80%", "80-90%", "90-100%"]

        for i in range(len(conf_bins) - 1):
            low, high = conf_bins[i], conf_bins[i + 1]
            count = sum(1 for c in confidences if low <= c < high)
            if count > 0:
                percentage = count / len(confidences) * 100
                print(f"  {conf_labels[i]:8s}: {count:4d} 个 ({percentage:6.1f}%)")

    # 时间统计
    processing_times = [r[3] for r in results if r[3] > 0]
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        total_time = sum(processing_times)

        print(f"\n性能统计:")
        print(f"  总处理时间: {total_time:.1f} 秒")
        print(f"  平均处理时间: {avg_time:.2f} 秒/文件")
        print(f"  最快处理: {min(processing_times):.2f} 秒")
        print(f"  最慢处理: {max(processing_times):.2f} 秒")
        if len(results) > 1:
            print(f"  总文件数: {len(results)} 个")
            print(f"  吞吐量: {len(results)/total_time:.1f} 文件/秒")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="PPT学科批量分类工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python batch_classify.py --input ./ppt_folder --output results.csv
  
  # 递归扫描子目录
  python batch_classify.py --input ./ppt_folder --output results.csv --recursive
  
  # 使用指定模型
  python batch_classify.py --input ./ppt_folder --model models/custom.joblib
  
  # 限制线程数（控制内存）
  python batch_classify.py --input ./ppt_folder --threads 1
  
  # 仅关键词匹配（最低内存）
  python batch_classify.py --input ./ppt_folder --keyword-only
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="输入目录（包含PPT文件）")
    parser.add_argument("--output", "-o", help="输出CSV文件路径")
    parser.add_argument(
        "--model", "-m", default=str(DEPLOYMENT_MODEL_PATH), help="模型文件路径"
    )
    parser.add_argument(
        "--threads", "-t", type=int, default=2, help="最大线程数（默认: 2）"
    )
    parser.add_argument("--recursive", "-r", action="store_true", help="递归扫描子目录")
    parser.add_argument(
        "--keyword-only", "-k", action="store_true", help="仅使用关键词匹配（最低内存）"
    )
    parser.add_argument("--no-progress", action="store_true", help="不显示进度条")

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"错误: 输入目录不存在 - {args.input}")
        return 1

    # 检查模型文件
    model_path = None if args.keyword_only else args.model
    if model_path and not os.path.exists(model_path):
        print(f"警告: 模型文件不存在 - {model_path}")
        print("将使用纯关键词匹配模式")
        model_path = None

    # 设置输出文件
    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"classification_results_{timestamp}.csv"

    print("=" * 60)
    print("PPT学科批量分类工具")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出文件: {args.output}")
    print(
        f"使用模型: {'关键词匹配' if model_path is None else os.path.basename(model_path)}"
    )
    print(f"线程数: {args.threads}")
    print(f"递归扫描: {'是' if args.recursive else '否'}")
    print()

    try:
        batch_classify(
            input_dir=args.input,
            output_csv=args.output,
            model_path=model_path,
            max_workers=args.threads,
            recursive=args.recursive,
            show_progress=not args.no_progress,
        )
        return 0
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 130
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

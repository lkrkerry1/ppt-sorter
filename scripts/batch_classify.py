"""批量分类脚本占位。
遍历目录，加载模型并输出分类结果。
"""


def batch_classify(input_dir: str, model_path: str):
    print(f"批量分类：{input_dir} -> 使用模型 {model_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python batch_classify.py <input_dir> <model_path>")
    else:
        batch_classify(sys.argv[1], sys.argv[2])

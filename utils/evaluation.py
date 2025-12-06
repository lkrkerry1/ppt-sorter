"""评估工具占位模块。
包含准确率、召回率等评估函数。
"""


def accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))

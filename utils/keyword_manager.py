"""关键词管理器占位模块。
管理学科关键词与匹配逻辑。
"""

import json


def load_keywords(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

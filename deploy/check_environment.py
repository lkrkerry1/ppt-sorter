"""
环境检查工具 - 检查部署环境是否满足要求
"""

import sys
import platform
import os
from typing import Dict, Tuple, List


def check_system_compatibility() -> str:
    """
    检查系统兼容性

    Returns:
        'normal': 正常模式
        'light': 轻量模式
        'unsupported': 不支持
    """
    print("=" * 50)
    print("系统兼容性检查")
    print("=" * 50)

    checks = []

    # 1. Python版本
    python_version = sys.version_info
    python_ok = python_version >= (3, 7)
    checks.append(
        (
            "Python版本",
            f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            python_ok,
        )
    )

    # 2. 操作系统
    system = platform.system()
    system_ok = system in ["Windows", "Linux", "Darwin"]  # Windows, Linux, Mac
    checks.append(("操作系统", system, system_ok))

    # 3. 架构
    arch = platform.machine()
    arch_ok = "64" in arch or "x86_64" in arch or "AMD64" in arch
    checks.append(("系统架构", arch, arch_ok))

    # 4. 内存
    memory_ok = True
    memory_info = "未知"
    try:
        import psutil

        memory = psutil.virtual_memory()
        memory_mb = memory.total / 1024 / 1024
        memory_info = f"{memory_mb:.0f}MB"

        if memory_mb < 2000:  # 小于2GB
            memory_ok = False
        elif memory_mb < 4000:  # 小于4GB
            memory_ok = True  # 仍然可以运行，但性能受限
    except ImportError:
        memory_info = "无法检测 (需要psutil)"

    checks.append(("系统内存", memory_info, memory_ok))

    # 5. 磁盘空间
    disk_ok = True
    disk_info = "未知"
    try:
        import psutil

        disk = psutil.disk_usage(os.path.expanduser("~"))
        disk_gb = disk.free / 1024 / 1024 / 1024
        disk_info = f"{disk_gb:.1f}GB"

        if disk_gb < 0.5:  # 小于500MB
            disk_ok = False
    except (ImportError, OSError):
        disk_info = "无法检测"

    checks.append(("可用磁盘空间", disk_info, disk_ok))

    # 6. 检查必需库
    required_libs = ["numpy", "sklearn", "joblib", "jieba"]
    missing_libs = []

    for lib in required_libs:
        try:
            __import__(lib)
            checks.append((f"库: {lib}", "✓", True))
        except ImportError:
            missing_libs.append(lib)
            checks.append((f"库: {lib}", "✗", False))

    # 7. 可选库
    optional_libs = ["python-pptx", "psutil", "tqdm"]
    for lib in optional_libs:
        try:
            __import__(lib.replace("-", "_"))
            checks.append((f"可选库: {lib}", "✓", True))
        except ImportError:
            checks.append((f"可选库: {lib}", "✗", False))

    # 输出检查结果
    print("\n检查项目            状态          结果")
    print("-" * 45)

    for name, value, ok in checks:
        status = "✓" if ok else "✗"
        color = "\033[92m" if ok else "\033[91m"
        reset = "\033[0m"
        print(f"{name:20s} [{color}{status}{reset}] {value}")

    print("-" * 45)

    # 总结
    all_passed = all(ok for _, _, ok in checks if not _.startswith("可选库"))
    optional_passed = sum(1 for _, _, ok in checks if _.startswith("可选库") and ok)

    if not all_passed:
        print("\n❌ 系统不满足最低要求")
        print("   请解决以下问题:")
        for name, value, ok in checks:
            if not ok and not name.startswith("可选库"):
                print(f"   - {name}: {value}")
        return "unsupported"
    elif memory_info != "未知" and "MB" in memory_info:
        # 检查内存大小决定模式
        memory_mb = float(memory_info.replace("MB", ""))
        if memory_mb < 4000:
            print("\n⚠️  内存较少，建议使用轻量模式")
            print("   运行命令添加 --keyword-only 参数")
            return "light"

    print("\n✅ 系统满足运行要求")
    return "normal"


def check_file_permissions() -> bool:
    """检查文件权限"""
    try:
        # 测试当前目录写权限
        test_file = "test_permission.tmp"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except:
        return False


def get_system_info() -> Dict[str, str]:
    """获取系统详细信息"""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "executable": sys.executable,
    }

    try:
        import psutil

        info["cpu_count"] = str(psutil.cpu_count())
        info["memory_total_gb"] = (
            f"{psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}"
        )
    except:
        pass

    return info


if __name__ == "__main__":
    mode = check_system_compatibility()

    print(f"\n运行模式建议: {mode}")

    if mode == "unsupported":
        sys.exit(1)
    else:
        sys.exit(0)

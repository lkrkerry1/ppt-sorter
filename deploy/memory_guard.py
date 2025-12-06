"""
内存守护 - 防止低内存设备崩溃
"""

import gc
import threading
import time
from typing import Optional, Callable


class MemoryGuard:
    """内存守护者"""

    def __init__(self, max_memory_mb: int = 1500, check_interval: float = 5.0) -> None:
        """
        初始化内存守护

        Args:
            max_memory_mb: 最大内存限制（MB）
            check_interval: 检查间隔（秒）
        """
        self.max_memory: int = max_memory_mb * 1024 * 1024  # 转换为字节
        self.check_interval: float = check_interval
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitor: bool = False
        self.memory_warnings: int = 0

    def check(self) -> bool:
        """检查内存是否充足"""
        try:
            import psutil

            # 检查系统内存
            system_memory = psutil.virtual_memory()
            if system_memory.percent > 95:
                print("⚠️  系统内存使用超过95%")
                return False

            # 检查进程内存
            process = psutil.Process()
            process_memory = process.memory_info().rss

            if process_memory > self.max_memory:
                print(f"⚠️  进程内存使用过高: {process_memory/1024/1024:.1f}MB")
                self.cleanup()
                return False

            return True

        except ImportError:
            # 无法导入psutil，跳过内存检查
            return True
        except Exception as e:
            print(f"内存检查出错: {str(e)}")
            return True

    def start_monitoring(self):
        """启动内存监控线程"""
        if self.monitor_thread is None:
            self.stop_monitor = False
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()
            print("内存监控已启动")

    def stop_monitoring(self):
        """停止内存监控"""
        self.stop_monitor = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
            print("内存监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while not self.stop_monitor:
            if not self.check():
                self.memory_warnings += 1

                if self.memory_warnings >= 3:
                    print("多次内存警告，建议重启程序")

            time.sleep(self.check_interval)

    def cleanup(self):
        """清理内存"""
        print("正在清理内存...")

        # 强制垃圾回收
        collected = gc.collect()
        print(f"垃圾回收: 清理了 {collected} 个对象")

        # 清理代际垃圾
        for gen in range(2, -1, -1):
            collected = gc.collect(gen)
            if collected > 0:
                print(f"代际清理({gen}): {collected} 个对象")

        # 清理导入模块的缓存（如果有）
        try:
            import sys

            # 清理模块缓存（保留核心模块）
            modules_to_keep = {"sys", "os", "builtins", "__main__", "gc", "threading"}
            modules_to_remove = []

            for module_name in list(sys.modules.keys()):
                if (
                    module_name not in modules_to_keep
                    and not module_name.startswith("__")
                    and not module_name.startswith("_")
                    and "ppt_subject_classifier" not in module_name
                ):
                    modules_to_remove.append(module_name)

            for module_name in modules_to_remove[:20]:  # 限制清理数量
                try:
                    del sys.modules[module_name]
                except:
                    pass

            print(f"清理了 {len(modules_to_remove[:20])} 个模块缓存")

        except Exception as e:
            print(f"清理模块缓存时出错: {str(e)}")

        # 清理其他可能的缓存
        self._clear_caches()

    def _clear_caches(self):
        """清理各种缓存"""
        try:
            # 清理numpy缓存
            import numpy as np

            np._globals._NoValue = None
        except:
            pass

        try:
            # 清理sklearn缓存
            import sklearn

            if hasattr(sklearn, "_memory"):
                sklearn._memory.clear()  # type: ignore
        except:
            pass

    def get_memory_info(self) -> dict:
        """获取内存信息"""
        try:
            import psutil

            process = psutil.Process()
            system = psutil.virtual_memory()

            return {
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "system_total_mb": system.total / 1024 / 1024,
                "system_available_mb": system.available / 1024 / 1024,
                "system_percent": system.percent,
                "memory_warnings": self.memory_warnings,
            }
        except:
            return {}

    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
        self.cleanup()

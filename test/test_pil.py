import sys

print(f"Python: {sys.version}")

try:
    from PIL import Image

    print("✓ PIL (Pillow) 导入成功")
    # 尝试创建一个简单图像，验证核心功能
    im = Image.new("RGB", (50, 50), color="red")
    im.verify()  # 快速验证图像数据完整性
    print("✓ Pillow 核心功能正常")
except Exception as e:
    print(f"✗ Pillow 导入失败: {e}")

try:
    import torchvision

    print(f"✓ torchvision 导入成功，版本: {torchvision.__version__}")
except Exception as e:
    print(f"✗ torchvision 导入失败: {e}")

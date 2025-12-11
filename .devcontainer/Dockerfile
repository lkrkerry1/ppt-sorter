# 使用指定版本的Python官方镜像
FROM python:3.10.19-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 复制requirements_train.txt文件
COPY requirements_train.txt .

# 安装Python依赖
# 先安装一些可能编译需要的基础包
RUN pip install --upgrade pip setuptools wheel

# 安装PyTorch相关依赖（CPU版本）
# 如果需要CUDA版本，可以将下一行替换为合适的CUDA版本
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 安装其他依赖（排除已安装的torch相关包）
# 创建一个临时文件，排除已安装的torch包
RUN grep -v "torch" requirements_train.txt > requirements_remaining.txt && \
    pip install -r requirements_remaining.txt && \
    rm requirements_remaining.txt


# 设置默认命令
CMD ["/bin/bash"]
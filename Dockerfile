# 使用 slim 镜像减小体积
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（根据你的需求调整）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app/ ./app/
COPY ml/ ./ml/

# 创建必要的目录
RUN mkdir -p logs models

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口（根据你的应用调整）
EXPOSE 8000

# 启动命令（根据你的应用调整）
CMD ["python", "app/main.py"]
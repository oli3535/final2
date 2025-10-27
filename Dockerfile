FROM python:3.11-slim

WORKDIR /app

# 先复制 requirements.txt
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 然后复制其他文件
COPY app/ ./app/
COPY ml/ ./ml/

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["python", "app/main.py"]

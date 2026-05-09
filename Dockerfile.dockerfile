FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（sqlite3 已内置）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# 创建数据目录（可被 volume 覆盖）
RUN mkdir -p /app/data

CMD ["python", "main.py"]
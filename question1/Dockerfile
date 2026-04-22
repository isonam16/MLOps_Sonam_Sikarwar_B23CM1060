FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers \
    torch \
    sacrebleu \
    striprtf \
    sentencepiece

COPY . /app/

CMD ["python", "translate.py"]

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[anthropic,openai]"
# pre-download local models so first start is fast
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"
ENV GITCONTEXT_DATA_DIR=/data
VOLUME /data
EXPOSE 8000 8501
CMD ["uvicorn", "gitcontext.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]

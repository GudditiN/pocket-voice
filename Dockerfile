FROM python:3.12-slim
RUN pip install --no-cache-dir uv
WORKDIR /srv/app
COPY pyproject.toml ./
RUN uv sync --no-dev
COPY app ./app
ENV DATA_DIR=/data HF_HOME=/data/hf PORT=8501
VOLUME /data
EXPOSE 8501
# Pass the key at runtime: docker run -e GROQ_API_KEY=... -p 8501:8501 -v pv-data:/data pocket-voice
CMD ["uv", "run", "streamlit", "run", "app/Home.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]

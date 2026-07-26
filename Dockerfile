FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY src/config.py ./src/config.py
COPY src/data_collection/ ./src/data_collection/
COPY src/__init__.py ./src/__init__.py
COPY config.yaml .
COPY models/artifacts/ ./models/artifacts/
COPY models/metadata/ ./models/metadata/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Forma shell (sem colchetes): necessária para a variável ${PORT} ser expandida pelo shell do
# container. Render/Railway injetam $PORT em runtime; ${PORT:-8000} garante que
# `docker run -p 8000:8000` local (sem $PORT definida) continue funcionando igual.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}

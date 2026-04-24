# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ ./

RUN npm run build

# Stage 2: Install Python deps
FROM python:3.10-slim AS python-deps

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN python -c "from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer; nli_model='cross-encoder/nli-deberta-v3-small'; minilm_model='sentence-transformers/all-MiniLM-L6-v2'; AutoTokenizer.from_pretrained(nli_model); AutoModelForSequenceClassification.from_pretrained(nli_model); AutoTokenizer.from_pretrained(minilm_model); AutoModel.from_pretrained(minilm_model)"

# Stage 3: Prepare backend runtime artifacts
FROM python-deps AS backend-artifacts

ENV CONTAINER_HOME=/var/www
ENV PYTHONPATH=$CONTAINER_HOME

WORKDIR $CONTAINER_HOME

COPY backend/ $CONTAINER_HOME/backend/
COPY data/ $CONTAINER_HOME/data/

RUN python -m backend.text_processing.text_preprocess --ensure-postings

# Stage 4: Final runtime image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y git

ENV CONTAINER_HOME=/var/www
ENV PYTHONPATH=$CONTAINER_HOME
ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
ENV TRANSFORMERS_OFFLINE=1

WORKDIR $CONTAINER_HOME

COPY --from=python-deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=python-deps /opt/huggingface /opt/huggingface
COPY backend/ $CONTAINER_HOME/backend/
COPY data/ $CONTAINER_HOME/data/
COPY --from=backend-artifacts $CONTAINER_HOME/backend/ $CONTAINER_HOME/backend/
COPY --from=backend-artifacts $CONTAINER_HOME/data/ $CONTAINER_HOME/data/
COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist

CMD ["python", "-m", "gunicorn", "backend.app:app", "--bind", "0.0.0.0:5000", "--log-level", "debug"]

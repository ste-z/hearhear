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
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; model='cross-encoder/nli-deberta-v3-small'; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model)"

# Stage 3: Prepare backend runtime artifacts
FROM python-deps AS backend-artifacts

ENV CONTAINER_HOME=/var/www

WORKDIR $CONTAINER_HOME

COPY backend/ $CONTAINER_HOME/backend/

RUN python backend/text_preprocess.py --ensure-postings

# Stage 4: Final runtime image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y git

ENV CONTAINER_HOME=/var/www
ENV PYTHONPATH=$CONTAINER_HOME:$CONTAINER_HOME/src
ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface
ENV TRANSFORMERS_OFFLINE=1

WORKDIR $CONTAINER_HOME

COPY --from=python-deps /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=python-deps /opt/huggingface /opt/huggingface
COPY src/ $CONTAINER_HOME/src/
COPY --from=backend-artifacts $CONTAINER_HOME/backend/ $CONTAINER_HOME/backend/
COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist

CMD ["python", "-m", "gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:5000", "--log-level", "debug"]

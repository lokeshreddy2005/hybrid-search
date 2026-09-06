# NOTE: Docker was not available in the development environment this project
# was built in (see README "Limitations"), so this Dockerfile is written to
# standard best practice but has not been build-tested end-to-end. The
# verified run path is the local venv instructions in the README.
FROM python:3.12-slim

WORKDIR /app

# System deps needed by faiss / torch wheels at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# index/ includes index/store/ — build it first with
# `python scripts/build_index.py` before `docker build` (see README).
# The image build itself does not download the corpus/model.
COPY index/ index/
COPY retrieval/ retrieval/
COPY api/ api/
COPY eval/ eval/

ENV INDEX_STORE_DIR=index/store
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

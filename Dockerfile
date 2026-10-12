FROM python:3.11-slim
RUN pip install --no-cache-dir \
    numpy pillow scipy scikit-learn umap-learn matplotlib
WORKDIR /work

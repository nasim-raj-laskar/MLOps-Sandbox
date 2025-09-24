FROM python:3.10-slim as builder

# BLAS, git for DVC, curl
RUN apt-get update && apt-get install -y \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

RUN pip install mlflow[extras] prefect dvc[all] minio boto3 jupyterlab

# Runtime image
FROM python:3.10-slim

WORKDIR /workspace

# Copy Python packages from builder
COPY --from=builder /usr/local /usr/local

# Copy app and entrypoint as root
COPY app/ /workspace/app/
COPY docker/entrypoint.sh /workspace/docker/entrypoint.sh

# Make entrypoint executable
RUN chmod +x /workspace/docker/entrypoint.sh

# Create and switch to non-root user
RUN useradd -ms /bin/bash mlopsuser
USER mlopsuser

# Expose ports (MLflow UI, Jupyter)
EXPOSE 5000 8888

ENTRYPOINT ["bash", "/workspace/docker/entrypoint.sh"]

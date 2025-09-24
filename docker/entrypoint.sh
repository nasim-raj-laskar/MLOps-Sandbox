#!/bin/bash
echo "Starting All-in-One MLOps Sandbox..."

#MLflow tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 --port 5000 &

#Jupyter
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port=8888 &

tail -f /dev/null

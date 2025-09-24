import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import joblib
import mlflow
import dagshub
dagshub.init(repo_owner='nasim-raj-laskar', repo_name='MLOps-Sandbox', mlflow=True)

with mlflow.start_run():
    data = load_iris(as_frame=True)
    X, y = data.data, data.target
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("train_accuracy", model.score(X, y))

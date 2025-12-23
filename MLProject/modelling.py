import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

def train_model():
    data_dir = os.path.join(os.getcwd(), "ChurnData_Prepocessing")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"🚀 Retrained model with accuracy = {acc:.4f}")
    with mlflow.start_run(run_name=f"CI_Retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train_model()
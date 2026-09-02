import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
def load_data(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.drop(columns=["id", "Unnamed: 32"])
    return df


def split_features_target(df: pd.DataFrame):
    
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    
    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    
    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_malignant": precision_score(y_test, y_pred),
        "recall_malignant": recall_score(y_test, y_pred),
        "f1_malignant": f1_score(y_test, y_pred),
    }
    return metrics

if __name__ == "__main__":
    df = load_data("data/raw/data.csv")
    df = clean_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    mlflow.set_experiment("disease_prediction")

    # === Logistic Regression ===
    with mlflow.start_run(run_name="logistic_regression"):
        lr_model = train_model(X_train, y_train)
        metrics = evaluate_model(lr_model, X_test, y_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 5000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(lr_model, "model")

    # === Random Forest ===
    with mlflow.start_run(run_name="random_forest"):
        rf_model = train_random_forest(X_train, y_train)
        metrics = evaluate_model(rf_model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
    rf_model,
    "model",
    registered_model_name="disease_prediction_model"
)
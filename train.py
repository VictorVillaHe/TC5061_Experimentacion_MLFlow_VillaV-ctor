import pandas as pd
import mlflow
import mlflow.sklearn
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo California Housing")

    parser.add_argument("--n_estimators", type=int, default=100, help="Número de árboles del Random Forest")

    parser.add_argument("--max_depth", type=int, default=10, help="Profundidad máxima de los árboles")

    parser.add_argument("--min_samples_split", type=int, default=2, help="Mínimo de muestras necesarias para dividir un nodo")

    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")

    return parser.parse_args()

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def prepare_data(data, seed):
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.20, random_state=seed)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, n_estimators, max_depth, min_samples_split, seed):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed)

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return mae, mse, rmse, r2, y_pred


def create_artifact(y_test, y_pred):
    plt.figure(figsize=(7, 5))

    plt.scatter(y_test, y_pred, alpha=0.5)

    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title("Valores reales vs. predichos")

    artifact_path = "predicciones.png"

    plt.savefig(artifact_path)
    plt.close()

    return artifact_path

def log_to_mlflow(model, args, mae, mse, rmse, r2, artifact_path):

    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("seed", args.seed)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    mlflow.log_artifact(artifact_path)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

def main():
    args = parse_args()
    filepath = r"DB\california_housing_train.csv"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("California_Housing_RandomForest")

    data = load_data(filepath)

    X_train, X_test, y_train, y_test = prepare_data(data, args.seed)

    with mlflow.start_run():

        model = train_model(X_train, y_train, args.n_estimators, args.max_depth, args.min_samples_split, args.seed)

        mae, mse, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

        artifact_path = create_artifact(y_test, y_pred)

        log_to_mlflow(model, args, mae, mse, rmse, r2, artifact_path)

        print("n_estimators     :", args.n_estimators)
        print("max_depth        :", args.max_depth)
        print("min_samples_split:", args.min_samples_split)
        print("seed             :", args.seed)
        print()
        print("MAE :", mae)
        print("MSE :", mse)
        print("RMSE:", rmse)
        print("R²  :", r2)


if __name__ == "__main__":
    main()
import mlflow
import datetime
import os
import pickle
import random
from joblib import dump
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import sys
from sklearn.ensemble import GradientBoostingRegressor
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath('..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp

    # Use the timestamp for Github Actions
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # Load diabetes dataset instead of synthetic data
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save data for evaluation script
    if os.path.exists('data'):
        with open('data/X_test.pickle', 'wb') as data:
            pickle.dump(X_test, data)
        with open('data/y_test.pickle', 'wb') as data:
            pickle.dump(y_test, data)
    else:
        os.makedirs('data/')
        with open('data/X_test.pickle', 'wb') as data:
            pickle.dump(X_test, data)
        with open('data/y_test.pickle', 'wb') as data:
            pickle.dump(y_test, data)

    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Diabetes Dataset"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_name}"):

        params = {
            "dataset_name": dataset_name,
            "number of datapoint": X_train.shape[0],
            "number of dimensions": X_train.shape[1],
            "model_type": "GradientBoostingRegressor"
        }

        mlflow.log_params(params)

        # Use GradientBoostingRegressor instead of RandomForest
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate metrics for regression
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        mlflow.log_metrics({
            'Train MSE': train_mse,
            'Test MSE': test_mse,
            'Train R2': train_r2,
            'Test R2': test_r2,
            'RMSE': np.sqrt(test_mse)
        })

        print("Model Performance:")
        print(f"Train R2 Score: {train_r2:.4f}")
        print(f"Test R2 Score: {test_r2:.4f}")
        print(f"Test RMSE: {np.sqrt(test_mse):.4f}")

        if not os.path.exists('models/'):
            os.makedirs("models/")

        # After training the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}_gb_model.joblib'
        dump(model, model_filename)

        print(f"Model saved as: {model_filename}")

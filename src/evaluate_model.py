import pickle
import os
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import sys
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

    try:
        # Use a timestamp as the version
        model_version = f'model_{timestamp}_gb_model'
        model = joblib.load(f'{model_version}.joblib')
        print(f"Model loaded: {model_version}.joblib")
    except:
        raise ValueError('Failed to load the latest model')

    try:
        # Load test data saved from training
        with open('data/X_test.pickle', 'rb') as f:
            X_test = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
        print(f"Test data loaded: {X_test.shape[0]} samples")
    except:
        raise ValueError('Failed to load the test data')

    # Make predictions
    y_predict = model.predict(X_test)

    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)

    metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2_Score": float(r2),
        "MAE": float(mae)
    }

    print("\nModel Evaluation Results:")
    print("-" * 30)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Save metrics to a JSON file
    if not os.path.exists('metrics/'):
        os.makedirs("metrics/")

    metrics_filename = f'{timestamp}_metrics.json'
    with open(metrics_filename, 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"\nMetrics saved to: {metrics_filename}")

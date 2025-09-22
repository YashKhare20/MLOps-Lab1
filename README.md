# MLOps Lab1 - Diabetes Prediction with GitHub Actions

[![Model Training](https://github.com/YashKhare20/MLOps-Lab1/workflows/Model%20Retraining%20on%20Push%20to%20Main/badge.svg)](https://github.com/YashKhare20/MLOps-Lab1/actions)

Automated ML pipeline using GitHub Actions for training and versioning a diabetes prediction model.

## What This Does

- **Automatically trains** a Gradient Boosting model when you push to `main`
- **Uses real data**: Scikit-learn's diabetes dataset (353 patients, 10 features)
- **Saves versioned models** with timestamps
- **Tracks metrics**: MSE, RMSE, R², MAE
- **Commits results** back to the repository

## Dataset

- **Source**: Scikit-learn diabetes dataset
- **Samples**: 353 patients
- **Features**: 10 (Age, Sex, BMI, Blood Pressure, 6 blood serum measurements)
- **Target**: Disease progression (continuous value)

## Quick Setup

### 1. Fork/Clone Repository

```bash
git clone https://github.com/YashKhare20/MLOps-Lab1.git
cd MLOps-Lab1
```

### 2. Project Structure

```
MLOps-Lab1/
├── .github/
│   └── workflows/
│       ├── model_retraining_on_push.yml    # Triggers on push
│       └── model_calibration.yml           # Daily schedule
├── src/
│   ├── train_model.py                      # GradientBoostingRegressor
│   └── evaluate_model.py                   # Regression metrics
├── models/                                 # Saved models (auto-created)
├── metrics/                                # Evaluation results (auto-created)
├── data/                                   # Test data (auto-created)
├── requirements.txt                        # Dependencies
└── README.md                               # This file
```

### 3. Install Dependencies Locally

```bash
pip install -r requirements.txt
```

### 4. Test Locally (Optional)

```bash
# Create timestamp
timestamp=$(date '+%Y%m%d_%H%M%S')

# Train model
python src/train_model.py --timestamp "$timestamp"

# Evaluate model
python src/evaluate_model.py --timestamp "$timestamp"

# Check outputs
ls models/    # model_TIMESTAMP_gb_model.joblib
ls metrics/   # TIMESTAMP_metrics.json
```

## How It Works

### Push Code → Automatic Training

1. **Make any change**:
```bash
echo "# Update $(date)" >> README.md
git add .
git commit -m "Trigger pipeline"
git push origin main
```

2. **Watch it run**:
   - Go to **Actions** tab on GitHub
   - See workflow running (yellow dot)
   - Click to view live logs

3. **Check results** (after ~90 seconds):
```bash
git pull
ls -la models/   # New model file
cat metrics/*.json  # View metrics
```

## Model Details

### Algorithm
- **Model**: GradientBoostingRegressor
- **Parameters**:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 3
  - random_state: 42

### Metrics
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error

### File Naming
- Models: `model_YYYYMMDD_HHMMSS_gb_model.joblib`
- Metrics: `YYYYMMDD_HHMMSS_metrics.json`

## Files Explained

### `src/train_model.py`
- Loads diabetes dataset
- Trains GradientBoostingRegressor
- Saves model with timestamp
- Logs to MLflow

### `src/evaluate_model.py`
- Loads trained model
- Evaluates on test set
- Calculates regression metrics
- Saves metrics as JSON

### `.github/workflows/model_retraining_on_push.yml`
- Triggers on push to `main`
- Runs training pipeline
- Commits results back

### `.github/workflows/model_calibration.yml`
- Runs daily at midnight
- Same as push workflow
- Scheduled retraining

## Expected Performance

| Metric | Expected Value | Description |
|--------|---------------|-------------|
| **R² Score** | 0.40 - 0.50 | Model explains ~45% of variance |
| **RMSE** | 50 - 60 | Average prediction error |
| **Training Time** | < 5 seconds | Fast training |
| **Pipeline Time** | ~90 seconds | Full GitHub Actions run |


## License

MIT License - Feel free to use for learning!

## Author

Created for MLOps Lab1 - Automated Model Training with GitHub Actions
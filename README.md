# SWYNEX End-to-End ML Project

An end-to-end machine learning classification project using scikit-learn's built-in Breast Cancer Wisconsin dataset. It demonstrates data loading, preprocessing, model training, evaluation, model persistence, and prediction.

> Educational project: the dataset is a standard public benchmark. This model is not intended for medical diagnosis or clinical decision-making.

## Project workflow

1. Load the dataset directly from scikit-learn (no manual download required).
2. Split the data into training and test sets with stratification.
3. Build a preprocessing + Logistic Regression pipeline.
4. Train the model and evaluate it on held-out test data.
5. Save the trained pipeline and evaluation metrics.
6. Run predictions on a sample or on a JSON file of feature values.

## Repository structure

```text
SWYNEX-End-to-End-ML-Project/
├── src/
│   ├── train.py
│   └── predict.py
├── models/              # created after training
├── reports/             # created after training
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train and evaluate

From the project root:

```bash
python src/train.py
```

The script prints accuracy, precision, recall, F1-score, and a classification report. It writes:

- `models/breast_cancer_pipeline.joblib` — fitted preprocessing/model pipeline
- `reports/metrics.json` — key test metrics
- `reports/classification_report.txt` — per-class evaluation report
- `reports/confusion_matrix.csv` — confusion matrix

The test split is held out from training. Results may vary if the configuration or library versions change.

## Make a prediction

Train the model first, then run:

```bash
python src/predict.py
```

To predict from a JSON file, provide a JSON object containing all 30 dataset feature names, with numeric values:

```bash
python src/predict.py --input sample_input.json
```

Output labels:
- `0`: malignant
- `1`: benign

This is a learning demonstration only, not a medical tool.

## Methodology

- **Problem:** binary classification of observations in the Breast Cancer Wisconsin dataset.
- **Preprocessing:** `StandardScaler` is included in a scikit-learn `Pipeline`, ensuring scaling is fitted only on training data and then applied consistently at inference.
- **Model:** Logistic Regression with class weighting and a fixed random seed.
- **Evaluation:** stratified 80/20 train/test split; accuracy, precision, recall, F1-score, classification report, and confusion matrix.
- **Reproducibility:** fixed split seed and saved pipeline.

## Limitations and responsible use

- The dataset is small and intended for benchmarking/education.
- Test performance does not establish real-world clinical validity.
- Do not use predictions to diagnose, screen, or make treatment decisions.
- Before real deployment, validate on representative external data, assess subgroup performance, monitor drift, and obtain appropriate domain and regulatory review.

## Suggested next improvements

- Add cross-validation and hyperparameter tuning.
- Compare Logistic Regression with tree-based models.
- Add automated tests and a CI workflow.
- Wrap inference in a small FastAPI application.

## License

For educational use. Check the dataset and dependency licenses before redistribution in other contexts.

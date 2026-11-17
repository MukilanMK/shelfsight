"""
Module 4 (UNIT-III) Orchestration Pipeline
==========================================
Runs supervised regression and classification benchmarking pipelines
using consensus-selected features from Module 3.
"""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.module2_engineering.preprocessing import MLDataPreprocessor, SplitStrategy
from src.module4_supervised.classification_restock import ClassificationBenchmark, RestockClassifier
from src.module4_supervised.regression_demand import DemandForecaster, RegressionBenchmark


def run_module4_pipeline(
    features_path: Path = PROCESSED_DATA_DIR / "daily_product_features.csv",
    selected_features_path: Path = PROCESSED_DATA_DIR / "selected_features.json",
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
) -> Tuple[RegressionBenchmark, ClassificationBenchmark]:
    """
    Executes Module 4 supervised pipelines:
    1. Demand regression benchmark (Linear, Ridge, Lasso)
    2. Restock urgency classification benchmark (Logistic, Decision Tree, Naïve Bayes, SVM, KNN)
    """
    print("\n=====================================================================")
    print("      MODULE 4: SUPERVISED MACHINE LEARNING BENCHMARK PIPELINE       ")
    print("=====================================================================")

    df = pd.read_csv(features_path)
    preprocessor = MLDataPreprocessor(scaler_type="robust", test_ratio=0.20)
    dataset = preprocessor.prepare_dataset(df, split_strategy=SplitStrategy.TEMPORAL)

    cv = preprocessor.make_time_cv(dataset)

    # Load selected consensus features if available
    reg_features = dataset.feature_names
    clf_features = dataset.feature_names
    if Path(selected_features_path).exists():
        with open(selected_features_path, "r") as f:
            selected_info = json.load(f)
            if "regression_consensus_features" in selected_info:
                reg_features = [f for f in selected_info["regression_consensus_features"] if f in dataset.X_train.columns]
            elif "consensus_features" in selected_info:
                reg_features = [f for f in selected_info["consensus_features"] if f in dataset.X_train.columns]
            
            if "classification_consensus_features" in selected_info:
                clf_features = [f for f in selected_info["classification_consensus_features"] if f in dataset.X_train.columns]
            elif "consensus_features" in selected_info:
                clf_features = [f for f in selected_info["consensus_features"] if f in dataset.X_train.columns]

    print(f"[Module 4] Using {len(reg_features)} features for regression and {len(clf_features)} for classification.")
    X_train_reg, X_test_reg = dataset.X_train[reg_features], dataset.X_test[reg_features]
    X_train_clf, X_test_clf = dataset.X_train[clf_features], dataset.X_test[clf_features]

    # 1. Demand & Sales Forecasting Regression
    forecaster = DemandForecaster(random_state=42)
    reg_benchmark = forecaster.train_and_compare(
        X_train=X_train_reg,
        y_train=dataset.y_train_reg,
        X_test=X_test_reg,
        y_test=dataset.y_test_reg,
        n_splits=5,
        cv=cv,
        baselines=dataset.baselines_test,
    )
    forecaster.save_best_model(reg_benchmark, output_dir=models_dir)

    # 2. Restock Urgency Action Classification
    classifier = RestockClassifier(random_state=42)
    clf_benchmark = classifier.train_and_compare(
        X_train=X_train_clf,
        y_train=dataset.y_train_clf,
        X_test=X_test_clf,
        y_test=dataset.y_test_clf,
        n_splits=5,
        cv=cv,
    )
    classifier.save_best_model(clf_benchmark, output_dir=models_dir)

    # Persist benchmark tables
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reg_benchmark.comparison_table.to_csv(reports_dir / "regression_comparison.csv", index=False)
    clf_benchmark.comparison_table.to_csv(reports_dir / "classification_comparison.csv", index=False)

    print("\n=====================================================================")
    print("      MODULE 4 COMPLETED SUCCESSFULLY                               ")
    print("=====================================================================\n")

    return reg_benchmark, clf_benchmark


if __name__ == "__main__":
    run_module4_pipeline()

"""
Module 3 Feature Selection Pipeline
===================================
Applies Filter, Wrapper, and Embedded methods to the processed dataset.
Persists the consensus rankings and selected feature subsets.
"""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd
from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.module2_engineering.preprocessing import MLDataPreprocessor, SplitStrategy
from src.module3_feature_selection.feature_selection import FeatureSelectionSummary, FeatureSelector


def run_module3_pipeline(
    features_path: Path = PROCESSED_DATA_DIR / "daily_product_features.csv",
    reports_dir: Path = REPORTS_DIR,
    top_k: int = 15,
) -> Tuple[FeatureSelectionSummary, Path]:
    """
    Executes Feature Selection across Filter, Wrapper, and Embedded algorithms.
    """
    print("\n=====================================================================")
    print("      MODULE 3: MULTI-METHOD FEATURE SELECTION PIPELINE              ")
    print("=====================================================================")

    df = pd.read_csv(features_path)
    preprocessor = MLDataPreprocessor(scaler_type="robust", test_ratio=0.20)
    dataset = preprocessor.prepare_dataset(df, split_strategy=SplitStrategy.TEMPORAL)

    # Save preprocessor for dashboard and inference
    from src.config import MODELS_DIR
    preprocessor.save(MODELS_DIR / "preprocessor.pkl")

    selector = FeatureSelector(random_state=42)
    print("\n--- Feature Selection: Demand Velocity Regression Target ---")
    summary_reg = selector.select_features(
        X_train=dataset.X_train,
        y_train=dataset.y_train_reg,
        task="regression",
        top_k=top_k,
    )

    print("\n--- Feature Selection: Stockout Risk Classification Target ---")
    summary_clf = selector.select_features(
        X_train=dataset.X_train,
        y_train=dataset.y_train_clf,
        task="classification",
        top_k=top_k,
    )

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_csv = reports_dir / "feature_selection_rankings.csv"
    summary_reg.ranking_table.to_csv(report_csv, index=False)
    summary_clf.ranking_table.to_csv(reports_dir / "feature_selection_rankings_clf.csv", index=False)

    json_path = PROCESSED_DATA_DIR / "selected_features.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "consensus_features": summary_reg.selected_features,
                "regression_consensus_features": summary_reg.selected_features,
                "classification_consensus_features": summary_clf.selected_features,
                "filter_kbest_features": summary_reg.filter_top_k,
                "rf_importance_features": summary_reg.rf_top_k,
                "rfe_features": summary_reg.rfe_selected,
                "lasso_active_features": summary_reg.lasso_active_features,
            },
            f,
            indent=2,
        )

    print(f"\n[Module 3 - Output]")
    print(f"  - Ranking Table:    {report_csv}")
    print(f"  - Selected Features: {json_path}")
    print("=====================================================================")
    print("      MODULE 3 COMPLETED SUCCESSFULLY                               ")
    print("=====================================================================\n")

    return summary_reg, report_csv


if __name__ == "__main__":
    run_module3_pipeline()

"""
Module 4 (UNIT-III): Supervised Restock Urgency Action Classifier
=================================================================
Trains, tunes, and systematically compares 5 core classification paradigms:
1. Logistic Regression (Multinomial with L2 regularization)
2. Decision Tree Classifier (CART & ID3 entropy/gini principles)
3. Gaussian Naïve Bayes (Probabilistic generative classifier)
4. Support Vector Classifier - SVM (RBF kernel & hyperplane margin maximization)
5. K-Nearest Neighbors - K-NN (Instance-based lazy learning)

Target (leakage-safe): `target_stockout_in_lead_time` = 1 if the SKU's stock hits zero
within its supplier lead time. (The old target, the rule-based urgency tier, was a
deterministic function of the model's own inputs.) Metrics work for binary and
multi-class targets; positive-class precision/recall/F1 and PR-AUC are reported
because macro averages hide rare-class behaviour. Champion = best cross-validated F1
(positive class when binary, macro otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, REPORTS_DIR
from src.module2_engineering.validation_tuning import (
    ModelTuner,
    ValidationResults,
    classification_test_metrics,
)


@dataclass
class ClassificationModelMetrics:
    """Evaluation metrics for a single classifier."""
    model_name: str
    best_params: Dict[str, Any]
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    roc_auc_ovr: Optional[float]
    cv_f1_mean: float
    cv_f1_std: float
    overfitting_gap: float
    is_overfitting: bool
    model_obj: BaseEstimator
    precision_pos: Optional[float] = None
    recall_pos: Optional[float] = None
    f1_pos: Optional[float] = None
    pr_auc: Optional[float] = None


@dataclass
class ClassificationBenchmark:
    """Benchmark comparative summary across all 5 classification models."""
    comparison_table: pd.DataFrame
    best_model_name: str
    best_model: BaseEstimator
    models: Dict[str, ClassificationModelMetrics]


class RestockClassifier:
    """
    Trains, validates, tunes, and compares the 5 classification algorithms.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.tuner = ModelTuner(random_state=random_state)

    def train_and_compare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_splits: int = 5,
        cv: Any = None,
    ) -> ClassificationBenchmark:
        """
        Executes model training and benchmark comparison across the 5 classifier architectures.
        """
        # Chronological CV by default (rows must be date-sorted, as MLDataPreprocessor guarantees).
        cv = cv if cv is not None else TimeSeriesSplit(n_splits=n_splits)
        # Binary target: score CV on positive-class F1. Macro-F1 is 1.0 on any fold that contains
        # only the majority class, which rewards models for doing nothing when positives are rare.
        self.clf_scoring = "f1" if pd.Series(y_train).nunique() == 2 else "f1_macro"
        results: Dict[str, ClassificationModelMetrics] = {}

        # 1. Logistic Regression (Multinomial)
        print("[RestockClassifier] 1. Training Logistic Regression...")
        log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=self.random_state)
        log_cv = self.tuner.validate_with_kfold(
            model=log_reg,
            model_name="LogisticRegression",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=self.clf_scoring,
        )
        log_reg.fit(X_train, y_train)
        results["LogisticRegression"] = self._evaluate_model(
            name="LogisticRegression",
            model=log_reg,
            params={"max_iter": 1000},
            X_test=X_test,
            y_test=y_test,
            cv_res=log_cv,
        )

        # 2. Decision Tree Classifier (CART / ID3 philosophy)
        print("[RestockClassifier] 2. Tuning Decision Tree Classifier (Entropy / Gini)...")
        dt_param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [4, 6, 8, 12, None],
            "min_samples_split": [2, 5, 10],
        }
        dt_tuned = self.tuner.tune_hyperparameters(
            model=DecisionTreeClassifier(class_weight="balanced", random_state=self.random_state),
            model_name="DecisionTree",
            param_grid=dt_param_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring=self.clf_scoring,
        )
        dt_model = dt_tuned.best_estimator
        dt_cv = self.tuner.validate_with_kfold(
            model=dt_model,
            model_name="DecisionTree",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=self.clf_scoring,
        )
        results["DecisionTree"] = self._evaluate_model(
            name="DecisionTree",
            model=dt_model,
            params=dt_tuned.best_params,
            X_test=X_test,
            y_test=y_test,
            cv_res=dt_cv,
        )

        # 3. Gaussian Naïve Bayes
        print("[RestockClassifier] 3. Training Gaussian Naïve Bayes...")
        gnb = GaussianNB()
        gnb_cv = self.tuner.validate_with_kfold(
            model=gnb,
            model_name="NaiveBayes",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=self.clf_scoring,
        )
        gnb.fit(X_train, y_train)
        results["NaiveBayes"] = self._evaluate_model(
            name="NaiveBayes",
            model=gnb,
            params={},
            X_test=X_test,
            y_test=y_test,
            cv_res=gnb_cv,
        )

        # 4. Support Vector Machine (SVM)
        print("[RestockClassifier] 4. Training Support Vector Classifier (SVM RBF Kernel)...")
        # Use CalibratedClassifierCV to provide well-calibrated class probabilities without scikit-learn deprecation
        try:
            from sklearn.calibration import CalibratedClassifierCV
            svm_model = CalibratedClassifierCV(estimator=SVC(kernel="rbf", C=2.0, class_weight="balanced", random_state=self.random_state), ensemble=False)
        except Exception:
            svm_model = SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced", random_state=self.random_state)
        svm_cv = self.tuner.validate_with_kfold(
            model=svm_model,
            model_name="SVM",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=self.clf_scoring,
        )
        svm_model.fit(X_train, y_train)
        results["SVM"] = self._evaluate_model(
            name="SVM",
            model=svm_model,
            params={"kernel": "rbf", "C": 2.0},
            X_test=X_test,
            y_test=y_test,
            cv_res=svm_cv,
        )

        # 5. K-Nearest Neighbors (K-NN)
        print("[RestockClassifier] 5. Tuning K-Nearest Neighbors (K-NN)...")
        knn_grid = {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
        knn_tuned = self.tuner.tune_hyperparameters(
            model=KNeighborsClassifier(),
            model_name="KNN",
            param_grid=knn_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring=self.clf_scoring,
        )
        knn_model = knn_tuned.best_estimator
        knn_cv = self.tuner.validate_with_kfold(
            model=knn_model,
            model_name="KNN",
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=self.clf_scoring,
        )
        results["KNN"] = self._evaluate_model(
            name="KNN",
            model=knn_model,
            params=knn_tuned.best_params,
            X_test=X_test,
            y_test=y_test,
            cv_res=knn_cv,
        )

        # 6. Naive baseline: what a model that always predicts the majority class scores
        print("[RestockClassifier] 6. Majority-class baseline (context for every score above)...")
        dummy = DummyClassifier(strategy="most_frequent")
        dummy_cv = self.tuner.validate_with_kfold(
            model=dummy, model_name="Baseline_Majority", X=X_train, y=y_train, cv=cv, scoring=self.clf_scoring,
        )
        dummy.fit(X_train, y_train)
        results["Baseline_Majority"] = self._evaluate_model(
            name="Baseline_Majority", model=dummy, params={"strategy": "most_frequent"},
            X_test=X_test, y_test=y_test, cv_res=dummy_cv,
        )

        # Build comparative leaderboard
        rows = []
        for name, m in results.items():
            rows.append({
                "model": name,
                "best_params": str(m.best_params),
                "accuracy": m.accuracy,
                "precision_macro": m.precision_macro,
                "recall_macro": m.recall_macro,
                "f1_macro": m.f1_macro,
                "roc_auc_ovr": m.roc_auc_ovr,
                "pr_auc": m.pr_auc,
                "precision_pos": m.precision_pos,
                "recall_pos": m.recall_pos,
                "f1_pos": m.f1_pos,
                "cv_f1_mean": m.cv_f1_mean,
                "cv_f1_std": m.cv_f1_std,
                "overfitting_gap": m.overfitting_gap,
                "is_overfitting": m.is_overfitting,
            })
        # Champion = best CROSS-VALIDATED F1-macro among real models (never the baseline, never chosen on test).
        comparison_df = pd.DataFrame(rows).sort_values(by="cv_f1_mean", ascending=False).reset_index(drop=True)
        candidates = comparison_df[~comparison_df["model"].str.startswith("Baseline")]
        best_model_name = candidates.iloc[0]["model"]
        best_row = candidates.iloc[0]
        best_model_obj = results[best_model_name].model_obj

        print(f"\n[RestockClassifier] Classification Benchmark Leaderboard:")
        print(comparison_df.to_string(index=False))
        print(f" Best Classifier (by CV F1): {best_model_name} "
              f"(test F1-Macro={best_row['f1_macro']:.4f}, Accuracy={best_row['accuracy']:.4f})")
        if pd.notna(best_row.get('f1_pos')):
            pos_rate = float(np.mean(np.asarray(y_test) == 1))
            print(f" Positive-class (stock-out) rate in test period: {pos_rate:.1%}; "
                  f"champion precision={best_row['precision_pos']}, recall={best_row['recall_pos']}, F1={best_row['f1_pos']}")
            if best_row['f1_pos'] == 0:
                print(" WARNING: the champion never detects a stock-out in the test period. "
                      "Check the positive rate; the accuracy above may just be the majority-class rate.")

        return ClassificationBenchmark(
            comparison_table=comparison_df,
            best_model_name=best_model_name,
            best_model=best_model_obj,
            models=results,
        )

    def _evaluate_model(
        self,
        name: str,
        model: BaseEstimator,
        params: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv_res: ValidationResults,
    ) -> ClassificationModelMetrics:
        """Calculates binary / multi-class classification metrics."""
        m = classification_test_metrics(model, X_test, y_test)
        return ClassificationModelMetrics(
            model_name=name,
            best_params=params,
            accuracy=m["accuracy"],
            precision_macro=m["precision_macro"],
            recall_macro=m["recall_macro"],
            f1_macro=m["f1_macro"],
            roc_auc_ovr=m["roc_auc_ovr"],
            cv_f1_mean=cv_res.val_scores_mean,
            cv_f1_std=cv_res.val_scores_std,
            overfitting_gap=cv_res.overfitting_gap,
            is_overfitting=cv_res.is_overfitting,
            model_obj=model,
            precision_pos=m["precision_pos"],
            recall_pos=m["recall_pos"],
            f1_pos=m["f1_pos"],
            pr_auc=m["pr_auc"],
        )

    def save_best_model(self, benchmark: ClassificationBenchmark, output_dir: Path = MODELS_DIR) -> Path:
        """Saves best classifier artifact to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "best_restock_classifier.pkl"
        joblib.dump(benchmark.best_model, model_path)
        print(f"[RestockClassifier] Saved {benchmark.best_model_name} model to {model_path}")
        return model_path
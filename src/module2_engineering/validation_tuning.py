"""
Module 2 (UNIT-I): Model Validation & Hyperparameter Tuning Engine
==================================================================
Implements:
1. K-Fold / time-ordered cross-validation with train-vs-validation score comparison
   to diagnose overfitting and underfitting.
2. Hyperparameter tuning with GridSearchCV & RandomizedSearchCV.
3. Shared evaluation helpers used by every model module:
   - regression_baseline_table(): naive forecasts every ML model must beat
   - classification_test_metrics(): binary- and multi-class-safe test metrics
   - make_time_cv(): chronological CV splitter (optionally with an embargo gap)

CHANGES vs the previous version
-------------------------------
* Overfitting flag now works for ANY scoring (the old thresholds assumed scores in
  [0, 1], so negative-RMSE scores were never diagnosed correctly) and is stricter.
* Added make_time_cv, regression_baseline_table, classification_test_metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)

# Scorers whose value lives in [0, 1] (higher is better)
_BOUNDED_SCORING = {
    "r2", "accuracy", "f1", "f1_macro", "f1_weighted", "precision_macro",
    "recall_macro", "average_precision", "roc_auc", "roc_auc_ovr",
}


@dataclass
class ValidationResults:
    """Stores detailed cross validation evaluation metrics."""
    model_name: str
    train_scores_mean: float
    train_scores_std: float
    val_scores_mean: float
    val_scores_std: float
    overfitting_gap: float  # train_score - val_score (positive = worse on validation)
    is_overfitting: bool
    is_underfitting: bool
    fit_time_mean: float


@dataclass
class TuningResult:
    """Encapsulates hyperparameter search results."""
    model_name: str
    best_estimator: BaseEstimator
    best_params: Dict[str, Any]
    best_score: float
    cv_results_df: pd.DataFrame


def make_time_cv(n_splits: int = 5, gap_rows: int = 0) -> TimeSeriesSplit:
    """
    Chronological CV: every validation fold lies AFTER its training fold.
    Rows must be sorted by date. `gap_rows` removes rows between train and validation
    (embargo) so forward-looking labels cannot see across the boundary.
    """
    return TimeSeriesSplit(n_splits=n_splits, gap=gap_rows)


class ModelTuner:
    """Standardized validation and hyperparameter optimization framework."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def validate_with_kfold(
        self,
        model: BaseEstimator,
        model_name: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: Any,
        scoring: str = "r2",
    ) -> ValidationResults:
        """Runs cross-validation tracking train and validation scores."""
        scores = cross_validate(
            model, X, y, cv=cv, scoring=scoring, return_train_score=True, n_jobs=-1,
        )

        train_mean = float(np.mean(scores["train_score"]))
        train_std = float(np.std(scores["train_score"]))
        val_mean = float(np.mean(scores["test_score"]))
        val_std = float(np.std(scores["test_score"]))
        fit_time = float(np.mean(scores["fit_time"]))

        gap = train_mean - val_mean  # scorers are "higher is better", so gap > 0 = overfit

        if scoring in _BOUNDED_SCORING:
            is_overfitting = bool(gap > 0.05)
            is_underfitting = bool((train_mean < 0.40) and (val_mean < 0.40))
        else:
            # Scale-free scores such as neg_root_mean_squared_error
            rel_gap = gap / max(abs(train_mean), 1e-8)
            is_overfitting = bool(gap > 0 and rel_gap > 0.10)
            is_underfitting = False

        return ValidationResults(
            model_name=model_name,
            train_scores_mean=round(train_mean, 4),
            train_scores_std=round(train_std, 4),
            val_scores_mean=round(val_mean, 4),
            val_scores_std=round(val_std, 4),
            overfitting_gap=round(gap, 4),
            is_overfitting=is_overfitting,
            is_underfitting=is_underfitting,
            fit_time_mean=round(fit_time, 4),
        )

    def tune_hyperparameters(
        self,
        model: BaseEstimator,
        model_name: str,
        param_grid: Dict[str, Any],
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: Any,
        search_type: str = "grid",
        n_iter: int = 15,
        scoring: str = "r2",
    ) -> TuningResult:
        """Tunes model hyperparameters across CV folds using Grid or Randomized search."""
        if search_type == "grid":
            search = GridSearchCV(
                estimator=model, param_grid=param_grid, cv=cv,
                scoring=scoring, n_jobs=-1, refit=True,
            )
        else:
            search = RandomizedSearchCV(
                estimator=model, param_distributions=param_grid, n_iter=n_iter, cv=cv,
                scoring=scoring, random_state=self.random_state, n_jobs=-1, refit=True,
            )

        search.fit(X, y)

        return TuningResult(
            model_name=model_name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_score=round(float(search.best_score_), 4),
            cv_results_df=pd.DataFrame(search.cv_results_),
        )


# ----------------------------------------------------------------------
# Shared evaluation helpers
# ----------------------------------------------------------------------
def regression_baseline_table(
    y_true: Union[pd.Series, np.ndarray],
    baselines: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Scores naive forecasts (last value, moving averages, ...) on the same test rows."""
    y = np.asarray(y_true, dtype=float)
    rows = []
    for name, pred in baselines.items():
        p = np.asarray(pred, dtype=float)
        ok = ~np.isnan(p) & ~np.isnan(y)
        if ok.sum() == 0:
            continue
        yt, yp = y[ok], p[ok]
        rows.append({
            "model": f"Baseline: {name}",
            "test_rmse": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "test_mae": round(float(mean_absolute_error(yt, yp)), 4),
            "test_r2": round(float(r2_score(yt, yp)), 4),
            "test_wape": round(float(np.abs(yt - yp).sum() / max(np.abs(yt).sum(), 1e-8)), 4),
        })
    if not rows:
        return pd.DataFrame(columns=["model", "test_rmse", "test_mae", "test_r2", "test_wape"])
    return pd.DataFrame(rows).sort_values("test_rmse").reset_index(drop=True)


def classification_test_metrics(
    model: BaseEstimator,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> Dict[str, Optional[float]]:
    """
    Test-set metrics that work for binary AND multi-class targets.
    For a binary target the positive class (label 1) gets its own precision/recall/F1
    and PR-AUC, because macro averages hide how a rare class is handled.
    """
    y_true = np.asarray(y_test)
    y_pred = np.asarray(model.predict(X_test))

    out: Dict[str, Optional[float]] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc_ovr": None,
        "pr_auc": None,
        "precision_pos": None,
        "recall_pos": None,
        "f1_pos": None,
    }

    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.unique(y_true)
    binary = len(classes) == 2

    if binary:
        pos = classes[1]
        out["precision_pos"] = round(float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)), 4)
        out["recall_pos"] = round(float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0)), 4)
        out["f1_pos"] = round(float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)), 4)

    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X_test)
            if binary:
                out["roc_auc_ovr"] = round(float(roc_auc_score(y_true, prob[:, 1])), 4)
                out["pr_auc"] = round(float(average_precision_score(y_true, prob[:, 1])), 4)
            else:
                out["roc_auc_ovr"] = round(float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro")), 4)
        except Exception:
            pass  # e.g. only one class present in the test window
    return out
"""
Module 6 (UNIT-V): Ensemble Techniques (Model Optimization & Boosting)
======================================================================
Implements, tunes, and justifies advanced Bagging and Boosting algorithms:
1. Bagging:
   - Random Forest Regressor & Random Forest Classifier
2. Boosting:
   - AdaBoost (Adaptive Boosting)
   - Gradient Boosting (GBDT)
   - XGBoost (Extreme Gradient Boosting with regularized trees)

Outputs comparative performance matrices (RMSE/MAE for regression;
Accuracy, F1-Score, ROC-AUC for classification) to justify the final production choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import xgboost as xgb

from src.config import MODELS_DIR, REPORTS_DIR
from src.module2_engineering.validation_tuning import (
    ModelTuner,
    ValidationResults,
    classification_test_metrics,
    regression_baseline_table,
)


@dataclass
class EnsembleBenchmark:
    """Benchmark results for regression and classification ensembles."""
    regression_leaderboard: pd.DataFrame
    classification_leaderboard: pd.DataFrame
    best_regressor_name: str
    best_regressor: BaseEstimator
    best_classifier_name: str
    best_classifier: BaseEstimator
    justification_report: str


class EnsembleModelEngine:
    """
    Trains, tunes, and benchmarks Bagging and Boosting ensembles for retail supply chain optimization.
    """

    # leaderboard label -> key in `fitted_models`
    # (the old `best_name.split()[0]` lookup silently returned the Random Forest whenever
    #  'Random Forest (Bagging)' or 'Gradient Boosting' won)
    _LABEL_TO_KEY = {
        "Random Forest (Bagging)": "RandomForest",
        "AdaBoost": "AdaBoost",
        "Gradient Boosting": "GradientBoosting",
        "XGBoost": "XGBoost",
    }

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.tuner = ModelTuner(random_state=random_state)

    def benchmark_regression_ensembles(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_splits: int = 5,
        cv: Any = None,
        baselines: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[pd.DataFrame, str, BaseEstimator]:
        """
        Trains and tunes Random Forest, AdaBoost, Gradient Boosting, and XGBoost regressors.
        """
        # Chronological CV by default (shuffled K-Fold trains on days after the days it scores).
        cv = cv if cv is not None else TimeSeriesSplit(n_splits=n_splits)
        records = []
        fitted_models: Dict[str, BaseEstimator] = {}

        # 1. Bagging: Random Forest Regressor
        print("[EnsembleEngine] 1. Tuning Random Forest Regressor (Bagging)...")
        rf_grid = {"n_estimators": [50, 100], "max_depth": [6, 10, None]}
        rf_tuned = self.tuner.tune_hyperparameters(
            model=RandomForestRegressor(random_state=self.random_state, n_jobs=-1),
            model_name="RandomForestRegressor",
            param_grid=rf_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="neg_root_mean_squared_error",
        )
        rf_model = rf_tuned.best_estimator
        fitted_models["RandomForest"] = rf_model
        records.append(self._eval_reg("Random Forest (Bagging)", rf_model, rf_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # 2. Boosting: AdaBoost Regressor
        print("[EnsembleEngine] 2. Tuning AdaBoost Regressor (Boosting)...")
        ada_grid = {"n_estimators": [40, 80], "learning_rate": [0.05, 0.1, 0.3]}
        ada_tuned = self.tuner.tune_hyperparameters(
            model=AdaBoostRegressor(random_state=self.random_state),
            model_name="AdaBoostRegressor",
            param_grid=ada_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="neg_root_mean_squared_error",
        )
        ada_model = ada_tuned.best_estimator
        fitted_models["AdaBoost"] = ada_model
        records.append(self._eval_reg("AdaBoost", ada_model, ada_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # 3. Boosting: Gradient Boosting Regressor (GBDT)
        print("[EnsembleEngine] 3. Tuning Gradient Boosting Regressor (GBDT)...")
        gb_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        gb_tuned = self.tuner.tune_hyperparameters(
            model=GradientBoostingRegressor(random_state=self.random_state),
            model_name="GradientBoostingRegressor",
            param_grid=gb_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="neg_root_mean_squared_error",
        )
        gb_model = gb_tuned.best_estimator
        fitted_models["GradientBoosting"] = gb_model
        records.append(self._eval_reg("Gradient Boosting", gb_model, gb_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # 4. Boosting: XGBoost Regressor
        print("[EnsembleEngine] 4. Tuning XGBoost Regressor (Extreme Gradient Boosting)...")
        xgb_grid = {"n_estimators": [60, 120], "learning_rate": [0.05, 0.1], "max_depth": [3, 5], "subsample": [0.8, 1.0]}
        xgb_tuned = self.tuner.tune_hyperparameters(
            model=xgb.XGBRegressor(random_state=self.random_state, n_jobs=-1, objective="reg:squarederror"),
            model_name="XGBRegressor",
            param_grid=xgb_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring="neg_root_mean_squared_error",
        )
        xgb_model = xgb_tuned.best_estimator
        fitted_models["XGBoost"] = xgb_model
        records.append(self._eval_reg("XGBoost", xgb_model, xgb_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # Champion chosen on cross-validated RMSE (time-ordered folds), not on the test set.
        df_leaderboard = pd.DataFrame(records).sort_values(by="cv_rmse_mean", ascending=True).reset_index(drop=True)
        best_name = df_leaderboard.iloc[0]["ensemble_model"]
        best_estimator = fitted_models[self._LABEL_TO_KEY[best_name]]

        if baselines:
            base_df = regression_baseline_table(y_test, baselines).rename(columns={"model": "ensemble_model"})
            champ_rmse = float(df_leaderboard.iloc[0]["test_rmse"])
            best_base = float(base_df["test_rmse"].min())
            print(f"[EnsembleEngine] Champion test RMSE {champ_rmse:.4f} vs best naive baseline {best_base:.4f}")
            if champ_rmse >= best_base:
                print("[EnsembleEngine] WARNING: no ensemble beats the naive baseline - report this honestly.")
            df_leaderboard = pd.concat([df_leaderboard, base_df], ignore_index=True)

        return df_leaderboard, best_name, best_estimator

    def benchmark_classification_ensembles(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_splits: int = 5,
        cv: Any = None,
    ) -> Tuple[pd.DataFrame, str, BaseEstimator]:
        """
        Trains and tunes Random Forest, AdaBoost, Gradient Boosting, and XGBoost classifiers.
        """
        cv = cv if cv is not None else TimeSeriesSplit(n_splits=n_splits)
        n_classes = int(pd.Series(y_train).nunique())
        self.clf_scoring = "f1" if n_classes == 2 else "f1_macro"
        counts = pd.Series(y_train).value_counts()
        xgb_kwargs: Dict[str, Any] = {}
        if n_classes == 2 and counts.get(1, 0) > 0:  # re-weight the rare positive class
            xgb_kwargs["scale_pos_weight"] = float(counts.get(0, 1) / counts.get(1, 1))
        records = []
        fitted_models: Dict[str, BaseEstimator] = {}

        # 1. Bagging: Random Forest Classifier
        print("[EnsembleEngine] 5. Tuning Random Forest Classifier (Bagging)...")
        rf_grid = {"n_estimators": [50, 100], "max_depth": [6, 10, None], "criterion": ["gini", "entropy"]}
        rf_tuned = self.tuner.tune_hyperparameters(
            model=RandomForestClassifier(class_weight="balanced", random_state=self.random_state, n_jobs=-1),
            model_name="RandomForestClassifier",
            param_grid=rf_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring=self.clf_scoring,
        )
        rf_model = rf_tuned.best_estimator
        fitted_models["RandomForest"] = rf_model
        records.append(self._eval_clf("Random Forest (Bagging)", rf_model, rf_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # 2. Boosting: AdaBoost Classifier
        print("[EnsembleEngine] 6. Tuning AdaBoost Classifier (Boosting)...")
        ada_grid = {"n_estimators": [40, 80], "learning_rate": [0.05, 0.1, 0.2]}
        ada_tuned = self.tuner.tune_hyperparameters(
            model=AdaBoostClassifier(random_state=self.random_state),
            model_name="AdaBoostClassifier",
            param_grid=ada_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring=self.clf_scoring,
        )
        ada_model = ada_tuned.best_estimator
        fitted_models["AdaBoost"] = ada_model
        records.append(self._eval_clf("AdaBoost", ada_model, ada_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # 3. Boosting: Gradient Boosting Classifier (GBDT)
        print("[EnsembleEngine] 7. Tuning Gradient Boosting Classifier (GBDT)...")
        gb_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        gb_tuned = self.tuner.tune_hyperparameters(
            model=GradientBoostingClassifier(random_state=self.random_state),
            model_name="GradientBoostingClassifier",
            param_grid=gb_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring=self.clf_scoring,
        )
        gb_model = gb_tuned.best_estimator
        fitted_models["GradientBoosting"] = gb_model
        records.append(self._eval_clf("Gradient Boosting", gb_model, gb_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # 4. Boosting: XGBoost Classifier
        print("[EnsembleEngine] 8. Tuning XGBoost Classifier (Extreme Gradient Boosting)...")
        xgb_grid = {"n_estimators": [60, 120], "learning_rate": [0.05, 0.1], "max_depth": [3, 5], "subsample": [0.8, 1.0]}
        xgb_tuned = self.tuner.tune_hyperparameters(
            model=xgb.XGBClassifier(
                random_state=self.random_state, n_jobs=-1,
                eval_metric=("logloss" if n_classes == 2 else "mlogloss"), **xgb_kwargs,
            ),
            model_name="XGBClassifier",
            param_grid=xgb_grid,
            X=X_train,
            y=y_train,
            cv=cv,
            search_type="grid",
            scoring=self.clf_scoring,
        )
        xgb_model = xgb_tuned.best_estimator
        fitted_models["XGBoost"] = xgb_model
        records.append(self._eval_clf("XGBoost", xgb_model, xgb_tuned.best_params, X_train, y_train, X_test, y_test, cv))

        # Champion chosen on cross-validated F1-macro, not on the test set.
        df_leaderboard = pd.DataFrame(records).sort_values(by="cv_f1_mean", ascending=False).reset_index(drop=True)
        best_name = df_leaderboard.iloc[0]["ensemble_model"]
        best_estimator = fitted_models[self._LABEL_TO_KEY[best_name]]

        return df_leaderboard, best_name, best_estimator

    def _eval_reg(
        self,
        name: str,
        model: BaseEstimator,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: Any,
    ) -> Dict[str, Any]:
        """Evaluates regression ensemble on holdout test partition."""
        cv_res = self.tuner.validate_with_kfold(model, name, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        # CV R2 as well, so Module 4 and Module 6 can be ranked on one common, test-free metric
        cv_r2 = self.tuner.validate_with_kfold(model, name, X_train, y_train, cv=cv, scoring="r2")
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        wape = float(np.abs(np.asarray(y_test) - np.asarray(y_pred)).sum() / max(np.abs(np.asarray(y_test)).sum(), 1e-8))

        return {
            "ensemble_model": name,
            "best_params": str(params),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "test_r2": round(r2, 4),
            "test_wape": round(wape, 4),
            "cv_r2_mean": round(cv_r2.val_scores_mean, 4),
            "cv_rmse_mean": round(abs(cv_res.val_scores_mean), 4),
            "cv_rmse_std": round(cv_res.val_scores_std, 4),
            "overfitting_gap": round(abs(cv_res.overfitting_gap), 4),
            "is_overfitting": cv_res.is_overfitting,
        }

    def _eval_clf(
        self,
        name: str,
        model: BaseEstimator,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        cv: Any,
    ) -> Dict[str, Any]:
        """Evaluates classification ensemble on holdout test partition."""
        cv_res = self.tuner.validate_with_kfold(model, name, X_train, y_train, cv=cv, scoring=self.clf_scoring)
        m = classification_test_metrics(model, X_test, y_test)
        return {
            "ensemble_model": name,
            "best_params": str(params),
            **m,
            "cv_f1_mean": round(cv_res.val_scores_mean, 4),
            "cv_f1_std": round(cv_res.val_scores_std, 4),
            "overfitting_gap": round(cv_res.overfitting_gap, 4),
            "is_overfitting": cv_res.is_overfitting,
        }

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame) -> str:
        """Converts DataFrame to GitHub-flavored markdown table without requiring tabulate."""
        try:
            return df.to_markdown(index=False)
        except Exception:
            headers = "| " + " | ".join(df.columns) + " |"
            dividers = "| " + " | ".join(["---"] * len(df.columns)) + " |"
            rows = []
            for _, r in df.iterrows():
                rows.append("| " + " | ".join(str(val) for val in r.values) + " |")
            return "\n".join([headers, dividers] + rows)

    def generate_justification_report(
        self,
        reg_df: pd.DataFrame,
        clf_df: pd.DataFrame,
        best_reg_name: str,
        best_clf_name: str,
    ) -> str:
        """
        Synthesizes comparative metrics into an academic justification report.
        """
        report = []
        report.append("# Module 6 (Unit-V) Ensemble Model Justification Report\n")
        report.append("## 1. Demand Velocity Regression Benchmark")
        report.append(self._df_to_markdown(reg_df))
        report.append(f"\n**Selected Production Model**: **{best_reg_name}**")
        report.append("- **Justification**: Selected on cross-validated RMSE over chronological folds (the test period is used once, for reporting). Rows prefixed 'Baseline' are naive forecasts the model must beat.")

        report.append("\n## 2. Stock-out Risk Classification Benchmark")
        report.append(self._df_to_markdown(clf_df))
        report.append(f"\n**Selected Production Model**: **{best_clf_name}**")
        report.append("- **Justification**: Selected on cross-validated F1 (positive class when binary, macro otherwise) over chronological folds. Check positive-class precision/recall (precision_pos, recall_pos, f1_pos) and PR-AUC, which Macro F1 can hide when stock-outs are rare.")

        return "\n".join(report)
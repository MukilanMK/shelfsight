"""
Module 3 (UNIT-II): Multi-Method Feature Selection Framework
============================================================
1. Filter methods:   SelectKBest (f_regression / f_classif) and Random Forest importance (MDI)
2. Wrapper method:   Recursive Feature Elimination (RFE)
3. Embedded method:  L1 sparsity (LassoCV for regression, L1 logistic regression for classification)
4. Consensus:        average rank across the four methods

CHANGES vs the previous version
-------------------------------
* `task` argument: the classifier now gets features selected for ITS target
  (before, the regression target chose the features for both).
* Hard guard: raises if a `target_*` column or a rule-label column is in X.
* Warns when any single feature correlates > 0.98 with the target (typical leakage symptom).
* Optional correlation filter (default 0.95) removes near-duplicate columns
  (e.g. `sales_units` vs `units_sold_daily`) before ranking.
Call it on TRAINING rows only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.linear_model import LassoCV, LogisticRegressionCV

FORBIDDEN_EXACT = {"restock_urgency_tier", "restock_target_encoded"}


def assert_no_leakage(feature_names: Sequence[str]) -> None:
    """Fail loudly if label-like columns are in the feature matrix."""
    bad = [f for f in feature_names if f.startswith("target_") or f in FORBIDDEN_EXACT]
    if bad:
        raise ValueError(f"Leakage guard: label columns present in X: {bad}")


@dataclass
class FeatureSelectionSummary:
    """Summary rankings and consensus scores across all feature selection methods."""
    ranking_table: pd.DataFrame
    selected_features: List[str]
    filter_top_k: List[str]
    rf_top_k: List[str]
    rfe_selected: List[str]
    lasso_active_features: List[str]
    dropped_correlated: List[str] = field(default_factory=list)
    leakage_warnings: List[str] = field(default_factory=list)


class FeatureSelector:
    """Filter, wrapper and embedded feature selection with consensus ranking."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @staticmethod
    def _ranks(order_desc: np.ndarray, n: int) -> np.ndarray:
        ranks = np.empty_like(order_desc)
        ranks[order_desc] = np.arange(1, n + 1)
        return ranks

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        top_k: int = 15,
        task: str = "regression",
        corr_threshold: Optional[float] = 0.95,
    ) -> FeatureSelectionSummary:
        if task not in ("regression", "classification"):
            raise ValueError("task must be 'regression' or 'classification'")
        assert_no_leakage(X_train.columns)

        y = pd.Series(np.asarray(y_train), index=X_train.index)

        # Leakage symptom check
        warnings_list: List[str] = []
        corr_y = X_train.corrwith(y.astype(float)).abs()
        for feat, val in corr_y.items():
            if val > 0.98:
                msg = f"'{feat}' has |corr| = {val:.3f} with the target - check for leakage"
                warnings_list.append(msg)
                print(f"[FeatureSelector] WARNING: {msg}")

        # Remove near-duplicate columns
        dropped: List[str] = []
        X = X_train
        if corr_threshold is not None:
            corr = X.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            dropped = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
            X = X.drop(columns=dropped)
            if dropped:
                print(f"[FeatureSelector] Dropped {len(dropped)} near-duplicate features (|r| > {corr_threshold}).")

        names = X.columns.tolist()
        n = len(names)
        k = min(top_k, n)
        print(f"[FeatureSelector] Evaluating {n} candidate features for {task} target...")

        if task == "regression":
            score_func = f_regression
            rf = RandomForestRegressor(n_estimators=60, max_depth=8, random_state=self.random_state, n_jobs=-1)
            rfe_base = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=self.random_state, n_jobs=-1)
        else:
            score_func = f_classif
            rf = RandomForestClassifier(
                n_estimators=60, max_depth=8, class_weight="balanced",
                random_state=self.random_state, n_jobs=-1,
            )
            rfe_base = RandomForestClassifier(
                n_estimators=30, max_depth=6, class_weight="balanced",
                random_state=self.random_state, n_jobs=-1,
            )

        # 1a. Univariate filter
        print("[FeatureSelector] 1. Filter: univariate F-scores (SelectKBest)...")
        kbest = SelectKBest(score_func=score_func, k=k).fit(X, y)
        f_scores = np.nan_to_num(kbest.scores_, nan=0.0)
        kbest_order = np.argsort(f_scores)[::-1]
        kbest_ranks = self._ranks(kbest_order, n)
        filter_top = [names[i] for i in kbest_order[:k]]

        # 1b. Random Forest importance
        print("[FeatureSelector] 2. Filter: Random Forest feature importances...")
        rf.fit(X, y)
        rf_imp = rf.feature_importances_
        rf_order = np.argsort(rf_imp)[::-1]
        rf_ranks = self._ranks(rf_order, n)
        rf_top = [names[i] for i in rf_order[:k]]

        # 2. RFE
        print(f"[FeatureSelector] 3. Wrapper: RFE to {k} features...")
        rfe = RFE(estimator=rfe_base, n_features_to_select=k, step=2).fit(X, y)
        rfe_ranks = rfe.ranking_
        rfe_selected = [f for f, s in zip(names, rfe.support_) if s]

        # 3. Embedded L1
        if task == "regression":
            print("[FeatureSelector] 4. Embedded: LassoCV (L1)...")
            emb = LassoCV(cv=5, random_state=self.random_state, max_iter=2000).fit(X, y)
            coefs = np.abs(emb.coef_)
        else:
            print("[FeatureSelector] 4. Embedded: L1 logistic regression (CV)...")
            from sklearn.linear_model import LogisticRegression
            # scikit-learn 1.7+ has a known issue passing scoring='f1_macro' inside LogisticRegressionCV path
            emb = LogisticRegression(
                penalty="l1", solver="liblinear", C=0.5, class_weight="balanced",
                max_iter=300, random_state=self.random_state,
            ).fit(X, y)
            coefs = np.abs(emb.coef_).max(axis=0)
        active = [f for f, c in zip(names, coefs) if c > 1e-4]
        lasso_ranks = self._ranks(np.argsort(coefs)[::-1], n)

        composite = (kbest_ranks + rf_ranks + rfe_ranks + lasso_ranks) / 4.0
        ranking_df = pd.DataFrame({
            "feature": names,
            "f_score": f_scores,
            "kbest_rank": kbest_ranks,
            "rf_importance": rf_imp,
            "rf_rank": rf_ranks,
            "rfe_rank": rfe_ranks,
            "lasso_coef": coefs,
            "lasso_rank": lasso_ranks,
            "composite_rank_score": composite,
        }).sort_values("composite_rank_score").reset_index(drop=True)

        selected = ranking_df["feature"].head(k).tolist()
        print(f"[FeatureSelector] Consensus top {len(selected)} features:")
        for i, feat in enumerate(selected, 1):
            print(f"  {i}. {feat}")

        return FeatureSelectionSummary(
            ranking_table=ranking_df,
            selected_features=selected,
            filter_top_k=filter_top,
            rf_top_k=rf_top,
            rfe_selected=rfe_selected,
            lasso_active_features=active,
            dropped_correlated=dropped,
            leakage_warnings=warnings_list,
        )
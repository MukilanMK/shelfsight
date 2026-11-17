"""
Module 2 (UNIT-I): ML Preprocessing and Split Pipelines
======================================================
Leakage-safe preprocessing:
- Targets are FORWARD-looking columns built by FeatureEngineer
    regression:      target_units_next_day
    classification:  target_stockout_in_lead_time
- Chronological train/test split by DATE (not by row count), with an EMBARGO so that
  forward-looking labels of the last training days never overlap the test period.
- Median imputation, scaling and one-hot encoding are fitted on TRAIN rows only.
- Naive baselines (last value, moving averages, seasonal naive) are computed for the
  test rows so every model can be compared with them.
- The fitted preprocessor can transform new rows (`transform`) and be saved/loaded,
  which the dashboard needs because the pickled models alone cannot reproduce scaling.

CHANGES vs the previous version
-------------------------------
* Regression target was `units_sold_daily` (same-day sales) although a comment said
  "next-day"; classification target was the rule-based tier computed from the same
  features. Both leaked. Now: forward-looking targets only.
* Same-day sales (`units_sold_daily`) is now a legitimate FEATURE (known at end of day t).
* Rule-label columns and every `target_*` column are excluded from X.
* Imputation medians come from training rows only (previously from the full data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

REG_TARGET = "target_units_next_day"
CLF_TARGET = "target_stockout_in_lead_time"

# Columns that must never be model inputs
RULE_LABEL_COLS = ["restock_urgency_tier", "restock_target_encoded"]
META_COLS = [
    "sku_id", "date", "upc", "product_name", "batch_expiry_date",
    "supplier_lead_time_recorded", "baseline_tier",
]
REDUNDANT_COLS = ["revenue_daily", "daily_gross_profit"]  # duplicates of units x price/margin


class SplitStrategy(str, Enum):
    TEMPORAL = "temporal"
    RANDOM = "random"  # NOT recommended for time series; kept for comparison only


@dataclass
class PreprocessedDataset:
    """Encapsulates preprocessed training and testing partitions."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train_reg: pd.Series
    y_test_reg: pd.Series
    y_train_clf: pd.Series
    y_test_clf: pd.Series
    feature_names: List[str]
    train_dates: pd.Series
    test_dates: pd.Series
    train_end_date: Optional[pd.Timestamp] = None
    embargo_days: int = 0
    cv_gap_rows: int = 0
    baselines_test: Dict[str, np.ndarray] = field(default_factory=dict)


class MLDataPreprocessor:
    """Cleans, encodes, scales and partitions supply-chain features without leakage."""

    def __init__(
        self,
        scaler_type: str = "robust",
        test_ratio: float = 0.20,
        n_splits: int = 5,
        reg_target: str = REG_TARGET,
        clf_target: str = CLF_TARGET,
    ) -> None:
        self.scaler_type = scaler_type
        self.test_ratio = test_ratio
        self.n_splits = n_splits
        self.reg_target = reg_target
        self.clf_target = clf_target

        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names: List[str] = []
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.medians_: Optional[pd.Series] = None
        self.train_end_date: Optional[pd.Timestamp] = None
        self.embargo_days: int = 0

    # ------------------------------------------------------------------
    @staticmethod
    def compute_train_end_date(dates: Union[pd.Series, np.ndarray], test_ratio: float = 0.20) -> pd.Timestamp:
        """Last training DATE such that the final `test_ratio` of distinct dates is the test period.
        Call this BEFORE feature engineering and pass the same date to FeatureEngineer."""
        uniq = np.sort(pd.to_datetime(pd.Series(dates)).dt.normalize().unique())
        idx = max(int(len(uniq) * (1.0 - test_ratio)) - 1, 0)
        return pd.Timestamp(uniq[idx])

    def _select_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = set(META_COLS + RULE_LABEL_COLS + REDUNDANT_COLS)
        drop |= {c for c in df.columns if c.startswith("target_")}
        return df.drop(columns=[c for c in df.columns if c in drop])

    @staticmethod
    def _embargo_days(df: pd.DataFrame) -> int:
        """Forward labels look up to max(7, lead time) days ahead; drop that many days before the cutoff."""
        horizon = 7
        if "dynamic_lead_time_mean" in df.columns:
            lt = df["dynamic_lead_time_mean"].max()
            if np.isfinite(lt):
                horizon = max(horizon, int(np.ceil(lt)))
        return horizon

    @staticmethod
    def _naive_baselines(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Naive forecasts of next-day sales that any ML model must beat."""
        out: Dict[str, np.ndarray] = {}
        if "units_sold_daily" in df.columns:
            out["last value (tomorrow = today)"] = df["units_sold_daily"].to_numpy(dtype=float)
        for col, label in [("velocity_7d", "7-day mean"), ("velocity_14d", "14-day mean")]:
            if col in df.columns:
                out[label] = df[col].to_numpy(dtype=float)
        if {"units_sold_daily", "sku_id", "date"} <= set(df.columns):
            ordered = df.sort_values(["sku_id", "date"])
            lag6 = ordered.groupby("sku_id")["units_sold_daily"].shift(6).reindex(df.index)  # = same weekday as t+1
            fallback = df["velocity_7d"] if "velocity_7d" in df.columns else df["units_sold_daily"]
            out["same weekday last week"] = lag6.fillna(fallback).to_numpy(dtype=float)
        return out

    # ------------------------------------------------------------------
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        split_strategy: SplitStrategy = SplitStrategy.TEMPORAL,
        train_end_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> PreprocessedDataset:
        """Transforms the feature matrix into model-ready, leakage-safe X and y sets."""
        df_clean = df.copy()
        df_clean["date"] = pd.to_datetime(df_clean["date"])

        missing = [c for c in (self.reg_target, self.clf_target) if c not in df_clean.columns]
        if missing:
            raise KeyError(
                f"Missing target column(s) {missing}. Rebuild features with the updated "
                "FeatureEngineer (it creates the forward-looking target_* columns)."
            )

        baselines_all = self._naive_baselines(df_clean)
        feature_df = self._select_feature_columns(df_clean)

        self.cat_cols = [c for c in ["category", "storage_type"] if c in feature_df.columns]
        self.num_cols = [
            c for c in feature_df.columns
            if c not in self.cat_cols and pd.api.types.is_numeric_dtype(feature_df[c])
        ]
        feature_df = feature_df[self.num_cols + self.cat_cols].copy()
        feature_df[self.num_cols] = feature_df[self.num_cols].replace([np.inf, -np.inf], np.nan)

        valid = (df_clean[self.reg_target].notna() & df_clean[self.clf_target].notna()).to_numpy()
        dates = df_clean["date"]

        if split_strategy == SplitStrategy.TEMPORAL:
            cutoff = (
                pd.Timestamp(train_end_date) if train_end_date is not None
                else self.compute_train_end_date(dates, self.test_ratio)
            )
            self.embargo_days = self._embargo_days(df_clean)
            train_limit = cutoff - pd.Timedelta(days=self.embargo_days)
            train_idx = np.flatnonzero(valid & (dates <= train_limit).to_numpy())
            test_idx = np.flatnonzero(valid & (dates > cutoff).to_numpy())
            # keep rows chronologically ordered so TimeSeriesSplit is valid
            train_idx = train_idx[np.argsort(dates.to_numpy()[train_idx], kind="stable")]
            test_idx = test_idx[np.argsort(dates.to_numpy()[test_idx], kind="stable")]
            self.train_end_date = cutoff
        else:
            rng = np.random.RandomState(42)
            idx = np.flatnonzero(valid)
            rng.shuffle(idx)
            n_train = int(len(idx) * (1.0 - self.test_ratio))
            train_idx, test_idx = idx[:n_train], idx[n_train:]
            self.train_end_date, self.embargo_days = None, 0

        X_train_raw = feature_df.iloc[train_idx].reset_index(drop=True)
        X_test_raw = feature_df.iloc[test_idx].reset_index(drop=True)

        # Imputation statistics from TRAIN rows only
        self.medians_ = X_train_raw[self.num_cols].median().fillna(0.0)
        X_train_raw[self.num_cols] = X_train_raw[self.num_cols].fillna(self.medians_)
        X_test_raw[self.num_cols] = X_test_raw[self.num_cols].fillna(self.medians_)

        X_train_num = pd.DataFrame(self.scaler.fit_transform(X_train_raw[self.num_cols]), columns=self.num_cols)
        X_test_num = pd.DataFrame(self.scaler.transform(X_test_raw[self.num_cols]), columns=self.num_cols)

        if self.cat_cols:
            tr_cat = self.cat_encoder.fit_transform(X_train_raw[self.cat_cols])
            te_cat = self.cat_encoder.transform(X_test_raw[self.cat_cols])
            cat_names = self.cat_encoder.get_feature_names_out(self.cat_cols).tolist()
            X_train_final = pd.concat([X_train_num, pd.DataFrame(tr_cat, columns=cat_names)], axis=1)
            X_test_final = pd.concat([X_test_num, pd.DataFrame(te_cat, columns=cat_names)], axis=1)
        else:
            X_train_final, X_test_final = X_train_num, X_test_num

        self.feature_names = X_train_final.columns.tolist()

        n_skus_train = df_clean["sku_id"].iloc[train_idx].nunique() if "sku_id" in df_clean.columns else 1
        cv_gap_rows = int(self.embargo_days * n_skus_train) if split_strategy == SplitStrategy.TEMPORAL else 0

        return PreprocessedDataset(
            X_train=X_train_final,
            X_test=X_test_final,
            y_train_reg=df_clean[self.reg_target].iloc[train_idx].astype(float).reset_index(drop=True),
            y_test_reg=df_clean[self.reg_target].iloc[test_idx].astype(float).reset_index(drop=True),
            y_train_clf=df_clean[self.clf_target].iloc[train_idx].astype(int).reset_index(drop=True),
            y_test_clf=df_clean[self.clf_target].iloc[test_idx].astype(int).reset_index(drop=True),
            feature_names=self.feature_names,
            train_dates=df_clean["date"].iloc[train_idx].reset_index(drop=True),
            test_dates=df_clean["date"].iloc[test_idx].reset_index(drop=True),
            train_end_date=self.train_end_date,
            embargo_days=self.embargo_days,
            cv_gap_rows=cv_gap_rows,
            baselines_test={k: v[test_idx] for k, v in baselines_all.items()},
        )

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the FITTED preprocessing to any rows (e.g. today's snapshot, which has no target yet)."""
        if self.medians_ is None:
            raise RuntimeError("Call prepare_dataset() before transform().")
        feature_df = self._select_feature_columns(df.copy())
        feature_df = feature_df.reindex(columns=self.num_cols + self.cat_cols)
        num = feature_df[self.num_cols].replace([np.inf, -np.inf], np.nan).fillna(self.medians_)
        parts = [pd.DataFrame(self.scaler.transform(num), columns=self.num_cols, index=feature_df.index)]
        if self.cat_cols:
            cat = self.cat_encoder.transform(feature_df[self.cat_cols])
            parts.append(pd.DataFrame(
                cat, columns=self.cat_encoder.get_feature_names_out(self.cat_cols), index=feature_df.index,
            ))
        return pd.concat(parts, axis=1)[self.feature_names]

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Union[str, Path]) -> "MLDataPreprocessor":
        return joblib.load(path)

    # ------------------------------------------------------------------
    def make_time_cv(self, dataset: PreprocessedDataset, n_splits: Optional[int] = None) -> TimeSeriesSplit:
        """Chronological CV with an embargo gap matching the label horizon."""
        return TimeSeriesSplit(n_splits=n_splits or self.n_splits, gap=dataset.cv_gap_rows)

    def get_cv_splitters(self) -> Dict[str, Union[KFold, StratifiedKFold, TimeSeriesSplit]]:
        """Provides configured cross-validation splitters (use time_series_split for model selection)."""
        return {
            "standard_kfold": KFold(n_splits=self.n_splits, shuffle=True, random_state=42),
            "stratified_kfold": StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42),
            "time_series_split": TimeSeriesSplit(n_splits=self.n_splits),
        }
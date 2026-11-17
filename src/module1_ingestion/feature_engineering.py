"""
Feature Engineering Engine (leakage-safe revision)
==================================================
Transforms raw cleaned sales, inventory, and product master tables
into a daily SKU-level feature dataset.

All features at row (sku, date t) use information available at the END of day t.
All supervised targets look FORWARD (t+1 ... t+h) and are named `target_*`.
Never put `target_*` columns, or the rule-based label columns
(`restock_urgency_tier`, `restock_target_encoded`), into a model's feature matrix.

Features Engineered:
1. Temporal & calendar dynamics.
2. Rolling velocities (7/14/30-day means and standard deviations).
3. Empirical supplier lead times (Empirical Bayes shrinkage), optionally fit on
   training dates only (train_end_date).
4. Days of Inventory Remaining (DIR) & inventory turnover.
5. GMROI.
6. Statistical safety stock and reorder point (ROP).
7. Trend acceleration.
8. Rule-based restock tier (business rule on stock/ROP; cutoffs are set by the
   user via `tier_cutoffs`, they are NOT learned from data).
9. NEW: forward-looking targets (next-day units, next-7-day units,
   stock-out within lead time).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class FeatureEngineer:
    def __init__(
        self,
        service_level: float = 0.95,
        tier_cutoffs: Tuple[float, float, float] = (0.6, 1.1, 2.8),  # CHANGED: explicit, not hidden
    ) -> None:
        self.service_level = service_level
        self.z_service = stats.norm.ppf(self.service_level)
        self.tier_cutoffs = tier_cutoffs
        self._train_end: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    def build_feature_dataset(
        self,
        products_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        train_end_date: Optional[str] = None,  # NEW
    ) -> pd.DataFrame:
        """
        train_end_date: last date of the training period (e.g. "2025-09-30").
        Data-derived constants (lead times, velocity floors) are fit only on
        rows up to this date, so test-period information does not leak in.
        """
        self._train_end = pd.Timestamp(train_end_date) if train_end_date else None

        print("[FeatureEngineer] Aggregating daily sales per product...")
        daily_sales = self._aggregate_daily_sales(sales_df)

        print("[FeatureEngineer] Merging inventory snapshots with product master and sales...")
        merged = pd.merge(inventory_df, products_df, on="sku_id", how="left", suffixes=("", "_prod"))
        merged = pd.merge(merged, daily_sales, on=["date", "sku_id"], how="left")

        merged["units_sold_daily"] = merged["units_sold_daily"].fillna(0)
        merged["revenue_daily"] = merged["revenue_daily"].fillna(0)
        merged["txn_count"] = merged["txn_count"].fillna(0)
        merged["avg_discount_pct"] = merged["avg_discount_pct"].fillna(0)
        merged["daily_gross_profit"] = merged["units_sold_daily"] * (
            merged["selling_price"] - merged["cost_price"]
        )

        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values(by=["sku_id", "date"]).reset_index(drop=True)

        print("[FeatureEngineer] Computing temporal calendar features...")
        merged = self._add_temporal_features(merged)

        print("[FeatureEngineer] Computing rolling velocity and demand variances...")
        merged = self._add_rolling_velocities(merged)

        print("[FeatureEngineer] Estimating empirical supplier lead times...")
        merged = self._add_dynamic_lead_times(merged)

        print("[FeatureEngineer] Calculating DIR, GMROI, and safety stock...")
        merged = self._add_inventory_and_financial_metrics(merged)

        print("[FeatureEngineer] Calculating trend acceleration and rule-based restock tiers...")
        merged = self._add_trend_and_target_labels(merged)

        print("[FeatureEngineer] Building forward-looking targets...")  # NEW
        merged = self._add_forward_targets(merged)

        return merged

    # ------------------------------------------------------------------
    def _fit_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows allowed to be used when fitting data-derived constants."""
        if self._train_end is None:
            return df
        return df[df["date"] <= self._train_end]

    def _aggregate_daily_sales(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        # CHANGED: removed unused `cost_component` column
        return (
            sales_df.groupby(["date", "sku_id"])
            .agg(
                units_sold_daily=("units_sold", "sum"),
                revenue_daily=("total_amount", "sum"),
                txn_count=("transaction_id", "count"),
                avg_discount_pct=("discount_pct", "mean"),
            )
            .reset_index()
        )

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["day_of_month"] = df["date"].dt.day
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["day_of_year"] = df["date"].dt.dayofyear
        df["is_payday"] = df["day_of_month"].isin([1, 2, 15, 16]).astype(int)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
        return df

    def _add_rolling_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Windows end at day t (inclusive). Safe because targets are t+1 or later."""
        grouped = df.groupby("sku_id")["units_sold_daily"]

        df["velocity_7d"] = grouped.transform(lambda s: s.rolling(7, min_periods=1).mean())
        df["demand_std_7d"] = grouped.transform(lambda s: s.rolling(7, min_periods=2).std().fillna(0))
        df["velocity_14d"] = grouped.transform(lambda s: s.rolling(14, min_periods=1).mean())
        df["demand_std_14d"] = grouped.transform(lambda s: s.rolling(14, min_periods=2).std().fillna(0))
        df["velocity_30d"] = grouped.transform(lambda s: s.rolling(30, min_periods=1).mean())
        df["demand_std_30d"] = grouped.transform(lambda s: s.rolling(30, min_periods=2).std().fillna(0))

        fit = self._fit_frame(df)  # CHANGED: floor fit on training dates only
        positive_vel = fit.loc[fit["velocity_30d"] > 0, "velocity_30d"]
        vel_floor = positive_vel.quantile(0.05) if len(positive_vel) > 0 else 0.1
        df["seasonality_index"] = df["velocity_7d"] / np.maximum(df["velocity_30d"], vel_floor)
        return df

    def _add_dynamic_lead_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Empirical Bayes lead times. CHANGED: fit on training dates only."""
        arrival_events = self._fit_frame(df).dropna(subset=["supplier_lead_time_recorded"])

        cat_stats = arrival_events.groupby("category")["supplier_lead_time_recorded"].agg(
            cat_lt_mean="mean", cat_lt_std="std"
        ).reset_index()
        cat_stats["cat_lt_std"] = cat_stats["cat_lt_std"].fillna(0.5)

        sku_stats = arrival_events.groupby("sku_id")["supplier_lead_time_recorded"].agg(
            sku_lt_count="count", sku_lt_mean="mean", sku_lt_std="std"
        ).reset_index()

        sku_meta = df[["sku_id", "category"]].drop_duplicates()
        merged_lt = pd.merge(sku_meta, cat_stats, on="category", how="left")
        merged_lt = pd.merge(merged_lt, sku_stats, on="sku_id", how="left")

        merged_lt["sku_lt_count"] = merged_lt["sku_lt_count"].fillna(0)
        n = merged_lt["sku_lt_count"]
        weight = n / (n + 3.0)  # CHANGED: comment previously said n+5, code uses 3

        merged_lt["dynamic_lead_time_mean"] = (
            weight * merged_lt["sku_lt_mean"].fillna(merged_lt["cat_lt_mean"])
            + (1 - weight) * merged_lt["cat_lt_mean"]
        )
        merged_lt["dynamic_lead_time_std"] = (
            weight * merged_lt["sku_lt_std"].fillna(merged_lt["cat_lt_std"])
            + (1 - weight) * merged_lt["cat_lt_std"]
        ).clip(lower=0.2)

        lt_lookup = merged_lt[["sku_id", "dynamic_lead_time_mean", "dynamic_lead_time_std"]]
        return pd.merge(df, lt_lookup, on="sku_id", how="left")

    def _add_inventory_and_financial_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        fit = self._fit_frame(df)  # CHANGED
        pos_vel = fit.loc[fit["velocity_14d"] > 0, "velocity_14d"]
        vel_eps = pos_vel.quantile(0.05) if len(pos_vel) > 0 else 0.1

        df["days_of_inventory_remaining"] = df["ending_stock"] / np.maximum(df["velocity_14d"], vel_eps)

        avg_inv_30d = df.groupby("sku_id")["ending_stock"].transform(lambda s: s.rolling(30, min_periods=1).mean())
        sales_sum_30d = df.groupby("sku_id")["units_sold_daily"].transform(lambda s: s.rolling(30, min_periods=1).sum())
        df["inventory_turn_ratio_30d"] = sales_sum_30d / np.maximum(avg_inv_30d, 1.0)

        df["daily_inv_cost"] = df["ending_stock"] * df["cost_price"]
        df["rolling_gp_30d"] = df.groupby("sku_id")["daily_gross_profit"].transform(
            lambda s: s.rolling(30, min_periods=1).sum()
        )
        df["avg_inv_cost_30d"] = df.groupby("sku_id")["daily_inv_cost"].transform(
            lambda s: s.rolling(30, min_periods=1).mean()
        )
        df["gmroi"] = df["rolling_gp_30d"] / np.maximum(df["avg_inv_cost_30d"], 1.0)

        mu_d, sigma_d = df["velocity_14d"], df["demand_std_14d"]
        mu_l, sigma_l = df["dynamic_lead_time_mean"], df["dynamic_lead_time_std"]
        variance_total = (mu_l * sigma_d ** 2) + (mu_d ** 2 * sigma_l ** 2)
        df["dynamic_safety_stock"] = np.ceil(self.z_service * np.sqrt(np.maximum(variance_total, 0.01)))
        df["dynamic_reorder_point"] = np.ceil((mu_d * mu_l) + df["dynamic_safety_stock"])

        if "batch_expiry_date" in df.columns:
            df["days_to_expiry"] = (df["batch_expiry_date"] - df["date"]).dt.days.fillna(999.0)
        else:
            df["days_to_expiry"] = 999.0
        return df

    def _add_trend_and_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend features and the RULE-BASED restock tier.
        The tier is a deterministic function of stock_to_rop_ratio, so it is a
        business rule / reporting label. Do NOT train a model to 'predict' it from
        stock_to_rop_ratio or DIR; use `target_stockout_in_lead_time` instead.
        """
        vel_diff = df["velocity_7d"] - df["velocity_30d"]
        fallback = df["demand_std_30d"].mean() or 1.0
        std_diff = df["demand_std_30d"].replace(0, np.nan).fillna(fallback)
        df["trend_acceleration_zscore"] = vel_diff / std_diff

        df["stock_to_rop_ratio"] = df["ending_stock"] / np.maximum(df["dynamic_reorder_point"], 1.0)

        c1, c2, c3 = self.tier_cutoffs  
        r = df["stock_to_rop_ratio"]
        conditions = [r <= c1, (r > c1) & (r <= c2), (r > c2) & (r <= c3), r > c3]
        choices = ["CRITICAL_RESTOCK", "REORDER_RECOMMENDED", "OPTIMAL_STOCK", "EXCESS_OVERSTOCK"]
        df["restock_urgency_tier"] = np.select(conditions, choices, default="OPTIMAL_STOCK")

        tier_mapping = {"CRITICAL_RESTOCK": 0, "REORDER_RECOMMENDED": 1, "OPTIMAL_STOCK": 2, "EXCESS_OVERSTOCK": 3}
        df["restock_target_encoded"] = df["restock_urgency_tier"].map(tier_mapping)
        return df

    # ------------------------------------------------------------------
    # NEW: forward-looking targets
    # ------------------------------------------------------------------
    @staticmethod
    def _forward(s: pd.Series, window: int, how: str) -> pd.Series:
        """Aggregate over t+1 ... t+window (strictly future). NaN where the window is incomplete."""
        rev = s.iloc[::-1]
        out = getattr(rev.rolling(window, min_periods=window), how)().iloc[::-1]
        return out.shift(-1)

    def _add_forward_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("sku_id")

        # Regression targets
        df["target_units_next_day"] = g["units_sold_daily"].shift(-1)
        df["target_units_next_7d"] = g["units_sold_daily"].transform(lambda s: self._forward(s, 7, "sum"))

        # Classification target: will stock hit zero within the next L days (L = lead time)?
        horizon = np.ceil(df["dynamic_lead_time_mean"]).clip(lower=1).fillna(1).astype(int)
        flag = pd.Series(np.nan, index=df.index)
        for _, idx in g.groups.items():
            h = int(horizon.loc[idx].iloc[0])
            fut_min = self._forward(df.loc[idx, "ending_stock"], h, "min")
            flag.loc[idx] = np.where(fut_min.isna(), np.nan, (fut_min <= 0).astype(float))
        df["target_stockout_in_lead_time"] = flag
        return df
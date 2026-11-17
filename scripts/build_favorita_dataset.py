"""
scripts/build_favorita_dataset.py
=================================
Builds a realistic grocery retail dataset from the real Corporación Favorita data:
1. Filters Store 44 and the 60 gap-verified items over the 731-day window (2015-08-16 to 2017-08-15).
2. Excludes the 4 store-closure dates (2015-12-25, 2016-01-01, 2016-12-25, 2017-01-01).
3. Emits `products.csv` with simulated costs/margins, selling_price = cost / (1 - margin), shelf lives, and storage types.
4. Emits `sales.csv` with unmodified real unit_sales as units_sold (demand proxy), strictly on days with positive sales.
5. Emits `inventory.csv` simulating a non-circular periodic order-up-to policy with FIFO perishable batch tracking.
6. Calibrates parameters against the real pipeline target: target_stockout_in_lead_time == 1 within 10-35%.
"""

from __future__ import annotations

import json
import shutil
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import yaml

from src.module1_ingestion.feature_engineering import FeatureEngineer
from src.module2_engineering.preprocessing import MLDataPreprocessor
CONFIG_PATH = BASE_DIR / "config" / "favorita_adapter.yaml"
FAVORITA_DIR = BASE_DIR / "ds" / "favorita-grocery-sales-forecasting"
RAW_OUT_DIR = BASE_DIR / "data" / "raw"
SCRATCH_DIR = BASE_DIR / "scratch"


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_favorita_dataset():
    t0 = time.time()
    cfg = load_config()
    adapter_cfg = cfg["adapter"]
    inv_cfg = cfg["inventory_policy"]
    pricing_cfg = cfg["pricing_and_shelf_life"]
    promo_cfg = cfg["promotion"]
    calib_cfg = cfg["calibration"]

    seed = int(adapter_cfg.get("seed", 42))
    rng = np.random.RandomState(seed)

    store_nbr = int(adapter_cfg["store_nbr"])
    start_date = adapter_cfg["date_window"]["start_date"]
    end_date = adapter_cfg["date_window"]["end_date"]
    closures = set(adapter_cfg["date_window"]["store_closures"])

    # 1. Operational dates
    full_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start_date, end_date)]
    op_dates = [d for d in full_dates if d not in closures]
    date_to_idx = {d: i for i, d in enumerate(op_dates)}
    n_days = len(op_dates)
    print(f"[Build] Operational dates: {n_days} days (from {start_date} to {end_date}, {len(closures)} closures excluded)")

    # 2. Load 60-item roster
    roster_path = SCRATCH_DIR / "final_gap_verified_60_roster.csv"
    if not roster_path.exists():
        raise FileNotFoundError(f"Roster not found at {roster_path}")
    roster_df = pd.read_csv(roster_path)
    selected_items = roster_df["item_nbr"].tolist()
    item_meta_map = roster_df.set_index("item_nbr").to_dict(orient="index")
    selected_set = set(selected_items)
    print(f"[Build] Selected 60 items loaded. Perishables: {roster_df['perishable'].sum()}/60")

    # 3. Read real sales from train.csv in chunks
    train_path = FAVORITA_DIR / "train.csv" / "train.csv"
    print(f"[Build] Scanning {train_path} for Store {store_nbr} and 60 items...")

    sales_records = []
    chunk_size = 4_000_000
    dtypes = {
        "date": "str",
        "store_nbr": "int16",
        "item_nbr": "int32",
        "unit_sales": "float32",
        "onpromotion": "float32",
    }

    negative_returns_count = 0
    total_raw_rows_scanned = 0

    for chunk in pd.read_csv(
        train_path,
        chunksize=chunk_size,
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype=dtypes,
    ):
        total_raw_rows_scanned += len(chunk)
        chunk = chunk[(chunk["store_nbr"] == store_nbr) & (chunk["date"] >= start_date) & (chunk["date"] <= end_date)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["item_nbr"].isin(selected_set)]
        if chunk.empty:
            continue

        # Check negative unit_sales (returns)
        neg_mask = chunk["unit_sales"] < 0
        if neg_mask.any():
            negative_returns_count += int(neg_mask.sum())
            chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0.0)

        for row in chunk.itertuples(index=False):
            if row.date in closures:
                continue
            sales_records.append((row.date, row.item_nbr, float(row.unit_sales), float(row.onpromotion) if not np.isnan(row.onpromotion) else 0.0))

    print(f"[Build] Scanned real train rows. Raw sales entries: {len(sales_records)}. Negative returns clipped to 0: {negative_returns_count}")

    # Build daily demand matrix: shape (60, n_days)
    # real unit_sales is the unmodified demand ground truth
    item_to_row = {itm: i for i, itm in enumerate(selected_items)}
    demand_matrix = np.zeros((len(selected_items), n_days), dtype=np.float32)
    onpromo_matrix = np.zeros((len(selected_items), n_days), dtype=np.float32)

    for d_str, itm, u_sales, onpromo in sales_records:
        if d_str in date_to_idx and itm in item_to_row:
            r_idx = item_to_row[itm]
            c_idx = date_to_idx[d_str]
            demand_matrix[r_idx, c_idx] = u_sales
            onpromo_matrix[r_idx, c_idx] = onpromo

    # 4. Generate Products Master
    print("[Build] Generating products master with pricing and shelf life...")
    products_rows = []
    item_prices = {}

    for itm in selected_items:
        meta = item_meta_map[itm]
        fam = meta["family"]
        is_per = int(meta["perishable"])
        fam_p = pricing_cfg.get(fam, pricing_cfg["GROCERY I"])

        cost = round(float(rng.uniform(fam_p["cost_range"][0], fam_p["cost_range"][1])), 2)
        margin = round(float(rng.uniform(fam_p["margin_pct_range"][0], fam_p["margin_pct_range"][1])), 4)
        selling_price = round(cost / (1.0 - margin), 2)
        margin_pct = round((selling_price - cost) / selling_price, 4)

        shelf_life = int(rng.randint(fam_p["shelf_life_days"][0], fam_p["shelf_life_days"][1] + 1))
        storage = fam_p["storage_type"]

        sku_id = f"SKU-GROC-{itm}"
        prod_name = f"{fam.title()} item {itm}"

        # Assign simulated upc
        upc = str(itm).zfill(12)

        products_rows.append({
            "sku_id": sku_id,
            "product_name": prod_name,
            "category": fam,
            "cost_price": cost,
            "selling_price": selling_price,
            "margin_pct": margin_pct,
            "is_perishable": bool(is_per),
            "shelf_life_days": shelf_life,
            "storage_type": storage,
            "upc": upc,
        })

        item_prices[itm] = {
            "sku_id": sku_id,
            "product_name": prod_name,
            "cost": cost,
            "selling_price": selling_price,
            "shelf_life": shelf_life,
            "is_perishable": bool(is_per),
            "family": fam,
        }

    products_df = pd.DataFrame(products_rows)

    # 5. Iterative Simulation & Calibration
    print("\n[Build] Starting inventory simulation and target calibration...")
    best_inv_df = None
    best_sales_df = None
    calib_history = []

    # Calibration parameters to tune
    buf_per_min = float(inv_cfg["buffer_factor"]["perishable"]["min"])
    buf_per_max = float(inv_cfg["buffer_factor"]["perishable"]["max"])
    buf_non_min = float(inv_cfg["buffer_factor"]["non_perishable"]["min"])
    buf_non_max = float(inv_cfg["buffer_factor"]["non_perishable"]["max"])
    miss_prob = float(inv_cfg.get("miss_review_prob", 0.05))
    late_prob = float(inv_cfg["lead_time"].get("late_delivery_prob", 0.06))

    target_min = float(calib_cfg["target_band"]["min_pct"])
    target_max = float(calib_cfg["target_band"]["max_pct"])
    max_iter = int(calib_cfg.get("max_iterations", 8))

    for iter_idx in range(1, max_iter + 1):
        print(f"\n--- Calibration Iteration {iter_idx}/{max_iter} ---")
        print(f"    Params: buf_per=[{buf_per_min:.2f}, {buf_per_max:.2f}], buf_non=[{buf_non_min:.2f}, {buf_non_max:.2f}], miss={miss_prob:.2f}, late={late_prob:.2f}")

        sim_rng = np.random.RandomState(seed + iter_idx * 100)
        inv_df, sales_df = simulate_inventory(
            selected_items=selected_items,
            item_prices=item_prices,
            op_dates=op_dates,
            demand_matrix=demand_matrix,
            onpromo_matrix=onpromo_matrix,
            inv_cfg=inv_cfg,
            promo_cfg=promo_cfg,
            buf_per_range=(buf_per_min, buf_per_max),
            buf_non_range=(buf_non_min, buf_non_max),
            miss_prob=miss_prob,
            late_prob=late_prob,
            rng=sim_rng,
        )

        # Test calibration on real pipeline label: target_stockout_in_lead_time == 1
        fe = FeatureEngineer(service_level=0.95)
        # Compute cutoff date for 80/20 train/test
        cutoff = MLDataPreprocessor.compute_train_end_date(inv_df["date"], test_ratio=0.20)
        ft_df = fe.build_feature_dataset(
            products_df=products_df,
            sales_df=sales_df,
            inventory_df=inv_df,
            train_end_date=str(cutoff.date()),
        )

        train_mask = ft_df["date"] <= cutoff
        test_mask = ft_df["date"] > cutoff

        valid_lbl = ft_df["target_stockout_in_lead_time"].dropna()
        overall_rate = float(100.0 * (valid_lbl == 1).mean())

        train_lbl = ft_df.loc[train_mask, "target_stockout_in_lead_time"].dropna()
        test_lbl = ft_df.loc[test_mask, "target_stockout_in_lead_time"].dropna()

        train_rate = float(100.0 * (train_lbl == 1).mean()) if len(train_lbl) > 0 else 0.0
        test_rate = float(100.0 * (test_lbl == 1).mean()) if len(test_lbl) > 0 else 0.0

        print(f"    Achieved Target Stock-out Rates: Overall={overall_rate:.2f}%, Train={train_rate:.2f}%, Test={test_rate:.2f}%")

        # Per-SKU stockout metrics: share of ending_stock == 0
        sku_stockout_shares = inv_df.groupby("sku_id")["ending_stock"].apply(lambda s: (s == 0).mean())
        sku_min = float(sku_stockout_shares.min() * 100.0)
        sku_med = float(sku_stockout_shares.median() * 100.0)
        sku_max = float(sku_stockout_shares.max() * 100.0)
        sku_with_any = float((sku_stockout_shares > 0).mean() * 100.0)

        print(f"    Per-SKU ending_stock == 0: Min={sku_min:.1f}%, Med={sku_med:.1f}%, Max={sku_max:.1f}%, SKUs with >=1 event={sku_with_any:.1f}%")

        calib_history.append({
            "iteration": iter_idx,
            "buf_per_range": [round(buf_per_min, 3), round(buf_per_max, 3)],
            "buf_non_range": [round(buf_non_min, 3), round(buf_non_max, 3)],
            "miss_prob": round(miss_prob, 3),
            "late_prob": round(late_prob, 3),
            "overall_rate_pct": round(overall_rate, 2),
            "train_rate_pct": round(train_rate, 2),
            "test_rate_pct": round(test_rate, 2),
            "sku_stockout_min": round(sku_min, 2),
            "sku_stockout_med": round(sku_med, 2),
            "sku_stockout_max": round(sku_max, 2),
            "skus_with_any_pct": round(sku_with_any, 2),
        })

        best_inv_df = inv_df
        best_sales_df = sales_df

        # Check conditions: target_stockout_in_lead_time in [10%, 35%] on both train and test,
        # no SKU > 60%, and >= 80% of SKUs have >= 1 event
        passed_band = (target_min <= train_rate <= target_max) and (target_min <= test_rate <= target_max)
        passed_sku_dist = (sku_max <= 60.0) and (sku_with_any >= 80.0)

        if passed_band and passed_sku_dist:
            print(f"[Build] SUCCESS: Calibration criteria satisfied on iteration {iter_idx}!")
            break

        # Adjust parameters strictly for the next iteration
        if train_rate < target_min:
            # Under stockout target: decrease buffers, slightly increase miss/late
            buf_per_min = max(0.95, buf_per_min - 0.04)
            buf_per_max = max(1.05, buf_per_max - 0.04)
            buf_non_min = max(0.98, buf_non_min - 0.04)
            buf_non_max = max(1.10, buf_non_max - 0.04)
            miss_prob = min(0.12, miss_prob + 0.01)
            late_prob = min(0.12, late_prob + 0.01)
        elif train_rate > target_max:
            # Over stockout target: increase buffers, decrease miss/late
            buf_per_min += 0.04
            buf_per_max += 0.04
            buf_non_min += 0.04
            buf_non_max += 0.04
            miss_prob = max(0.02, miss_prob - 0.01)
            late_prob = max(0.02, late_prob - 0.01)
        elif sku_max > 60.0:
            # A few SKUs starved: boost lower bound of buffers
            buf_per_min += 0.05
            buf_non_min += 0.05

    # 6. Save calibrated outputs to data/raw/
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    products_out = RAW_OUT_DIR / "products.csv"
    sales_out = RAW_OUT_DIR / "sales.csv"
    inv_out = RAW_OUT_DIR / "inventory.csv"

    print("\n[Build] Saving raw CSV deliverables...")
    products_df.to_csv(products_out, index=False)
    best_sales_df.to_csv(sales_out, index=False)
    best_inv_df.to_csv(inv_out, index=False)
    print(f"  - {products_out} ({len(products_df)} rows)")
    print(f"  - {sales_out} ({len(best_sales_df)} rows, positive sales only)")
    print(f"  - {inv_out} ({len(best_inv_df)} rows, full SKU-day grid)")

    # Freeze final calibrated values in favorita_adapter.yaml
    final_calib = calib_history[-1]
    cfg["inventory_policy"]["buffer_factor"]["perishable"]["min"] = final_calib["buf_per_range"][0]
    cfg["inventory_policy"]["buffer_factor"]["perishable"]["max"] = final_calib["buf_per_range"][1]
    cfg["inventory_policy"]["buffer_factor"]["non_perishable"]["min"] = final_calib["buf_non_range"][0]
    cfg["inventory_policy"]["buffer_factor"]["non_perishable"]["max"] = final_calib["buf_non_range"][1]
    cfg["inventory_policy"]["miss_review_prob"] = final_calib["miss_prob"]
    cfg["inventory_policy"]["lead_time"]["late_delivery_prob"] = final_calib["late_prob"]
    cfg["calibrated_results"] = final_calib

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[Build] Frozen final calibrated parameters into {CONFIG_PATH}")

    # Save calibration log
    with open(SCRATCH_DIR / "calibration_log.json", "w") as f:
        json.dump(calib_history, f, indent=2)

    print(f"\n[Build] Favorita dataset generation complete in {time.time()-t0:.1f}s!")
    return calib_history


def simulate_inventory(
    selected_items: List[int],
    item_prices: Dict[int, Dict[str, Any]],
    op_dates: List[str],
    demand_matrix: np.ndarray,
    onpromo_matrix: np.ndarray,
    inv_cfg: Dict[str, Any],
    promo_cfg: Dict[str, Any],
    buf_per_range: Tuple[float, float],
    buf_non_range: Tuple[float, float],
    miss_prob: float,
    late_prob: float,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulates inventory operations day by day:
    - Independent periodic review cycles.
    - Orders based on trailing average demand from past days only.
    - FIFO batch tracking for perishables; unsold units expire on arrival + shelf_life.
    - batch_expiry_date = oldest active batch expiry for perishables, NaT for non-perishables.
    - Records supplier_lead_time_recorded only on delivery days.
    """
    n_items = len(selected_items)
    n_days = len(op_dates)

    # Pre-draw per-SKU static simulation attributes
    sku_params = {}
    init_cov_min = float(inv_cfg["initial_stock"]["days_of_cover_min"])
    init_cov_max = float(inv_cfg["initial_stock"]["days_of_cover_max"])

    for i, itm in enumerate(selected_items):
        info = item_prices[itm]
        is_per = info["is_perishable"]
        
        # Review period R
        if is_per:
            r_period = int(rng.randint(inv_cfg["review_period"]["perishable"]["min"], inv_cfg["review_period"]["perishable"]["max"] + 1))
            buf_factor = float(rng.uniform(buf_per_range[0], buf_per_range[1]))
            lt_mean = float(inv_cfg["lead_time"]["perishable"]["mean"])
            lt_std = float(inv_cfg["lead_time"]["perishable"]["std"])
            lt_min = float(inv_cfg["lead_time"]["perishable"]["min"])
            lt_max = float(inv_cfg["lead_time"]["perishable"]["max"])
        else:
            r_period = int(rng.randint(inv_cfg["review_period"]["non_perishable"]["min"], inv_cfg["review_period"]["non_perishable"]["max"] + 1))
            buf_factor = float(rng.uniform(buf_non_range[0], buf_non_range[1]))
            lt_mean = float(inv_cfg["lead_time"]["non_perishable"]["mean"])
            lt_std = float(inv_cfg["lead_time"]["non_perishable"]["std"])
            lt_min = float(inv_cfg["lead_time"]["non_perishable"]["min"])
            lt_max = float(inv_cfg["lead_time"]["non_perishable"]["max"])

        # Offset initial review day to stagger orders across SKUs
        review_offset = int(rng.randint(0, r_period))
        
        # Initial cover days
        init_cover = float(rng.uniform(init_cov_min, init_cov_max))
        # Initial mean demand estimate: first 14 days mean (positive floor 1.0)
        mean_init_d = max(1.0, float(np.mean(demand_matrix[i, :14])))
        starting_stock = int(round(mean_init_d * init_cover))

        # Pack size covers at most ~3 days of demand (min 1)
        mean_demand_all = max(1.0, float(np.mean(demand_matrix[i, :])))
        pack_size = max(1, int(round(mean_demand_all * float(inv_cfg["pack_size"]["max_days_cover"]))))

        sku_params[itm] = {
            "is_perishable": is_per,
            "shelf_life": info["shelf_life"],
            "r_period": r_period,
            "review_offset": review_offset,
            "buf_factor": buf_factor,
            "lt_mean": lt_mean,
            "lt_std": lt_std,
            "lt_min": lt_min,
            "lt_max": lt_max,
            "pack_size": pack_size,
            "current_stock": starting_stock,
            # FIFO batches: list of dicts: [{'units': int, 'expiry_day_idx': int, 'arrival_date': str}]
            "batches": [{"units": starting_stock, "expiry_day_idx": min(n_days + 100, int(info["shelf_life"])), "arrival_date": op_dates[0]}] if is_per else [],
            # Pipeline orders: list of dicts: [{'arrival_day_idx': int, 'units': int, 'lead_time': float}]
            "in_flight": []
        }

    # Tracking arrays for inventory.csv
    # ending_stock, sales_units, unfulfilled_demand, spoilage_units, supplier_lead_time_recorded, batch_expiry_date
    res_starting_stock = np.zeros((n_items, n_days), dtype=np.int32)
    res_ending_stock = np.zeros((n_items, n_days), dtype=np.int32)
    res_sales_units = np.zeros((n_items, n_days), dtype=np.int32)
    res_unfulfilled = np.zeros((n_items, n_days), dtype=np.int32)
    res_spoilage = np.zeros((n_items, n_days), dtype=np.int32)
    res_lead_time = np.full((n_items, n_days), np.nan, dtype=np.float32)
    res_expiry_date = [[None] * n_days for _ in range(n_items)]

    # Sales rows (positive sales only)
    sales_rows = []
    p_min = float(promo_cfg["discount_range"]["min"])
    p_max = float(promo_cfg["discount_range"]["max"])

    # Day-by-day continuous simulation
    for d_idx, d_str in enumerate(op_dates):
        d_cal = pd.to_datetime(d_str)

        for i, itm in enumerate(selected_items):
            p = sku_params[itm]
            is_per = p["is_perishable"]

            # 1. Deliveries arrival
            arrived_units = 0
            rec_lead_time = np.nan
            
            still_in_flight = []
            for order in p["in_flight"]:
                if order["arrival_day_idx"] <= d_idx:
                    arrived_units += order["units"]
                    rec_lead_time = order["lead_time"]
                else:
                    still_in_flight.append(order)
            p["in_flight"] = still_in_flight

            # 2. Add arrived units to stock and batches
            p["current_stock"] += arrived_units
            if is_per and arrived_units > 0:
                p["batches"].append({
                    "units": arrived_units,
                    "expiry_day_idx": d_idx + p["shelf_life"],
                    "arrival_date": d_str,
                })

            res_starting_stock[i, d_idx] = p["current_stock"]
            if not np.isnan(rec_lead_time):
                res_lead_time[i, d_idx] = rec_lead_time

            # 3. Demand fulfillment
            real_d = demand_matrix[i, d_idx]
            d_units = int(round(real_d))
            avail = p["current_stock"]

            sold = min(avail, d_units)
            unfulfilled = d_units - sold
            p["current_stock"] -= sold

            res_sales_units[i, d_idx] = sold
            res_unfulfilled[i, d_idx] = unfulfilled

            # Deduct from FIFO batches
            if is_per and sold > 0:
                remain_deduct = sold
                for b in p["batches"]:
                    if b["units"] <= remain_deduct:
                        remain_deduct -= b["units"]
                        b["units"] = 0
                    else:
                        b["units"] -= remain_deduct
                        remain_deduct = 0
                        break
                p["batches"] = [b for b in p["batches"] if b["units"] > 0]

            # 4. Spoilage check for perishables at end of day
            spoilage = 0
            if is_per:
                active_batches = []
                for b in p["batches"]:
                    if b["expiry_day_idx"] <= d_idx:
                        spoilage += b["units"]
                    else:
                        active_batches.append(b)
                p["batches"] = active_batches
                p["current_stock"] = max(0, p["current_stock"] - spoilage)

            res_spoilage[i, d_idx] = spoilage
            res_ending_stock[i, d_idx] = p["current_stock"]

            # 5. Oldest batch expiry date for perishables
            if is_per and len(p["batches"]) > 0:
                oldest_b = min(p["batches"], key=lambda x: x["expiry_day_idx"])
                days_ahead = oldest_b["expiry_day_idx"] - d_idx
                exp_dt = d_cal + pd.Timedelta(days=days_ahead)
                res_expiry_date[i][d_idx] = exp_dt.strftime("%Y-%m-%d")
            else:
                res_expiry_date[i][d_idx] = None

            # 6. Sales row emission (ONLY if units_sold > 0)
            if d_units > 0:
                info = item_prices[itm]
                onpromo = onpromo_matrix[i, d_idx]
                if onpromo > 0:
                    disc = round(float(rng.uniform(p_min, p_max)), 4)
                else:
                    disc = 0.0

                unit_price = round(info["selling_price"] * (1.0 - disc), 2)
                total_amt = round(unit_price * d_units, 2)
                
                # Unique basket_id and transaction_id per row
                txn_id = f"TXN-{itm}-{d_str.replace('-', '')}"
                bsk_id = f"BSK-{itm}-{d_str.replace('-', '')}"

                sales_rows.append({
                    "transaction_id": txn_id,
                    "date": d_str,
                    "sku_id": info["sku_id"],
                    "units_sold": d_units,
                    "unit_price": unit_price,
                    "total_amount": total_amt,
                    "discount_pct": disc,
                    "basket_id": bsk_id,
                    "timestamp": f"{d_str} 14:30:00",
                })

            # 7. Replenishment Review & Ordering
            # Trailing demand lookback: past 28 days (PAST ONLY, up to d_idx inclusive)
            # Periodic review on (d_idx + review_offset) % r_period == 0
            is_review_day = ((d_idx + p["review_offset"]) % p["r_period"]) == 0
            # Stochastic missed review
            if is_review_day and (rng.uniform() >= miss_prob):
                lookback_start = max(0, d_idx - int(inv_cfg["trailing_demand_window_days"]) + 1)
                past_sales = demand_matrix[i, lookback_start : d_idx + 1]
                trailing_avg_d = max(0.5, float(np.mean(past_sales)))

                # Order-up-to target S = trailing_avg_d * (R + L) * buffer_factor
                effective_lead_time = p["lt_mean"]
                order_up_to = trailing_avg_d * (p["r_period"] + effective_lead_time) * p["buf_factor"]

                # Inventory position = on_hand + in_flight
                in_flight_units = sum(o["units"] for o in p["in_flight"])
                inv_position = p["current_stock"] + in_flight_units

                shortfall = order_up_to - inv_position
                if shortfall > 0:
                    # Round to pack size
                    order_qty = int(np.ceil(shortfall / p["pack_size"]) * p["pack_size"])
                    if order_qty > 0:
                        # Draw lead time
                        drawn_lt = float(rng.normal(p["lt_mean"], p["lt_std"]))
                        drawn_lt = float(np.clip(drawn_lt, p["lt_min"], p["lt_max"]))
                        # Late delivery shock
                        if rng.uniform() < late_prob:
                            drawn_lt += float(inv_cfg["lead_time"].get("late_delivery_delay_days", 1))

                        arrival_idx = d_idx + max(1, int(round(drawn_lt)))
                        p["in_flight"].append({
                            "arrival_day_idx": arrival_idx,
                            "units": order_qty,
                            "lead_time": round(drawn_lt, 1),
                        })

    # Compile inventory dataframe
    inv_records = []
    for i, itm in enumerate(selected_items):
        sku_id = item_prices[itm]["sku_id"]
        for d_idx, d_str in enumerate(op_dates):
            inv_records.append({
                "date": d_str,
                "sku_id": sku_id,
                "starting_stock": res_starting_stock[i, d_idx],
                "ending_stock": res_ending_stock[i, d_idx],
                "sales_units": res_sales_units[i, d_idx],
                "unfulfilled_demand": res_unfulfilled[i, d_idx],
                "spoilage_units": res_spoilage[i, d_idx],
                "supplier_lead_time_recorded": res_lead_time[i, d_idx],
                "batch_expiry_date": res_expiry_date[i][d_idx],
            })

    inv_df = pd.DataFrame(inv_records)
    inv_df["date"] = pd.to_datetime(inv_df["date"])
    inv_df["batch_expiry_date"] = pd.to_datetime(inv_df["batch_expiry_date"])
    sales_df = pd.DataFrame(sales_rows)
    sales_df["date"] = pd.to_datetime(sales_df["date"])

    return inv_df, sales_df


if __name__ == "__main__":
    build_favorita_dataset()

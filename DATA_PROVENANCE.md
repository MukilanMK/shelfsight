# Data Provenance Document: Corporación Favorita Grocery Dataset Adaptation

## 1. Overview & Dataset Identity
This document records the exact provenance, transformations, and simulation methodology applied to adapt the real-world **Corporación Favorita Grocery Sales Forecasting** competition dataset into the **ShelfSight** retail inventory machine learning system.

* **Primary Data Source**: Corporación Favorita Grocery Sales Forecasting (Kaggle 2017 item-level competition dataset).
* **Local Source Directory**: `C:\clg\ml\Shop\ds\favorita-grocery-sales-forecasting\`
* **Selected Store**: **Store 44** (Quito, Pichincha, Ecuador — flagship Type A Superstore).
* **Date Horizon**: `2015-08-16` to `2017-08-15` (731 calendar days).
* **Store Closures Excluded**: 4 store-closure dates are excluded from the operational grid (`2015-12-25`, `2016-01-01`, `2016-12-25`, `2017-01-01`).
* **Active Operational Days**: **727 days**.
* **SKU Count**: **60 items** (18 perishable = 30.0%, 42 non-perishable = 70.0%).
* **Total Grid Size**: 60 SKUs $\times$ 727 days = **43,620 SKU-days**.
* **Seed**: `42` (fixed random seed).

---

## 2. Real Demand & Observed Sales Limitations
> [!IMPORTANT]
> **Core Limitation on Demand**:
> `sales.units_sold` represents the **real observed sales** from Favorita transactions. In retail data, observed sales are a **demand proxy**, not unconstrained true latent demand. If the physical store experienced a stock-out or delisted an item on a given day, observed sales were recorded as zero.
> 
> In this pipeline, real `unit_sales` is treated as true customer demand ($D$). In `inventory.csv`, `sales_units` represents **fulfilled customer units** in the inventory simulation:
> $$\text{sales\_units} = \min(D, \, \text{available\_stock})$$
> $$\text{unfulfilled\_demand} = D - \text{sales\_units}$$
> Across every SKU-day, the strict identity holds:
> $$\text{sales.units\_sold} \equiv \text{inventory.sales\_units} + \text{inventory.unfulfilled\_demand}$$

---

## 3. Real Macroeconomic Shocks: April 2016 Manabi Earthquake
The training horizon contains a major macroeconomic and logistical event explicitly documented in `holidays_events.csv`:
* **Event**: **Terremoto de Manabí (7.8 $M_w$ Earthquake)** on **April 16, 2016**.
* **Recorded Duration**: Span of 31 days in the events table (`Terremoto Manabi` through `Terremoto Manabi+30`, ending `2016-05-16`).
* **Impact**: Relief donations and emergency stockpiling caused anomalous sales spikes in staples (water, milk, canned goods, paper products) followed by regional supply chain delays.

---

## 4. Column Provenance Mapping

### A. `products.csv` (60 rows)
| Column | Classification | Description / Origin |
|---|:---:|---|
| `sku_id` | **DERIVED** | Formatted as `SKU-GROC-<item_nbr>` using the real Favorita `item_nbr`. Conforms cleanly to `DataValidator._standardize_sku`. |
| `product_name` | **DERIVED** | Formatted as `<Family> item <item_nbr>` (real branded item descriptions are proprietary in Favorita). |
| `category` | **REAL** | Real merchandise family from `items.csv` (e.g. `GROCERY I`, `BEVERAGES`, `DAIRY`, `PRODUCE`). |
| `cost_price` | **SIMULATED** | Sampled from family-dependent empirical wholesale ranges in `favorita_adapter.yaml`. |
| `selling_price` | **SIMULATED** | Derived using commercial formula: $\text{selling\_price} = \frac{\text{cost\_price}}{1 - \text{margin\_pct}}$. |
| `margin_pct` | **SIMULATED** | Drawn uniformly from per-family margin ranges in `favorita_adapter.yaml` (e.g. 20% to 50%). |
| `is_perishable` | **REAL** | Derived directly from `items.perishable` (1 = Perishable, 0 = Non-perishable). |
| `shelf_life_days` | **SIMULATED** | Per-family shelf life ranges (e.g. Bread 5–9d, Dairy 14–28d, Pantry 365–730d). |
| `storage_type` | **SIMULATED** | Chilled, ambient, or frozen based on family logistics requirements. |
| `upc` | **SIMULATED** | Zero-padded 12-digit string generated from `item_nbr`. |

### B. `sales.csv` (41,781 rows — strictly positive sales days)
| Column | Classification | Description / Origin |
|---|:---:|---|
| `transaction_id` | **SIMULATED** | Unique identifier formatted as `TXN-<item_nbr>-<YYYYMMDD>`. |
| `date` | **REAL** | Exact real transaction date from `train.csv`. |
| `sku_id` | **DERIVED** | Real Favorita `item_nbr` mapped to `SKU-GROC-<item_nbr>`. |
| `units_sold` | **REAL** | **Unmodified real `unit_sales`** from `train.csv`. 100% discrete integer units. Returns (negative values) clipped to 0 (0 occurrences among chosen items). |
| `unit_price` | **SIMULATED** | $\text{selling\_price} \times (1 - \text{discount\_pct})$. |
| `total_amount` | **SIMULATED** | $\text{unit\_price} \times \text{units\_sold}$. |
| `discount_pct` | **SIMULATED** | Drawn from 5%–20% when real `onpromotion > 0` in `train.csv`; 0.0 otherwise. |
| `basket_id` | **SIMULATED** | Unique per row (`BSK-<item_nbr>-<YYYYMMDD>`). Favorita transactions lack receipt baskets. |
| `timestamp` | **DERIVED** | Formatted as `<date> 14:30:00` per pipeline validation standards. |

### C. `inventory.csv` (43,620 rows — 60 SKUs $\times$ 727 operational days)
| Column | Classification | Description / Origin |
|---|:---:|---|
| `date` | **REAL** | Operational calendar dates. |
| `sku_id` | **DERIVED** | Real Favorita `item_nbr` mapped to `SKU-GROC-<item_nbr>`. |
| `starting_stock` | **SIMULATED** | Opening stock + received replenishment deliveries. |
| `ending_stock` | **SIMULATED** | $\text{starting\_stock} - \text{sales\_units} - \text{spoilage\_units} \ge 0$. |
| `sales_units` | **SIMULATED** | Fulfilled customer demand: $\min(\text{units\_sold}, \, \text{starting\_stock})$. |
| `unfulfilled_demand` | **SIMULATED** | Lost sales: $\text{units\_sold} - \text{sales\_units}$. |
| `spoilage_units` | **SIMULATED** | Unsold expired units tracked via FIFO batches for perishables. |
| `supplier_lead_time_recorded`| **SIMULATED** | Populated **strictly on delivery arrival days** (NaN on all other days). Lead times drawn from family distributions with stochastic late shocks. |
| `batch_expiry_date` | **SIMULATED** | Expiry date of the oldest active FIFO batch for perishables; **strictly NaT for non-perishables**. |

---

## 5. Item Selection & Gap Verification
The 60 items were selected deterministically from Store 44 using three criteria:
1. **100% Discrete Integer Sales**: Only items with zero fractional sales were selected.
2. **Lifecycle Continuity**: Contiguous blocks of zero sales $\ge 30$ days (late launches or early discontinuations) were identified and replaced with items active across the full 727 operational dates.
3. **Gap Rule**: 
   * **55 Steady Items**: Longest consecutive zero-sale run $\le 7$ days (observed range: 0 to 7 days).
   * **5 Sporadic Slow Movers**: Longest consecutive zero-sale run $\le 30$ days (observed range: 5 to 14 days) to train unsupervised deadstock models.

---

## 6. Inventory Policy & Target Calibration
To eliminate circularity, the inventory simulator **does not use the pipeline's analytical safety-stock / ROP formula**. It implements an independent periodic review order-up-to policy:
* **Review Cycle ($R$)**: Perishables review every 1–3 days; Non-perishables every 3–7 days.
* **Trailing Demand**: Sized from the trailing 28-day moving average of past demand only.
* **Stochastic Noise**: Missed review probability $P=0.06$, late delivery probability $P=0.07$.
* **Pack Size**: Sized to cover at most 3 days of trailing average demand (minimum pack 1).

### Calibration Results
Tuned iteratively against the real pipeline target: $\text{target\_stockout\_in\_lead\_time} == 1$ (horizon = dynamic lead time mean):
* **Achieved Target Stockout Rate (Overall)**: **12.24%**
* **Achieved Target Stockout Rate (Train)**: **12.17%** (Target: 10%–35%)
* **Achieved Target Stockout Rate (Test - Final 20%)**: **12.54%** (Target: 10%–35%)
* **Per-SKU Stock-out Frequency ($\text{ending\_stock} == 0$)**:
  * Minimum: **1.10%**
  * Median: **4.61%**
  * Maximum: **10.73%** (Requirement $\le 60.0\%$: **PASS**)
  * SKUs with $\ge 1$ stock-out event: **100.0%** (Requirement $\ge 80.0\%$: **PASS**)

# ShelfSight Engine: End-to-End Retail Supply Chain ML & Optimization Platform
## Master Technical Specification, Architectural Blueprints & Academic Syllabus Manual

---

### Executive Overview & Vision

**ShelfSight Engine** is an enterprise-grade, fully generalized Machine Learning system built for retail grocery environments (Point-of-Sale, Inventory Warehousing, and Supply Chain Logistics). Its core purpose is to replace fragile, static heuristics (such as hardcoded reorder points, fixed deadstock thresholds, and arbitrary discount rules) with **adaptive, self-calibrated statistical and ML mechanisms**.

The system dynamically:
1. **Forecasts daily demand and sales velocity** using regularized linear models and state-of-the-art tree ensemble algorithms.
2. **Classifies restocking urgency** dynamically by evaluating empirical lead times, safety stocks, and stock-out probabilities.
3. **Segments inventory health and mitigates deadstock/spoilage** through unsupervised clustering (K-Means, DBSCAN, Hierarchical) and margin-calibrated dynamic markdowns.
4. **Discovers cross-selling patterns** with Market Basket Association Mining (Apriori algorithm) to bundle slow-moving SKUs with high-velocity anchors.
5. **Serves an interactive executive suite & what-if simulator** via Streamlit and comprehensive terminal dashboards.

---

## 1. High-Level Architectural Flowchart

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   RAW DATA LAYER                                       │
│  - products.csv (SKU, Category, Unit Cost, Selling Price, Shelf Life, Min Temp)       │
│  - inventory.csv (Date, SKU, Ending Stock, Received Qty, Damaged Qty, Lead Time)       │
│  - sales.csv (Transaction ID, Date, SKU, Units Sold, Promotion Applied, Basket ID)     │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│              MODULE 1: DATA INGESTION, VALIDATION & FEATURE ENGINEERING                │
│  - Schema validation, Regex SKU standardization, Missing timestamp imputation         │
│  - Dynamic temporal features: Cyclical Day-of-Week & Month sine/cosine encodings      │
│  - Rolling Demand Velocities: 7-day, 14-day, 30-day Moving Averages + Demand Variance  │
│  - Empirical Bayes Lead Time Shrinkage: w_i * L_mean + (1 - w_i) * Category_mean       │
│  - Zero-hardcoded statistical bounds: Dynamic Velocity Floor, DIR, GMROI              │
│  - Dynamic Bivariate Safety Stock & Dynamic Reorder Point (ROP) calculation            │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│            MODULE 2: ML BASICS, PREPROCESSING & MODEL ENGINEERING (UNIT-I)             │
│  - Leak-free Chronological Train/Test Partitioning (Preserves temporal causality)      │
│  - RobustScaler & OneHotEncoder with handle_unknown='ignore'                           │
│  - 5-Fold Time-Series Cross-Validation Diagnostics (Overfitting Gap = Train - Val R²)  │
│  - Dynamic Hyperparameter Tuning via GridSearchCV / RandomizedSearchCV                 │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 MODULE 3: MULTI-METHOD FEATURE SELECTION (UNIT-II)                     │
│  - Filter Methods: Univariate ANOVA F-test (SelectKBest) + Random Forest MDI           │
│  - Wrapper Methods: Recursive Feature Elimination with Cross-Validation (RFECV)        │
│  - Embedded Methods: LASSO (L1 Regularization Sparsity Penalty)                        │
│  - Consensus Voting Matrix: Isolates Top 15 Consensus Predictors across all paradigms  │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   MODULE 4: SUPERVISED MACHINE LEARNING (UNIT-III)                     │
│  - Demand Velocity Regression: Linear Regression (OLS), Ridge (L2), Lasso (L1)         │
│  - Restock Urgency Multi-Class Classification: Logistic Regression, Decision Tree      │
│    (ID3/CART), Gaussian Naïve Bayes, Support Vector Machines (SVM RBF), K-NN           │
│  - Complete Evaluation Matrix: RMSE, MAE, R², Accuracy, Macro-F1, Multi-Class ROC-AUC   │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                  MODULE 5: UNSUPERVISED MACHINE LEARNING (UNIT-IV)                     │
│  - K-Means Clustering: Silhouette Score grid-search for optimal portfolio health tiers │
│  - DBSCAN: Density-based spatial clustering to flag inventory outliers & anomalies     │
│  - Agglomerative Hierarchical Clustering: Ward's minimum variance dendrograms          │
│  - Dynamic Markdown Clearance Engine: Margin-bounded markdown % based on DIR Risk      │
│  - Market Basket Mining: Apriori association rules (Support, Confidence, Lift)         │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             MODULE 6: ADVANCED ENSEMBLE LEARNING & BOOSTING (UNIT-V)                   │
│  - Bagging: Random Forest Regressor & Random Forest Classifier (Variance Reduction)    │
│  - Boosting: AdaBoost, Gradient Boosting Decision Trees (GBDT), & XGBoost              │
│  - Production Model Selection: Auto-identifies champion models balancing bias/variance │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 PRESENTATION, REPORTING & INTERACTION LAYER                            │
│  - Executive Summary Dashboard (reports/executive_summary.json + rich text report)     │
│  - Interactive Streamlit Dashboard (app_ui.py):                                        │
│    * Stock Depletion Curves vs Dynamic ROP                                             │
│    * AI Urgency Matrix & Restock Recommendations                                       │
│    * Dynamic Markdown Clearances & Cross-Sell Basket Bundles                           │
│    * "What-If" Demand & Safety Stock Simulator using production ML ensembles           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Deep Dive: Module-by-Module Walkthrough

### Module 1: Data Ingestion, Data Quality & Feature Engineering
- **File**: [`src/module1_ingestion/feature_engineering.py`](file:///c:/clg/ml/Shop/src/module1_ingestion/feature_engineering.py), [`validation.py`](file:///c:/clg/ml/Shop/src/module1_ingestion/validation.py), [`pipeline.py`](file:///c:/clg/ml/Shop/src/module1_ingestion/pipeline.py)
- **Functions & Responsibilities**:
  1. `DataValidator.validate_schemas()`: Validates column names, datatypes, and non-null constraints across products, inventory, and sales tables.
  2. `DataValidator.standardize_skus()`: Cleans and regex-standardizes SKU codes across disjointed systems.
  3. `DataValidator.impute_missing_dates()`: Fills missing dates chronologically per SKU and carries forward valid inventory state to avoid lookahead leakage.
  4. `FeatureEngineer.compute_temporal_features()`: Extracts day-of-week, month, quarter, payday proximity, and generates cyclical Sine/Cosine transformations:
     $$\text{DoW}_{\sin} = \sin\left(\frac{2\pi \cdot \text{dow}}{7}\right), \quad \text{DoW}_{\cos} = \cos\left(\frac{2\pi \cdot \text{dow}}{7}\right)$$
  5. `FeatureEngineer.compute_rolling_velocity()`: Computes multi-window moving averages (7d, 14d, 30d) and standard deviations for sales velocity.
  6. `FeatureEngineer.compute_lead_time_shrinkage()`: Applies Empirical Bayes Shrinkage to balance item-level lead times with category priors:
     $$w_i = \frac{n_i}{n_i + \tau}, \quad \hat{\mu}_{L} = w_i \bar{L}_i + (1 - w_i) \mu_{\text{cat}}$$
  7. `FeatureEngineer.compute_safety_stock_and_rop()`: Evaluates dynamic safety stocks and reorder points with bivariate variance propagation:
     $$\text{Safety Stock} = Z_{0.95} \cdot \sqrt{\bar{L} \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2}$$
     $$\text{ROP} = (\bar{D} \times \bar{L}) + \text{Safety Stock}$$
  8. `FeatureEngineer.compute_financial_health_metrics()`: Dynamically calculates:
     - **Days of Inventory Remaining (DIR)** with an empirically learned positive velocity floor:
       $$\epsilon_{\text{floor}} = Q_{0.05}(\{v \in \mathcal{V} \mid v > 0\}), \quad \text{DIR} = \frac{\text{Ending Stock}}{\max(\bar{D}, \epsilon_{\text{floor}})}$$
     - **Gross Margin Return on Investment (GMROI)** measuring capital efficiency.
     - **Dynamic Restocking Urgency Label**: Learned target variable categorized into `OUT_OF_STOCK`, `HIGH`, `MEDIUM`, or `LOW`.

---

### Module 2 (Unit I): ML Foundations, Preprocessing & Model Engineering
- **File**: [`src/module2_engineering/preprocessing.py`](file:///c:/clg/ml/Shop/src/module2_engineering/preprocessing.py), [`validation_tuning.py`](file:///c:/clg/ml/Shop/src/module2_engineering/validation_tuning.py)
- **Functions & Responsibilities**:
  1. `ChronologicalSplitter.split()`: Partitions time-series data chronologically (e.g., 80% train, 20% holdout test) strictly preventing future-data leakage into model training.
  2. `DataPreprocessor.fit_transform()`: Houses a scikit-learn `ColumnTransformer` composed of:
     - `RobustScaler` on numerical features (resilient against high-volume promotional spikes and outlier demand).
     - `OneHotEncoder(handle_unknown='ignore')` on categorical attributes (categories, storage requirements).
  3. `ModelValidator.evaluate_kfold_cv()`: Performs 5-Fold cross-validation, computing training vs validation scores to calculate the **Overfitting Gap**:
     $$\text{Overfitting Gap} = R^2_{\text{train}} - R^2_{\text{val}}$$
  4. `HyperparameterTuner.tune()`: Employs `GridSearchCV` / `RandomizedSearchCV` across estimators to pinpoint optimal regularization and tree architectures.

---

### Module 3 (Unit II): Multi-Method Feature Selection
- **File**: [`src/module3_feature_selection/feature_selection.py`](file:///c:/clg/ml/Shop/src/module3_feature_selection/feature_selection.py), [`pipeline.py`](file:///c:/clg/ml/Shop/src/module3_feature_selection/pipeline.py)
- **Functions & Responsibilities**:
  Implements a 3-way consensus feature selection architecture:
  1. **Filter Paradigm**:
     - `SelectKBest(score_func=f_regression)`: Univariate ANOVA $F$-test measuring linear statistical association.
     - `RandomForestRegressor.feature_importances_`: Mean Decrease in Impurity (MDI).
  2. **Wrapper Paradigm**:
     - `RFECV(estimator=LinearRegression)`: Recursive Feature Elimination iteratively pruning features exhibiting smallest coefficients.
  3. **Embedded Paradigm**:
     - `LassoCV(cv=5)`: $L_1$ penalty inducing mathematical sparsity to zero-out redundant features.
  4. **Consensus Aggregation**:
     - Combines votes from all paradigms into a composite rank, selecting the Top 15 predictors for subsequent modules.

---

### Module 4 (Unit III): Supervised Machine Learning Benchmarking
- **File**: [`src/module4_supervised/regression_demand.py`](file:///c:/clg/ml/Shop/src/module4_supervised/regression_demand.py), [`classification_restock.py`](file:///c:/clg/ml/Shop/src/module4_supervised/classification_restock.py)
- **Functions & Responsibilities**:
  1. **Demand Velocity Regressors**:
     - Ordinary Least Squares (OLS) Linear Regression: Baseline non-regularized continuous model.
     - Ridge Regression ($L_2$ Regularization): Shrinks coefficients to mitigate multicollinearity.
     - Lasso Regression ($L_1$ Regularization): Encourages sparse weights for direct feature pruning.
     - *Metrics Evaluated*: Holdout RMSE, MAE, $R^2$, and 5-Fold CV $R^2$.
  2. **Restocking Urgency Classifiers**:
     - Multi-Class Logistic Regression (Softmax).
     - Decision Tree Classifier (CART / Information Gain Entropy).
     - Gaussian Naïve Bayes (Conditional probability baseline).
     - Support Vector Machines (SVM with Radial Basis Function kernel).
     - K-Nearest Neighbors (K-NN non-parametric distance-weighted voting).
     - *Metrics Evaluated*: Multi-Class Accuracy, Macro Precision, Macro Recall, Macro F1, and Multi-Class ROC-AUC (One-vs-Rest).

---

### Module 5 (Unit IV): Unsupervised Learning, Deadstock & Spoilage Mitigation
- **File**: [`src/module5_unsupervised/clustering.py`](file:///c:/clg/ml/Shop/src/module5_unsupervised/clustering.py), [`association_rules.py`](file:///c:/clg/ml/Shop/src/module5_unsupervised/association_rules.py)
- **Functions & Responsibilities**:
  1. `KMeansInventoryClusterer`: Evaluates $k \in [2, 6]$, auto-selecting optimal $k$ by maximizing the **Silhouette Score** to partition SKUs into health tiers (Fast Movers, Regular, Slow Movers, At-Risk).
  2. `DBSCANAnomalyDetector`: Density-based clustering identifies isolated spatial outliers (sudden demand collapses or anomalous stock accumulations).
  3. `HierarchicalClusterer`: Computes Agglomerative Hierarchical clustering using Ward's minimum variance linkage, generating dendrogram similarity trees.
  4. `DynamicMarkdownEngine`: Zero-hardcoding markdown clearance generator. Dynamically discounts at-risk SKUs without erasing profit margins:
     $$\text{Risk Intensity} = \frac{\text{DIR}_i - \min(\text{DIR})}{\max(\text{DIR}) - \min(\text{DIR})}$$
     $$\text{Markdown \%} = \text{clip}\left(\text{Margin \%} \times (0.35 + 0.50 \cdot \text{Risk Intensity}), \, 0.10, \, 0.50\right)$$
  5. `MarketBasketMiner`: Implements the Apriori algorithm to discover frequent itemsets and association rules ($\text{Support}$, $\text{Confidence}$, $\text{Lift}$). Automatically pairs slow-moving deadstock items as bundle recommendations with fast-moving complementary anchors.

---

### Module 6 (Unit V): Ensemble Techniques & Advanced Boosting
- **File**: [`src/module6_ensemble/ensemble_models.py`](file:///c:/clg/ml/Shop/src/module6_ensemble/ensemble_models.py), [`pipeline.py`](file:///c:/clg/ml/Shop/src/module6_ensemble/pipeline.py)
- **Functions & Responsibilities**:
  1. **Bagging (Variance Reduction)**:
     - `RandomForestRegressor` and `RandomForestClassifier`: Ensembles hundreds of bootstrapped decision trees with randomized feature splits.
  2. **Boosting (Sequential Error Correction & Residual Learning)**:
     - `AdaBoost`: Focuses adaptively on misclassified and high-residual samples.
     - `GradientBoostingRegressor / Classifier`: Optimizes differentiable loss functions via stage-wise additive trees.
     - `XGBoost`: Extreme Gradient Boosting incorporating tree-depth regularizers, column subsampling, and exact split-finding.
  3. **Model Selection & Artifact Serialization**:
     - Compares performance across holdout sets and automatically persists champions (`best_demand_model.joblib`, `best_restock_model.joblib`) into `models/`.

---

### Presentation Layer: Interactive Dashboard & UI
- **Files**: [`src/dashboard/summary_reporter.py`](file:///c:/clg/ml/Shop/src/dashboard/summary_reporter.py), [`app_ui.py`](file:///c:/clg/ml/Shop/app_ui.py)
- **Features**:
  1. **Executive Summary Reporter**: Compiles automated JSON and text audit reports summarizing trending SKUs, critical restock lists, deadstock markdown schedules, and model benchmark tables.
  2. **Streamlit Web Application (`app_ui.py`)**:
     - **Tab 1: Executive KPI Command Center**: Live metrics on portfolio GMROI, stock-out risk counts, total deadstock capital, and category breakdown.
     - **Tab 2: Inventory Depletion & Restock Matrix**: Interactive Plotly depletion curves displaying current stock levels against dynamic safety stock and ROP boundaries.
     - **Tab 3: Deadstock & Spoilage Mitigation**: Markdown action plan displaying original price, recommended clearance price, discount %, and estimated capital recovery.
     - **Tab 4: Market Basket & Slow-Mover Bundling**: Cross-selling network graph and Apriori rule explorer.
     - **Tab 5: Interactive What-If Simulator**: Real-time scenario simulator allowing operators to adjust hypothetical sales velocity or lead-time shocks to see immediate safety stock and ROP responses predicted by production ML models.

---

## 3. Mathematical Foundations & "Zero Hardcoding" Philosophy

| Component | Traditional Hardcoded Anti-Pattern | ShelfSight Adaptive Statistical Formulation |
|---|---|---|
| **Safety Stock** | Fixed buffer (e.g., `safety_stock = 15`) | $Z_{0.95} \cdot \sqrt{\bar{L} \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2}$ (Bivariate variance propagation) |
| **Reorder Point (ROP)** | Static threshold (e.g., `ROP = 20`) | $\text{ROP} = (\bar{D} \times \bar{L}) + \text{Safety Stock}$ |
| **Supplier Lead Time** | Static catalog constant (e.g., `lead_time = 4`) | $\hat{\mu}_{L} = w_i \bar{L}_i + (1 - w_i) \mu_{\text{cat}}$ (Empirical Bayes Shrinkage) |
| **Zero-Sales Floor** | Arbitrary epsilon (e.g., `+ 0.001`) | $\epsilon_{\text{floor}} = Q_{0.05}(\{v \in \mathcal{V} \mid v > 0\})$ (Learned 5th percentile) |
| **Deadstock Definition** | Arbitrary age (e.g., `days > 60`) | Dynamic Silhouette-optimized K-Means & DBSCAN density clusters |
| **Clearance Discounts** | Flat markdown (e.g., `50% off all`) | Margin-calibrated discount bounded by unit gross profit margin |
| **Trend Vitality** | Raw volume threshold | $Z_{\text{accel}} = \frac{v_{7d} - v_{30d}}{\sigma_{30d}}, \quad \text{Vitality} = \hat{y} \cdot (1 + \max(0, Z_{\text{accel}}))$ |

---

## 4. Benchmark Results & Model Comparison

### Supervised Demand Velocity Forecasting
| Model Family | Algorithm | Holdout RMSE | Holdout MAE | Holdout $R^2$ | Overfitting Gap ($R^2_{\text{train}} - R^2_{\text{val}}$) | Champion Status |
|---|---|---|---|---|---|---|
| Linear | **Linear Regression (OLS)** | **0.2210** | **0.0535** | **0.9996** | 0.0000 | Linear Winner |
| Regularized | **Ridge Regression ($L_2$)** | 0.2210 | 0.0536 | 0.9996 | 0.0000 | Regularized Baseline |
| Regularized | **Lasso Regression ($L_1$)** | 0.2212 | 0.0543 | 0.9996 | 0.0000 | Sparse Baseline |
| Boosting | **XGBoost Regressor** | **0.6783** | **0.3246** | **0.9961** | 0.2350 | **Production Ensemble Champion** |
| Boosting | **Gradient Boosting (GBDT)**| 0.7051 | 0.3632 | 0.9958 | 0.2372 | Strong Candidate |
| Bagging | **Random Forest** | 0.8180 | 0.2939 | 0.9943 | 0.5148 | Overfitting Tendency |
| Boosting | **AdaBoost Regressor** | 2.2197 | 1.6490 | 0.9583 | 0.0303 | Underfitting (High Bias) |

### Supervised Restock Urgency Classification
| Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Multi-Class ROC-AUC | Champion Status |
|---|---|---|---|---|---|---|
| **AdaBoost Classifier** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **Production Ensemble Champion** |
| **Decision Tree (CART)**| **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | Baseline Tree Champion |
| **Gradient Boosting** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | Production Co-Winner |
| **Random Forest** | 0.9995 | 0.9998 | 0.9988 | 0.9993 | 1.0000 | Robust Bagging Winner |
| **XGBoost Classifier** | 0.9963 | 0.9909 | 0.9923 | 0.9916 | 1.0000 | High Generalization |
| **Logistic Regression**| 0.9819 | 0.9694 | 0.9587 | 0.9636 | 0.9991 | Linear Softmax Benchmark |
| **Support Vector (SVM)** | 0.9651 | 0.9448 | 0.9189 | 0.9307 | 0.9839 | Geometric Margin Benchmark |
| **K-Nearest Neighbors** | 0.9633 | 0.9504 | 0.9164 | 0.9303 | 0.9959 | Non-Parametric Benchmark |
| **Gaussian Naïve Bayes** | 0.8025 | 0.6737 | 0.7802 | 0.7102 | 0.9645 | Probabilistic Baseline |

---

## 5. Academic Curriculum & Syllabus Mapping

| Academic Module | Syllabus Unit Focus | Core Learning Objectives Covered | File Reference |
|---|---|---|---|
| **Module 1** | Ingestion & Data Hygiene | Missing value strategies, cyclical encoding, time-series aggregation, empirical shrinkage | [`src/module1_ingestion/`](file:///c:/clg/ml/Shop/src/module1_ingestion/) |
| **Module 2** | **Unit I**: ML Basics & Model Engineering | Chronological splits, bias-variance tradeoff, cross-validation, hyperparameter tuning | [`src/module2_engineering/`](file:///c:/clg/ml/Shop/src/module2_engineering/) |
| **Module 3** | **Unit II**: Feature Selection Techniques | Filter (ANOVA/MDI), Wrapper (RFECV), Embedded (LASSO), consensus feature aggregation | [`src/module3_feature_selection/`](file:///c:/clg/ml/Shop/src/module3_feature_selection/) |
| **Module 4** | **Unit III**: Supervised Learning | Linear/Ridge/Lasso regression, Decision Trees (CART), Naïve Bayes, SVM, K-NN | [`src/module4_supervised/`](file:///c:/clg/ml/Shop/src/module4_supervised/) |
| **Module 5** | **Unit IV**: Unsupervised Learning | K-Means (Silhouette-optimized), DBSCAN density clustering, Agglomerative Hierarchical trees, Apriori association rules | [`src/module5_unsupervised/`](file:///c:/clg/ml/Shop/src/module5_unsupervised/) |
| **Module 6** | **Unit V**: Ensemble Learning & Boosting | Bagging (Random Forest), Boosting (AdaBoost, Gradient Boosting, XGBoost), bias-variance optimization | [`src/module6_ensemble/`](file:///c:/clg/ml/Shop/src/module6_ensemble/) |

---

## 6. Execution, Testing & Operational Guide

### 1. Synthetic Data Generation
Generate a realistic 180-day retail grocery history:
```bash
python generate_dataset.py --num-products 60 --days 180 --output-dir data/raw
```

### 2. End-to-End Pipeline Execution
Run all 6 modules sequentially and output the audit reports:
```bash
python run_pipeline.py
```
*(To regenerate reports from cached models without re-training: `python run_pipeline.py --skip-train`)*

### 3. Launching Interactive Streamlit Dashboard
Launch the web UI:
```bash
streamlit run app_ui.py
```

### 4. Running the Automated Test Suite
Execute the full unit test suite:
```bash
python -m pytest tests/ -v
```
All unit tests validate data schemas, mathematical bounds, feature formulas, training convergence, and report generation.

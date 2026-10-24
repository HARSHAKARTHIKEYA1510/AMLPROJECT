# Processed Data Directory (`data/processed/`)

This directory contains the cleaned, standardized, and feature-engineered datasets produced by [`src/data_cleaning.py`](../../src/data_cleaning.py).

## Available Files:

| File Name | Format | Size | On GitHub? | Description & Usage |
| :--- | :--- | :--- | :--- | :--- |
| **`cleaned_calendar.csv`** | CSV | 201 KB | ✅ Yes | Daily calendar with cyclical trigonometric features, SNAP flags, Christmas markers |
| **`sku_lifecycle_metadata.csv`** | CSV | 2.5 MB | ✅ Yes | Product release dates, active shelf days, and **cold-start indicators** (`is_cold_start`) |
| **`m5_starter_modeling_panel.csv`** | CSV | 18.9 MB | ✅ Yes | **Recommended starter modeling panel** (CA_1 `FOODS_1`, 50 SKUs, 95.6K rows) with complete dates, causal lags ($t-1, 7, 14, 28$), rolling stats, and **stockout flags** |
| **`cleaned_sales_sample.csv`** | CSV | 1.9 MB | ✅ Yes | 500-row sample of the cleaned sales matrix for instant CSV inspection |
| **`cleaned_prices_sample.csv`** | CSV | 229 KB | ✅ Yes | 5,000-row sample of the cleaned prices table for instant CSV inspection |
| **`cleaned_sales_matrix.parquet`** | Parquet | 31.0 MB | ✅ Yes | Complete downcasted 30,490 x 1,919 unit sales matrix (compressed from 114 MB CSV) |
| **`cleaned_prices.parquet`** | Parquet | 3.2 MB | ✅ Yes | Complete 6.84M price records with regular price baselines & discounts (compressed from 307 MB CSV) |
| **`cleaned_sales_matrix.csv`** | CSV | 114 MB | ⚙️ Local | Full clean sales matrix in raw CSV (local only due to GitHub 100 MB limit) |
| **`cleaned_prices.csv`** | CSV | 307 MB | ⚙️ Local | Full clean prices table in raw CSV (local only due to GitHub 100 MB limit) |

---

## 💡 How to Load the Cleaned Data:

### In Python / Pandas:
```python
import pandas as pd

# Load CSV files:
df_calendar = pd.read_csv("data/processed/cleaned_calendar.csv")
df_lifecycle = pd.read_csv("data/processed/sku_lifecycle_metadata.csv")
df_starter_panel = pd.read_csv("data/processed/m5_starter_modeling_panel.csv")

# Load full compressed matrix:
df_sales = pd.read_parquet("data/processed/cleaned_sales_matrix.parquet")
df_prices = pd.read_parquet("data/processed/cleaned_prices.parquet")
```

### In Microsoft Excel / Google Sheets:
You can directly open `cleaned_calendar.csv`, `sku_lifecycle_metadata.csv`, or `m5_starter_modeling_panel.csv` in any spreadsheet editor.

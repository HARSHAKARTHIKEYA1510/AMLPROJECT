# Processed Data Directory (`data/processed/`)

This directory contains the cleaned, standardized, and feature-engineered datasets in **CSV format** produced by [`src/data_cleaning.py`](../../src/data_cleaning.py).

## Available CSV Files:

| File Name | Format | Size | On GitHub? | Description & Usage |
| :--- | :--- | :--- | :--- | :--- |
| **`cleaned_calendar.csv`** | CSV | **201 KB** | ✅ **Yes** | Daily calendar with cyclical trigonometric features, SNAP flags, Christmas markers |
| **`sku_lifecycle_metadata.csv`** | CSV | **2.5 MB** | ✅ **Yes** | Product release dates, active shelf days, and **cold-start indicators** (`is_cold_start`) |
| **`m5_starter_modeling_panel.csv`** | CSV | **18.9 MB** | ✅ **Yes** | **Recommended starter modeling panel** (CA_1 `FOODS_1`, 50 SKUs, 95.6K rows) with complete dates, causal lags ($t-1, 7, 14, 28$), rolling stats, and **stockout flags** |
| **`cleaned_sales_sample.csv`** | CSV | **1.9 MB** | ✅ **Yes** | 500-row preview sample of the cleaned sales matrix for instant CSV inspection |
| **`cleaned_prices_sample.csv`** | CSV | **229 KB** | ✅ **Yes** | 5,000-row preview sample of the cleaned prices table for instant CSV inspection |
| **`cleaned_sales_matrix.csv`** | CSV | **114 MB** | ⚙️ **Local** | Full 30,490 SKU sales matrix in CSV (local only due to GitHub's 100 MB limit) |
| **`cleaned_prices.csv`** | CSV | **307 MB** | ⚙️ **Local** | Full 6.84M price records in CSV (local only due to GitHub's 100 MB limit) |

---

## 💡 How to Load the Cleaned CSV Data:

### In Python / Pandas:
```python
import pandas as pd

# Load cleaned CSV files:
df_calendar = pd.read_csv("data/processed/cleaned_calendar.csv")
df_lifecycle = pd.read_csv("data/processed/sku_lifecycle_metadata.csv")
df_starter_panel = pd.read_csv("data/processed/m5_starter_modeling_panel.csv")
df_sales = pd.read_csv("data/processed/cleaned_sales_matrix.csv")
df_prices = pd.read_csv("data/processed/cleaned_prices.csv")
```

### In Microsoft Excel / Google Sheets:
You can directly open `cleaned_calendar.csv`, `sku_lifecycle_metadata.csv`, `m5_starter_modeling_panel.csv`, or the preview samples in any spreadsheet editor.

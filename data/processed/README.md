# Processed Data Directory (`data/processed/`)

This directory contains the cleaned, standardized datasets in **CSV format** produced by [`src/data_cleaning.py`](../../src/data_cleaning.py).

Every raw file in `data/raw/` has an exact **1-to-1 cleaned counterpart** in `data/processed/`:

## 1-to-1 Dataset Mapping:

| # | Raw File (`data/raw/`) | Cleaned File (`data/processed/`) | Rows | Size | On GitHub? | Cleaning Applied |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | **`calendar.csv`** | **`cleaned_calendar.csv`** | 1,969 | **216 KB** | ✅ **Yes** | Date parsing, cyclical temporal features (`day_sin`/`cos`, `month_sin`/`cos`), null events labeled 'None', SNAP & Christmas markers. |
| **2** | **`sell_prices.csv`** | **`cleaned_sell_prices.csv`** | 6,841,121 | **274 MB** | ⚙️ **Local** | Verified non-negative prices, regular price baselines, discount depth percentages, promo markers. |
| **3** | **`sales_train_validation.csv`** | **`cleaned_sales_train_validation.csv`** | 30,490 | **114 MB** | ⚙️ **Local** | Validated 0 nulls, 0 negative values across 1,913 days, non-negative integer clipping. |
| **4** | **`sales_train_evaluation.csv`** | **`cleaned_sales_train_evaluation.csv`** | 30,490 | **116 MB** | ⚙️ **Local** | Validated 0 nulls, 0 negative values across 1,941 days. |
| **5** | **`sample_submission.csv`** | **`cleaned_sample_submission.csv`** | 60,980 | **5.0 MB** | ✅ **Yes** | Verified forecast horizon schema ($F_1$ to $F_{28}$), zero-filled formatted baseline template. |

> ℹ️ **GitHub 100 MB Rule:**  
> Files under 100 MB (`cleaned_calendar.csv` and `cleaned_sample_submission.csv`) are tracked directly on GitHub.  
> The 3 large CSV files (`cleaned_sell_prices.csv`, `cleaned_sales_train_validation.csv`, and `cleaned_sales_train_evaluation.csv`) are $>100$ MB and reside locally in your `data/processed/` folder to prevent GitHub push errors. Run `python3 src/data_cleaning.py` to regenerate all 5 cleaned files at any time.

---

## 💡 How to Load in Python:
```python
import pandas as pd

df_calendar = pd.read_csv("data/processed/cleaned_calendar.csv")
df_prices   = pd.read_csv("data/processed/cleaned_sell_prices.csv")
df_sales    = pd.read_csv("data/processed/cleaned_sales_train_validation.csv")
df_eval     = pd.read_csv("data/processed/cleaned_sales_train_evaluation.csv")
df_sub      = pd.read_csv("data/processed/cleaned_sample_submission.csv")
```

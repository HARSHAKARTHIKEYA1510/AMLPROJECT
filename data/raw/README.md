# Raw Data Directory (`data/raw/`)

This directory stores the official raw dataset for the **Walmart M5 Forecasting Competition**.

## Files in this Directory:
| File Name | Size | On GitHub? | Description |
| :--- | :--- | :--- | :--- |
| **`calendar.csv`** | **101 KB** | ✅ **Yes** | Daily calendar dates, SNAP program days, holiday event names |
| **`sample_submission.csv`** | **5.0 MB** | ✅ **Yes** | 28-day forecasting submission template format ($F_1$ to $F_{28}$) |
| **`sales_train_validation.csv`** | **114.4 MB** | ⚙️ **Local (via script)** | Unit sales across 30,490 SKUs for days $d_1$ through $d_{1913}$ |
| **`sales_train_evaluation.csv`** | **116.1 MB** | ⚙️ **Local (via script)** | Unit sales extended through evaluation day $d_{1941}$ |
| **`sell_prices.csv`** | **193.9 MB** | ⚙️ **Local (via script)** | 6.84M weekly price records per store and item |

> ℹ️ **Why are the 3 large CSV files downloaded locally?**  
> GitHub enforces a strict **100 MB hard limit** on individual files. Uploading files $>100$ MB is permanently blocked by GitHub's servers (`remote: error: GH001: Large files detected`).  
> 
> All cleaned and memory-optimized datasets derived from these raw files are saved and tracked in **[`data/processed/`](../processed/)**.

## 🚀 One-Click Raw Data Ingestion:
If you cloned this repository and need the raw CSV files locally, run:
```bash
python3 src/data_loader.py
```
This automatically streams the complete raw files from the official M5 mirror directly into `data/raw/`.

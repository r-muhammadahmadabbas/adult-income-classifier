# Data

The UCI Adult Census dataset is **not committed** to this repository.

Download options:

1. **Automatic (default).** `load_raw()` fetches the data directly from the
   UCI repository:
   <https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data>

2. **Offline.** Download `adult.data`, place it in this directory, and pass the
   path explicitly:

   ```bash
   python -m income_classifier.cli evaluate --data data/adult.data
   ```

The raw file has 15 columns and no header. Rows containing `?` are dropped,
leaving roughly 45,000 clean rows.

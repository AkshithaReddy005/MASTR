# Dataset Setup Guide

## Using the Kaggle VRP Dataset

The code is now configured to use the **real Kaggle dataset** instead of random data.

---

## Step 1: Download the Dataset

1. Go to: https://www.kaggle.com/datasets/abhilashg23/vehicle-routing-problem-ga-dataset
2. Click **Download** to get the ZIP file
3. Extract the file `VRP - C101.csv` from the ZIP

---

## Step 2: Place the File

Put the CSV file in this exact location:
```
MASTR/data/raw/VRP - C101.csv
```

Full path:
```
c:\Users\akshi\OneDrive\Documents\AKKI\projects\MASTR\data\raw\VRP - C101.csv
```

The `data/raw/` folder has already been created for you.

---

## Step 3: Run Training

Once the file is in place, just run your training as normal:

```bash
python train/train_rl.py
```

Or the quick demo:

```bash
python scripts/quick_start.py
```

---

## How It Works

- **If the CSV file exists**: The code will automatically load real customer data from the file
- **If the CSV file doesn't exist**: The code falls back to generating random data (like before)

When you run the training, you'll see one of these messages:

### ✅ Using Real Data
```
✓ Loaded real data from data/raw/VRP - C101.csv
  - 20 customers
  - Depot at: [40. 50.]
```

### ⚠ Using Random Data (CSV not found)
```
⚠ Warning: Could not load CSV (File not found). Using random data instead.
```

---

## Dataset Format

The CSV file has this format:

| CUST_NO | XCOORD | YCOORD | DEMAND | READY_TIME | DUE_DATE | SERVICE_TIME |
|---------|--------|--------|--------|------------|----------|--------------|
| 0       | 40     | 50     | 0      | 0          | 1236     | 0            |
| 1       | 45     | 68     | 10     | 912        | 967      | 90           |
| 2       | 45     | 70     | 30     | 825        | 870      | 90           |
| ...     | ...    | ...    | ...    | ...        | ...      | ...          |

- **Row 0**: Depot location
- **Rows 1+**: Customer data

---

## Verify It's Working

Run this quick test:

```bash
python test_components.py
```

If you see the "✓ Loaded real data" message, you're all set!

---

## Alternative: Download via Command Line

If you have the Kaggle API set up:

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d abhilashg23/vehicle-routing-problem-ga-dataset

# Extract to the right folder
unzip vehicle-routing-problem-ga-dataset.zip -d data/raw/
```

---

## Summary

1. ✅ Folder created: `data/raw/`
2. ⏳ **YOU NEED TO**: Download `VRP - C101.csv` and place it in `data/raw/`
3. ✅ Code updated: Will automatically use real data when file exists
4. ✅ Fallback: Uses random data if file not found

**Once you download the file, your training will use REAL customer data from the Kaggle dataset!**

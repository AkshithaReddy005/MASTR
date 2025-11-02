from kagglehub import KaggleDatasetAdapter
import kagglehub
import pandas as pd
from utils.data_utils import add_soft_time_windows, save_processed_dataset

# Load raw dataset from Kaggle
# Requires: pip install kagglehub

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "abhilashg23/vehicle-routing-problem-ga-dataset",
    "VRP - C101.csv"
)

# Add soft time windows
df_processed = add_soft_time_windows(df)

# Save to processed folder
save_processed_dataset(df_processed, "MASTR/data/processed/VRP_C101_soft.csv")
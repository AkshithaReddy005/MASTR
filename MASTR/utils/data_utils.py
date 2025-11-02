# data_utils.py
import pandas as pd
import numpy as np

def add_soft_time_windows(df):
    df['start_time'] = np.random.randint(0, 50, size=len(df))
    df['end_time'] = df['start_time'] + np.random.randint(30, 80, size=len(df))
    df['penalty_early'] = np.random.uniform(0.5, 1.5, size=len(df))
    df['penalty_late'] = np.random.uniform(1.0, 2.5, size=len(df))
    return df

def save_processed_dataset(df, path):
    df.to_csv(path, index=False)
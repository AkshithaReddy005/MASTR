import os

folders = [
    "MASTR/data/raw",
    "MASTR/data/processed",
    "MASTR/env",
    "MASTR/model",
    "MASTR/train",
    "MASTR/utils",
    "MASTR/notebooks"
]

files = [
    "MASTR/env/mvrp_env.py",
    "MASTR/model/maam_model.py",
    "MASTR/train/train_rl.py",
    "MASTR/utils/data_utils.py",
    "MASTR/utils/metrics.py",
    "MASTR/notebooks/analysis.ipynb",
    "MASTR/README.md",
    "MASTR/requirements.txt"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    with open(file, "w") as f:
        f.write("# " + os.path.basename(file) + "\n")
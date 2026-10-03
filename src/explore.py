# %% Imports
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 100)

# %% Find project root
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()

DATA_DIR = ROOT / "data" / "raw"
print(ROOT)
print([f.name for f in DATA_DIR.glob("*.parquet")])

# %% Load
FILE = DATA_DIR / "Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet"
df = pd.read_parquet(FILE)
print(df.head())

# %% Inspect
print(df.shape)
print(df.columns.tolist())
print(df["Label"].value_counts())
df.info()

# %% Sanity checks
print("Duplicate rows:", df.duplicated().sum())
print("NaN values:", df.isna().sum().sum())
num = df.select_dtypes(include="number")
print("Inf values:", np.isinf(num).sum().sum())
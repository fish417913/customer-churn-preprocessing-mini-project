from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/customer_churn.csv")

df = pd.read_csv(DATA_PATH)

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values by column:")
print(df.isna().sum())

print("\nNumerical summary:")
print(df[["age", "tenure_months", "monthly_charges"]].describe())

print("\nCategorical values:")

categorical_columns = [
    "contract_type",
    "payment_method",
    "internet_service",
    "churn"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))

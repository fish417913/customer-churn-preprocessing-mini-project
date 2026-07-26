import json 
from pathlib import Path 


import joblib 
import pandas as pd

from preprocess import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    clean_data,
    validate_feature_schema
)

NEW_DATA_PATH = Path("data/raw/new_customers.csv")
PREPROCESSOR_PATH = Path("artifacts/preprocessor.joblib")
PROCESSED_DATA_PATH = Path(
    "data/processed/new_customers_processed.csv"
)

QUALITY_REPORT_PATH = Path(
    "artifacts/new_data_quality_report.json"
)

def find_unknown_categories(
    cleaned_df: pd.DataFrame,
    preprocessor
) -> dict[str, list[str]]:
    """Find categorical values not learned during training."""
    categorical_pipeline = preprocessor.named_transformers_[
        "categorical"
    ]
    
    encoder = categorical_pipeline.named_steps["encoder"]
    
    unknown_categories = {}
    
    for feature, known_categories in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_
    ):
        observed_values = set(
            cleaned_df[feature].dropna().unique()
        )
        
        known_values = set(known_categories)
        
        unexpected_values = sorted(
            observed_values - known_values
        )
        
        if unexpected_values:
            unknown_categories[feature] = unexpected_values
            
    return unknown_categories

def save_outputs(
    processed_df: pd.DataFrame,
    unknown_categories: dict[str, list[str]],
) -> None:
    """Save transformed records and the data-quality report."""
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    QUALITY_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    quality_report = {
        "records_processed": len(processed_df),
        "processed_feature_count": len(processed_df.columns) - 1,
        "missing_values_after_preprocessing": int(
            processed_df.isna().sum().sum()
        ),
        "unknown_categories": unknown_categories,
    }

    with QUALITY_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            quality_report,
            report_file,
            indent=2,
        )

def main() -> None:
    new_df = pd.read_csv(NEW_DATA_PATH)
    validate_feature_schema(new_df)
    
    cleaned_df = clean_data(new_df)
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    unknown_categories = find_unknown_categories(
        cleaned_df,
        preprocessor
    )
    
    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    processed_array = preprocessor.transform(
        cleaned_df[feature_columns]
    )
    
    processed_feature_names = preprocessor.get_feature_names_out()
    
    processed_df = pd.DataFrame(
        processed_array,
        columns=processed_feature_names,
        index=cleaned_df.index 
    )
    
    processed_df.insert(
        0,
        "customer_id",
        cleaned_df["customer_id"]
    )
    
    save_outputs(
        processed_df,
        unknown_categories
    )
    
    payment_columns = [
        column 
        for column in processed_df.columns 
        if column.startswith("payment_method_")
    ]
    
    categorical_pipeline = preprocessor.named_transformers_[
        "categorical"
    ]
    
    encoder = categorical_pipeline.named_steps["encoder"]
    
    print("\nCategories learned during training:")
    for feature, categories in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_
    ):
        print(f"{feature}: {list(categories)}")
    
    print("\nEncoded payment methods:")
    print(processed_df[["customer_id"] + payment_columns])
    
    print(f"Loaded {len(new_df)} new customer records.")
    print(f"Loaded preprocessor from: {PREPROCESSOR_PATH}")
    
    print("\nProcessed new data:")
    print(processed_df)
    
    print("\nMissing values after preprocessing:")
    print(processed_df.isna().sum().sum())
    
    print("\nUnknown categories detected:")
    
    if unknown_categories:
        for feature, values in unknown_categories.items():
            print(f"{feature}: {values}")
            
    else:
        print("None")
        
    print("\nSaved outputs:")
    print(PROCESSED_DATA_PATH)
    print(QUALITY_REPORT_PATH)
    
    
if __name__ == "__main__":
    main()
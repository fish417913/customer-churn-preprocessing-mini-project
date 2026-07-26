from pathlib import Path
import joblib 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

RAW_DATA_PATH = Path("data/raw/customer_churn.csv")

PROCESSED_DATA_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"

REQUIRED_COLUMNS = {
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charges",
    "contract_type",
    "payment_method",
    "internet_service",
    "churn"
}

NUMERICAL_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
]

TARGET_COLUMN = "churn"

def load_data(path: Path) -> pd.DataFrame:
    """Load customer churn data from a CSV file."""
    return pd.read_csv(path)

def validate_schema(df: pd.DataFrame) -> None:
    """Raise an error if any required columns are missing."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        missing_list = "".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_list}")
    
def validate_feature_schema(df: pd.DataFrame) -> None:
    """Raise an error if any required model features are missing."""
    required_features = set(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )
    
    missing_features = required_features - set(df.columns)
    
    if missing_features:
        missing_list = ", ".join(sorted(missing_features))
        raise ValueError(f"Missing required features: {missing_list}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize values and mark invalid observations as missing."""
    cleaned_df = df.copy()

    categorical_columns = CATEGORICAL_FEATURES.copy()
    
    if TARGET_COLUMN in cleaned_df.columns:
        categorical_columns.append(TARGET_COLUMN)

    for column in categorical_columns:
        cleaned_df[column] = (
            cleaned_df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

    cleaned_df.loc[
        ~cleaned_df["age"].between(18, 100),
        "age",
    ] = pd.NA

    cleaned_df.loc[
        cleaned_df["tenure_months"] < 0,
        "tenure_months",
    ] = pd.NA

    cleaned_df.loc[
        cleaned_df["monthly_charges"] <= 0,
        "monthly_charges",
    ] = pd.NA

    return cleaned_df

def separate_features_and_target(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the binary prediction target."""
    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_columns].copy()

    invalid_targets = set(df[TARGET_COLUMN].dropna().unique()) - {
        "yes",
        "no"
    }

    if invalid_targets:
        invalid_list = ",".join(sorted(invalid_targets))
        raise ValueError(f"Unexpected target values: {invalid_list}")

    y = df[TARGET_COLUMN].map(
        {
            "no": 0,
            "yes": 1
        }
    )

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    return X, y.astype("int64")

def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into stratified training and test sets."""
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

def build_numerical_pipeline() -> Pipeline:
    """Create the transformations applied to numerical features."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
         ]
    )

def build_categorical_pipeline() -> Pipeline:
    """Create the transformations applied to categorical features."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=pd.NA,
                    strategy="most_frequent"
                    )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
            )
        )
    ]
)
    
def build_preprocessor() -> ColumnTransformer:
    """Create the complete preprocessing system."""
    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                build_numerical_pipeline(),
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                build_categorical_pipeline(),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False 
    )
    
def save_artifacts(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    """Save the fitted preprocessor and model-ready datasets."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train 
    
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test 
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    joblib.dump(preprocessor,PREPROCESSOR_PATH)

def main() -> None:
    df = load_data(RAW_DATA_PATH)
    validate_schema(df)

    cleaned_df = clean_data(df)
    X, y = separate_features_and_target(cleaned_df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()
   
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    processed_feature_names = preprocessor.get_feature_names_out()
    
    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=processed_feature_names,
        index=X_train.index
    )
    
    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=processed_feature_names,
        index=X_test.index
    )
    
    save_artifacts(
        preprocessor,
        X_train_processed_df,
        X_test_processed_df,
        y_train,
        y_test
    )
   

    print(f"Loaded {len(df)} rows.")
    print("Schema validation passed.")
    
    print("\nProcessed training shape:")
    print(X_train_processed_df.shape)
    
    print("\nProcessed test shape:")
    print(X_test_processed_df.shape)
    
    print("\nMissing values in processed training data:")
    print(X_train_processed_df.isna().sum().sum())
    
    print("\nMissing values in processed test data:")
    print(X_test_processed_df.isna().sum().sum())
    
    print("\nFirst processed training rows:")
    print(X_train_processed_df.head())
    
    print("\nProcessed feature names:")
    for feature_name in processed_feature_names:
        print(feature_name)
        
    print("\nSaved artifacts:")
    print(TRAIN_DATA_PATH)
    print(TEST_DATA_PATH)
    print(PREPROCESSOR_PATH)

if __name__ == "__main__":
    main()

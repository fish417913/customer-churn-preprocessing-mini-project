import pandas as pd
import pytest

from preprocess import (
    build_preprocessor,
    clean_data, 
    separate_features_and_target,
    split_data, 
    validate_feature_schema,
    validate_inference_schema,
    validate_schema
)

from transform_new_data import find_unknown_categories

def test_validate_schema_rejects_missing_columns() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "age": [34],
        }
    )
    
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)

def test_clean_data_standardizes_and_invalidates_values() -> None:
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "age": [34, 150],
            "tenure_months": [12, -2],
            "monthly_charges": [74.50, 0],
            "contract_type": [" Month-to-month ", "month-to-month"],
            "payment_method": ["Credit card", None],
            "internet_service": ["Fiber optic", "DSL"],
            "churn": ["Yes", "No"]
        }
    )

    cleaned_df = clean_data(df)

    assert cleaned_df.loc[0, "contract_type"] == "month-to-month"
    assert cleaned_df.loc[1, "contract_type"] == "month-to-month"
    assert cleaned_df.loc[0, "churn"] == "yes"

    assert pd.isna(cleaned_df.loc[1, "age"])
    assert pd.isna(cleaned_df.loc[1, "tenure_months"])
    assert pd.isna(cleaned_df.loc[1, "monthly_charges"])
    assert pd.isna(cleaned_df.loc[1, "payment_method"])
    
def test_complete_preprocessor_produces_model_ready_data() -> None:
    df = pd.DataFrame(
            {
                "customer_id": [
                    "C001",
                    "C002",
                    "C003",
                    "C004",
                    "C005",
                    "C006",
                    "C007",
                    "C008",
                ],
                "age": [34, 150, 45, 28, 52, 39, None, 61],
                "tenure_months": [12, -2, 24, 4, 30, 18, 6, 48],
                "monthly_charges": [
                    74.50,
                    60.00,
                    None,
                    55.25,
                    85.30,
                    69.40,
                    58.75,
                    91.00,
                ],
                "contract_type": [
                    "Month-to-month",
                    "One year",
                    "Two year",
                    "Month-to-month",
                    "One year",
                    "Month-to-month",
                    "Month-to-month",
                    "Two year",
                ],
                "payment_method": [
                    "Credit card",
                    None,
                    "Bank transfer",
                    "Electronic check",
                    "Credit card",
                    "Bank transfer",
                    "Electronic check",
                    "Credit card",
                ],
                "internet_service": [
                    "Fiber optic",
                    "DSL",
                    "DSL",
                    "Fiber optic",
                    "Fiber optic",
                    "DSL",
                    "DSL",
                    "Fiber optic",
                ],
                "churn": [
                    "Yes",
                    "No",
                    "No",
                    "Yes",
                    "No",
                    "Yes",
                    "Yes",
                    "No",
                ],
            }
        )
    
    cleaned_df = clean_data(df)
    X, y = separate_features_and_target(cleaned_df)
    
    X_train, X_test, _, _ = split_data(X, y)
    
    preprocessor = build_preprocessor()
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out()
    
    assert X_train_processed.shape[0] == len(X_train)
    assert X_test_processed.shape[0] == len(X_test)
    assert X_train_processed.shape[1] == X_test_processed.shape[1]
    
    assert not pd.isna(X_train_processed).any()
    assert not pd.isna(X_test_processed).any()
    
    assert "customer_id" not in feature_names
    
def test_clean_data_accepts_unlabeled_future_data() -> None:
    future_df = pd.DataFrame(
        {
            "age": [150],
            "tenure_months": [6],
            "monthly_charges": [72.50],
            "contract_type": [" Month-to-month "],
            "payment_method": [None],
            "internet_service": ["Fiber optic"]
        }
    )
    
    validate_feature_schema(future_df)
    cleaned_df = clean_data(future_df)
    
    assert "churn" not in cleaned_df.columns 
    assert pd.isna(cleaned_df.loc[0, "age"])
    assert pd.isna(cleaned_df.loc[0, "payment_method"])
    
    assert (
        cleaned_df.loc[0, "contract_type"]
        == "month-to-month"
    )
    assert (
        cleaned_df.loc[0, "internet_service"]
        == "fiber optic"
    )
    
def test_find_unknown_categories_detects_unseen_value() -> None:
    training_df = pd.DataFrame(
        {
            "age": [34, 45, 58, 27],
            "tenure_months": [12, 24, 48, 3],
            "monthly_charges": [74.50, 65.00, 89.90, 55.25],
            "contract_type": [
                "month-to-month",
                "one year",
                "two year",
                "month-to-month",
            ],
            "payment_method": [
                "credit card",
                "credit card",
                "bank transfer",
                "electronic check",
            ],
            "internet_service": [
                "fiber optic",
                "dsl",
                "dsl",
                "fiber optic",
            ],
        }
    )

    preprocessor = build_preprocessor()
    preprocessor.fit(training_df)

    future_df = pd.DataFrame(
        {
            "age": [42],
            "tenure_months": [10],
            "monthly_charges": [75.00],
            "contract_type": ["month-to-month"],
            "payment_method": ["mailed check"],
            "internet_service": ["fiber optic"],
        }
    )

    unknown_categories = find_unknown_categories(
        future_df,
        preprocessor,
    )

    assert unknown_categories == {
        "payment_method": ["mailed check"]
    }
    
def test_validate_inference_schema_requires_customer_id() -> None:
    future_df = pd.DataFrame(
        {
            "age": [41],
            "tenure_months": [14],
            "monthly_charges": [76.25],
            "contract_type": ["month-to-month"],
            "payment_method": ["credit card"],
            "internet_service": ["fiber optic"]
        }
    )
    
    with pytest.raises(
        ValueError,
        match="customer_id"
    ):
        validate_inference_schema(future_df)
        
def test_clean_data_coerces_invalid_numeric_strings() -> None:
    future_df = pd.DataFrame(
        {
            "customer_id": ["N001", "N002"],
            "age": ["41", "unknown"],
            "tenure_months": ["14", "-3"],
            "monthly_charges": ["76.25", "not available"],
            "contract_type": ["Month-to-month", "One year"],
            "payment_method": ["Credit card", "Bank transfer"],
            "internet_service": ["Fiber optic", "DSL"]
        }
    )
    
    cleaned_df = clean_data(future_df)
    
    assert cleaned_df.loc[0, "age"] == 41
    assert cleaned_df.loc[0, "tenure_months"] == 14
    assert cleaned_df.loc[0, "monthly_charges"] == 76.25
    
    assert pd.isna(cleaned_df.loc[1, "age"])
    assert pd.isna(cleaned_df.loc[1, "tenure_months"])
    assert pd.isna(cleaned_df.loc[1, "monthly_charges"])
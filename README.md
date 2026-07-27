# Customer Churn Data Preprocessing Mini-Project

A foundational AI engineering mini-project demonstrating how to transform messy customer data into consistent, model-ready datasets.

The project focuses on the data handling and preprocessing steps that occur before model training. It builds a reusable preprocessing workflow for both labeled training data and new, unlabeled customer records.

## Project Objectives

This project demonstrates how to:

* Load and inspect CSV data with pandas
* Validate required columns before processing
* Identify missing, invalid, and inconsistent values
* Standardize categorical text
* Separate model features from the prediction target
* Encode a binary target
* Create a reproducible stratified train/test split
* Prevent preprocessing data leakage
* Impute missing numerical and categorical values
* Standardize numerical features
* One-hot encode categorical features
* Combine transformations with a `ColumnTransformer`
* Save processed training and test datasets
* Serialize and reuse a fitted preprocessing pipeline
* Transform new, unlabeled customer records
* Detect categorical values not observed during training
* Generate a data-quality report
* Test preprocessing behavior with pytest

## Project Workflow

### Training Workflow

```text
Raw labeled CSV
        ↓
Schema validation
        ↓
Rule-based cleaning
        ↓
Feature and target separation
        ↓
Stratified train/test split
        ↓
Fit preprocessing on training data only
        ↓
Transform training and test data
        ↓
Save processed datasets
        ↓
Save fitted preprocessor
```

### New-Data Workflow

```text
New unlabeled CSV
        ↓
Feature-schema validation
        ↓
Rule-based cleaning
        ↓
Load saved preprocessor
        ↓
Detect unknown categories
        ↓
Transform without refitting
        ↓
Save model-ready data
        ↓
Save data-quality report
```

## Project Structure

```text
customer-churn-preprocessing-mini-project/
├── artifacts/
│   ├── new_data_quality_report.json
│   └── preprocessor.joblib
├── data/
│   ├── processed/
│   │   ├── new_customers_processed.csv
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw/
│       ├── customer_churn.csv
│       └── new_customers.csv
├── tests/
│   └── test_preprocess.py
├── .gitignore
├── inspect_data.py
├── preprocess.py
├── pytest.ini
├── README.md
├── requirements.txt
└── transform_new_data.py
```

## Raw Dataset

The training dataset represents fictional customer churn records.

### Numerical Features

* `age`
* `tenure_months`
* `monthly_charges`

### Categorical Features

* `contract_type`
* `payment_method`
* `internet_service`

### Target

* `churn`

The target is converted to a binary representation:

```text
No  → 0
Yes → 1
```

### Identifier

* `customer_id`

The customer identifier is retained for traceability but excluded from model features.

## Intentional Data-Quality Problems

The raw dataset includes several deliberate problems:

* Missing customer age
* Missing monthly charges
* Missing payment method
* Negative customer tenure
* An unrealistic customer age
* Inconsistent capitalization in contract types
* Identifier data that should not be used as a predictive feature

These problems provide realistic preprocessing cases for the project.

## Rule-Based Cleaning

The `clean_data()` function performs deterministic cleanup before the scikit-learn preprocessing pipeline runs.

### Categorical Cleanup

Categorical values are:

* Converted to pandas string values
* Stripped of leading and trailing whitespace
* Converted to lowercase

For example:

```text
" Month-to-month " → "month-to-month"
"Credit card"      → "credit card"
"Fiber optic"      → "fiber optic"
```

This prevents capitalization or whitespace differences from creating false categories.

### Numerical Validation Rules

The project applies the following validity rules:

```text
18 ≤ age ≤ 100
tenure_months ≥ 0
monthly_charges > 0
```

Values outside these ranges are changed to missing values rather than deleting the entire customer record.

Examples:

```text
age = 150          → missing
tenure_months = -2 → missing
monthly_charges = 0 → missing
```

The preprocessing pipeline later imputes those missing feature values.

## Schema Validation

The project uses two schema-validation functions.

### Training Schema

`validate_schema()` checks that labeled training data contains:

* Customer identifier
* All numerical features
* All categorical features
* Churn target

### Feature Schema

`validate_feature_schema()` checks new inference data for the required model features.

New data does not require a `churn` column because churn is the unknown outcome a future model would predict.

## Preventing Data Leakage

The data is split into training and test sets before the preprocessing pipeline is fitted.

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
```

The preprocessing system is fitted only on `X_train`.

```text
X_train → fit and transform
X_test  → transform only
```

This prevents information from the test set from influencing:

* Numerical medians
* Numerical means
* Numerical standard deviations
* Most-frequent categorical values
* One-hot encoded category definitions

The `random_state` makes the split reproducible.

The `stratify` argument preserves the churn-class proportions as closely as possible in both subsets.

## Numerical Preprocessing

The numerical pipeline performs two transformations.

### Median Imputation

```python
SimpleImputer(strategy="median")
```

Missing numerical values are replaced using medians learned from the training set.

For the current training split, the learned medians are:

```text
age:             37.500
tenure_months:   16.500
monthly_charges: 69.825
```

Median imputation is less sensitive to extreme values than mean imputation.

### Standard Scaling

```python
StandardScaler()
```

Numerical values are centered and scaled using the training-set mean and standard deviation.

The general transformation is:

```text
standardized value = (original value - training mean) / training standard deviation
```

A negative standardized value does not mean the original value was negative. It means the original value was below the training-set mean.

## Categorical Preprocessing

The categorical pipeline performs two transformations.

### Most-Frequent Imputation

```python
SimpleImputer(
    missing_values=pd.NA,
    strategy="most_frequent",
)
```

Missing categorical values are replaced using the most common category learned from the training set.

For the current split, the learned categorical replacements are:

```text
contract_type:    month-to-month
payment_method:   electronic check
internet_service: dsl
```

### One-Hot Encoding

```python
OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
)
```

Categorical strings are converted into binary indicator columns.

For example:

```text
contract_type_month-to-month
contract_type_one year
contract_type_two year
```

A month-to-month customer is represented as:

```text
1, 0, 0
```

A two-year customer is represented as:

```text
0, 0, 1
```

The encoder produces eight categorical output columns:

```text
contract_type_month-to-month
contract_type_one year
contract_type_two year
payment_method_bank transfer
payment_method_credit card
payment_method_electronic check
internet_service_dsl
internet_service_fiber optic
```

## Combined Preprocessor

The numerical and categorical pipelines are combined using a scikit-learn `ColumnTransformer`.

```text
Input DataFrame
    ├── Numerical columns → imputation → scaling
    └── Categorical columns → imputation → one-hot encoding
                                      ↓
                         Combined feature matrix
```

The final model-ready dataset contains:

```text
3 standardized numerical features
8 one-hot encoded categorical features
──────────────────────────────────
11 processed model features
```

The `customer_id` column is not passed through the transformer.

## Processed Training and Test Data

Running the training preprocessing workflow generates:

```text
data/processed/train.csv
data/processed/test.csv
```

The training file contains:

```text
15 rows
11 processed features
1 churn target
12 total columns
0 missing values
```

The test file contains:

```text
5 rows
11 processed features
1 churn target
12 total columns
0 missing values
```

## Saved Preprocessor

The fitted preprocessing object is saved as:

```text
artifacts/preprocessor.joblib
```

The saved object contains the preprocessing values learned from the training data, including:

* Numerical medians
* Numerical means
* Numerical standard deviations
* Most-frequent categorical values
* Known categorical levels
* One-hot output ordering

New records must use this fitted object rather than fitting a new preprocessor.

## Processing New Customer Records

The `transform_new_data.py` script processes new, unlabeled customer records.

The sample file contains three cases:

* A normal customer record
* A customer with missing age and payment method
* A customer with a previously unseen payment category

The input file is:

```text
data/raw/new_customers.csv
```

The transformed output is:

```text
data/processed/new_customers_processed.csv
```

The saved output contains:

```text
3 customer records
1 customer identifier column
11 model-ready features
0 missing values
```

## Unknown Categories

The fitted encoder learned these payment-method categories:

```text
bank transfer
credit card
electronic check
```

The new customer data contains:

```text
mailed check
```

Because the encoder uses:

```python
handle_unknown="ignore"
```

the pipeline does not crash. The unknown value is encoded as zero across all known payment-method columns.

```text
payment_method_bank transfer       0
payment_method_credit card         0
payment_method_electronic check    0
```

The project separately detects and reports unknown categories so they are not silently overlooked.

Unknown categories may indicate:

* Data drift
* New business categories
* Typographical errors
* Changes in source systems
* A need to retrain the preprocessing system

## Data-Quality Report

The new-data workflow generates:

```text
artifacts/new_data_quality_report.json
```

Example:

```json
{
  "records_processed": 3,
  "processed_feature_count": 11,
  "missing_values_after_preprocessing": 0,
  "unknown_categories": {
    "payment_method": [
      "mailed check"
    ]
  }
}
```

This separates two concerns:

```text
Transformation status:
The records can be processed by the model.

Data-quality status:
The incoming data contains a value not observed during training.
```

## Progress and Key Learnings

This mini-project produced a reproducible preprocessing workflow that converts messy customer-churn data into validated, model-ready datasets. The completed pipeline separates data-quality checks from statistical preprocessing and supports both model-training data and future inference records.

### Progress Completed

The project now includes:

* Schema validation to confirm that required columns are present.
* Rule-based cleaning for missing, malformed, and invalid values.
* Numeric type coercion using `pd.to_numeric(..., errors="coerce")`.
* A train/test split performed before fitting preprocessing steps, preventing data leakage.
* Median imputation and standardization for numeric features.
* Most-frequent imputation and one-hot encoding for categorical features.
* A fitted scikit-learn `ColumnTransformer` saved as `artifacts/preprocessor.joblib`.
* Separate validation and transformation logic for new customer records.
* Processed training, testing, and inference datasets saved under `data/processed/`.
* A data-quality report for new records saved as `artifacts/new_data_quality_report.json`.
* Five automated tests covering important preprocessing behaviors.

### Key Learnings

The most important lesson was that successful preprocessing requires more than handling visibly missing values. A value can be present in a dataset while still being invalid, such as a text value in a numeric column or a category that was not observed during training. Validation must therefore examine column presence, expected data types, valid ranges, and categorical values.

The project also reinforced the importance of separating training-time preprocessing from inference-time preprocessing. The preprocessor must be fitted only on the training data and then reused without refitting for test data or future customer records. This prevents leakage and ensures that production predictions use the same transformations as the model originally received.

Another key learning involved schema drift. New data may contain all required columns but still fail because their underlying data types have changed. Explicitly coercing numeric features before transformation makes the pipeline more resilient while allowing invalid values to be identified and handled consistently.

Finally, the project demonstrated the value of preserving row alignment and producing quality reports alongside transformed data. A preprocessing pipeline should not only generate model-ready features; it should also make it possible to trace problems back to the original records, understand how the data changed, and verify the workflow through automated tests.

Overall, the mini-project strengthened my understanding of how validation, cleaning, feature transformation, artifact persistence, and inference-time safeguards work together to create a reliable AI engineering pipeline.


## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Inspect the Raw Data

Run:

```bash
python inspect_data.py
```

The inspection script displays:

* Initial rows
* Dataset dimensions
* Column data types
* Missing-value counts
* Numerical summary statistics
* Categorical value counts

## Run the Training Preprocessing Workflow

Run:

```bash
python preprocess.py
```

This command:

1. Loads the labeled churn dataset.
2. Validates the schema.
3. Cleans invalid and inconsistent values.
4. Separates features and target.
5. Creates a stratified train/test split.
6. Fits the preprocessor on training data.
7. Transforms training and test data.
8. Saves the processed datasets.
9. Saves the fitted preprocessor.

Generated outputs:

```text
data/processed/train.csv
data/processed/test.csv
artifacts/preprocessor.joblib
```

## Transform New Customer Data

The fitted preprocessor must exist before this command is run.

Run:

```bash
python transform_new_data.py
```

This command:

1. Loads new customer records.
2. Validates the feature schema.
3. Cleans incoming values.
4. Loads the saved preprocessor.
5. Detects unknown categorical values.
6. Transforms the data without refitting.
7. Saves the processed records.
8. Saves the data-quality report.

Generated outputs:

```text
data/processed/new_customers_processed.csv
artifacts/new_data_quality_report.json
```

## Run the Tests

Run:

```bash
python -m pytest -v
```

The test suite verifies:

* Missing required columns are rejected.
* Categorical values are standardized.
* Invalid numerical values become missing.
* The complete preprocessor removes missing values.
* Training and test feature widths match.
* Customer identifiers are excluded from model features.
* Unlabeled future data can be cleaned.
* Unknown categories are detected.

Current test result:

```text
5 passed
```

## Key AI Engineering Lessons

### Raw data should remain unchanged

The workflow creates copies before cleaning and saves transformed data separately under `data/processed`.

### Validation should happen early

Clear schema and target validation prevents confusing failures later in the pipeline.

### Missing and invalid are different concepts

A value may be absent, or it may be present but outside an accepted range. Both cases require explicit handling.

### Features and targets require different treatment

Missing feature values may be imputed. Missing target values should not be guessed because the target is the answer the model is supposed to learn.

### Preprocessing must be fitted on training data only

Using test data to calculate medians, means, standard deviations, or categories would create data leakage.

### Training and inference must use the same transformations

Loading the fitted preprocessor ensures that future records use the same statistics, categories, and column ordering as the training data.

### Unknown categories should be handled and monitored

Ignoring unknown categories keeps inference operational, but reporting them provides visibility into possible drift or upstream changes.

### Tests should be self-contained

The tests build their own data and fit their own preprocessing objects rather than depending on files created by previous script executions.

## Known Limitations

* The dataset is fictional and contains only 20 labeled records.
* The processed data is not large enough for meaningful churn-model evaluation.
* Numerical validation rules are manually defined.
* Unknown categories are reported but not automatically escalated.
* No trained classification model is included.
* The serialized preprocessor may depend on compatible versions of Python, NumPy, pandas, joblib, and scikit-learn.
* `handle_unknown="ignore"` represents unseen categories with zeros, which may hide meaningful distinctions from a future model.
* The quality report checks unknown categorical values but does not yet report distribution drift or numerical-range drift.

## Potential Next Improvements

* Train a logistic regression churn classifier using the processed data.
* Combine preprocessing and modeling into one scikit-learn pipeline.
* Add cross-validation and evaluation metrics.
* Add command-line arguments for input and output paths.
* Add logging instead of relying only on print statements.
* Add duplicate-record detection.
* Add stronger datatype validation.
* Add numerical drift monitoring.
* Add category-frequency drift monitoring.
* Add configurable validation rules.
* Save metadata such as training date and package versions.
* Package the source code as an installable Python module.
* Add continuous integration with GitHub Actions.

## Technologies Used

* Python
* pandas
* NumPy
* scikit-learn
* joblib
* pytest

## Educational Outcome

This project demonstrates that data preprocessing is not merely a collection of one-time CSV edits.

A reusable AI engineering preprocessing system must:

* Validate assumptions
* Clean deterministic data problems
* Prevent leakage
* Learn transformations only from training data
* Preserve feature consistency
* Support future unlabeled data
* Save reusable artifacts
* Detect unexpected inputs
* Produce auditable outputs
* Include automated tests

The resulting datasets are ready to serve as inputs to a future customer-churn classification model.

## Debugging Improvements

This project was hardened against two inference-time preprocessing failures.

### 1. Inference Schema Validation

The original feature validation checked only the model input columns. However, the future-data transformation workflow also required `customer_id` when creating the processed output.

A new `validate_inference_schema()` function now verifies that incoming inference data contains:

* `customer_id`
* All numerical features
* All categorical features

This provides a clear validation error instead of allowing the workflow to fail later with a `KeyError`.

### 2. Numeric Data-Type Drift

Future CSV files may contain numeric values represented as strings or invalid text values such as `"unknown"` or `"not available"`.

The cleaning process now converts numerical columns using:

```python
pd.to_numeric(column, errors="coerce")
```

Valid numeric strings are converted to numbers, while malformed values are converted to missing values. The existing median imputation step then handles those missing values during preprocessing.

### Regression Testing

Regression tests were added to verify that:

* Inference data without `customer_id` is rejected with a clear error.
* Valid numeric strings are converted correctly.
* Invalid numeric strings are treated as missing values.
* The fitted preprocessor can transform future data without changing the feature structure.
* No missing values remain in the processed model input.

Run the test suite with:

```bash
python -m pytest -q
```

Current result:

```text
7 passed
```

### Debugging Outcome

The final workflow successfully:

* Processes training and test data into 11 consistent model features.
* Preserves `customer_id` in future-data output.
* Handles missing and malformed numerical values.
* Ignores unseen categories during one-hot encoding.
* Reports unseen categories separately for data-quality monitoring.
* Produces transformed datasets with no remaining missing values.

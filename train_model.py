import argparse
import pathlib

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


NUMERICAL_COLS = [
    "CredRate",
    "Age",
    "Tenure",
    "Balance",
    "Prod Number",
    "HasCrCard",
    "ActMem",
    "EstimatedSalary",
]


def train_and_save(csv_path, out_dir):
    df = pd.read_csv(csv_path)
    if "CustomerId" in df.columns:
        df = df.drop("CustomerId", axis=1)

    categorical_cols = df.select_dtypes(include=["object"]).columns
    df_final = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    if df_final.isnull().sum().sum() > 0:
        for col in df_final.columns:
            if df_final[col].isnull().any() and pd.api.types.is_numeric_dtype(df_final[col]):
                df_final[col] = df_final[col].fillna(df_final[col].median())

    if "Exited" not in df_final.columns:
        raise ValueError("Target column 'Exited' not found in the CSV.")

    X = df_final.drop("Exited", axis=1)
    y = df_final["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train[NUMERICAL_COLS] = scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test[NUMERICAL_COLS] = scaler.transform(X_test[NUMERICAL_COLS])

    model = LogisticRegression(max_iter=1000, class_weight="balanced") #threshold = 0.5
    model.fit(X_train, y_train)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "churn_model.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(X_train.columns.tolist(), out_dir / "model_columns.joblib")


def main():
    parser = argparse.ArgumentParser(
        description="Train churn model and save artifacts."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to Churn_Modelling.csv",
    )
    parser.add_argument(
        "--out",
        default="model",
        help="Output directory for model artifacts (default: model folder)",
    )
    args = parser.parse_args()
    train_and_save(args.csv, args.out)


if __name__ == "__main__":
    main()

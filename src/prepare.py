
import numpy as np
import pandas as pd
from load_data import load_raw

FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


def load_clean_dataset():
    df = load_raw()

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    if "Output (S)" in df.columns:
        df = df.rename(columns={"Output (S)": "fault"})
    elif "Output" in df.columns:
        df = df.rename(columns={"Output": "fault"})

    required = FEATURES + ["fault"]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    if not set(df["fault"].unique()).issubset({0, 1}):
        raise ValueError("fault column must contain only 0 and 1")

    return df


def split_dataset(df=None):
    if df is None:
        df = load_clean_dataset()

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(df))

    train_end = int(0.60 * len(df))
    val_end = int(0.80 * len(df))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["fault"].to_numpy(dtype=int)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_train_validation_test(X_train, X_val, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_val_scaled = (X_val - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled, mean, std


def prepare_data():
    df = load_clean_dataset()
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_dataset(df)
    X_train_scaled, X_val_scaled, X_test_scaled, mean, std = standardize_train_validation_test(
        X_train_raw, X_val_raw, X_test_raw
    )

    print("\n========== DATA SPLIT ==========")
    print("Training:", X_train_scaled.shape, y_train.shape)
    print("Validation:", X_val_scaled.shape, y_val.shape)
    print("Testing:", X_test_scaled.shape, y_test.shape)
    print("\nTraining feature means (scaled; near zero):")
    print(X_train_scaled.mean(axis=0))
    print("\nTraining feature standard deviations (scaled; near one):")
    print(X_train_scaled.std(axis=0))
    print("\nData preparation completed successfully.")

    return (
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        mean, std,
    )


def prepare_data_raw():
    df = load_clean_dataset()
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_dataset(df)
    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test


if __name__ == "__main__":
    prepare_data()
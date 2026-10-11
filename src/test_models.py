
from pathlib import Path

import joblib
import pandas as pd

from logistic import add_intercept, make_set_q, predict
from metrics import print_metrics
from prepare import prepare_data_raw

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def load_model(path):
    return joblib.load(path)


def prepare_feature_matrix(raw, model_info, feature_set):
    if feature_set == "R":
        X = raw
    elif feature_set == "Q":
        X = make_set_q(raw)
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    mean = model_info["mean"]
    std = model_info["std"]
    std = std.copy()
    std[std == 0] = 1.0
    return (X - mean) / std


def main():
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data_raw()

    rows = []

    # Logistic Regression: Set R and Set Q
    for set_name in ["R", "Q"]:
        model = load_model(RESULTS / f"logistic_{set_name}.pkl")
        X_test_prepared = prepare_feature_matrix(X_test_raw, model, set_name)
        predictions = predict(add_intercept(X_test_prepared), model["theta"])

        acc, cm, missed = print_metrics(
            y_test, predictions, f"Logistic Set {set_name}"
        )

        rows.append({
            "model": f"Logistic Set {set_name}",
            "accuracy": acc,
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN_missed_faults": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
            "missed_faults": int(missed),
        })

    # Best Linear and best RBF SVM
    for name in ["svm_linear", "svm_rbf"]:
        saved = load_model(RESULTS / f"{name}.pkl")
        model = saved["model"]
        X_test_prepared = (X_test_raw - saved["mean"]) / saved["std"]
        predictions = model.predict(X_test_prepared)

        acc, cm, missed = print_metrics(
            y_test, predictions, name
        )

        rows.append({
            "model": name,
            "accuracy": acc,
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN_missed_faults": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
            "missed_faults": int(missed),
        })

    df = pd.DataFrame(rows)
    output = TABLES / "table_4_test_results.csv"
    df.to_csv(output, index=False)
    df.to_csv(TABLES / "test_results.csv", index=False)

    print("\n========== FINAL COMPARISON ==========")
    for _, row in df.iterrows():
        print(
            f"{row['model']}: accuracy={row['accuracy']:.4f}, "
            f"missed faults={row['missed_faults']}"
        )

    print("\nSaved final test results to:", output)


if __name__ == "__main__":
    main()
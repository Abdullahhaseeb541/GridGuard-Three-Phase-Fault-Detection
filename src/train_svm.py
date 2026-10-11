
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

from metrics import accuracy_score_manual
from prepare import prepare_data

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

C_VALUES = [0.1, 1, 10, 100]
GAMMAS = [0.1, 1]


def save_table(rows, filename):
    pd.DataFrame(rows).to_csv(TABLES / filename, index=False)


def main():
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        mean, std,
    ) = prepare_data()

    validation_rows = []
    best_linear = None
    best_rbf = None

    print("\n========== LINEAR SVM ==========")
    for C in C_VALUES:
        model = SVC(kernel="linear", C=C)
        model.fit(X_train, y_train)

        pred = model.predict(X_val)
        acc = accuracy_score_manual(y_val, pred)
        support_count = len(model.support_)

        row = {
            "kernel": "linear",
            "C": C,
            "gamma": "N/A",
            "validation_accuracy": float(acc),
            "support_vectors": int(support_count),
        }
        validation_rows.append(row)

        print(
            f"C={C}: validation accuracy={acc:.4f}, "
            f"support vectors={support_count}"
        )

        if best_linear is None or acc > best_linear["validation_accuracy"]:
            best_linear = {
                "model": model,
                "kernel": "linear",
                "C": C,
                "gamma": None,
                "validation_accuracy": float(acc),
                "support_vectors": int(support_count),
                "mean": mean,
                "std": std,
            }

    print("\n========== RBF SVM ==========")
    for C in C_VALUES:
        for gamma in GAMMAS:
            model = SVC(kernel="rbf", C=C, gamma=gamma)
            model.fit(X_train, y_train)

            pred = model.predict(X_val)
            acc = accuracy_score_manual(y_val, pred)
            support_count = len(model.support_)

            row = {
                "kernel": "rbf",
                "C": C,
                "gamma": float(gamma),
                "validation_accuracy": float(acc),
                "support_vectors": int(support_count),
            }
            validation_rows.append(row)

            print(
                f"C={C}, gamma={gamma}: validation accuracy={acc:.4f}, "
                f"support vectors={support_count}"
            )

            if best_rbf is None or acc > best_rbf["validation_accuracy"]:
                best_rbf = {
                    "model": model,
                    "kernel": "rbf",
                    "C": C,
                    "gamma": float(gamma),
                    "validation_accuracy": float(acc),
                    "support_vectors": int(support_count),
                    "mean": mean,
                    "std": std,
                }

    save_table(validation_rows, "table_3_svm_validation_results.csv")

    linear_summary = [
        {
            "kernel": row["kernel"],
            "C": row["C"],
            "gamma": row["gamma"],
            "validation_accuracy": row["validation_accuracy"],
            "support_vectors": row["support_vectors"],
        }
        for row in validation_rows if row["kernel"] == "linear"
    ]
    rbf_summary = [
        {
            "kernel": row["kernel"],
            "C": row["C"],
            "gamma": row["gamma"],
            "validation_accuracy": row["validation_accuracy"],
            "support_vectors": row["support_vectors"],
        }
        for row in validation_rows if row["kernel"] == "rbf"
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    linear_curve = {
        row["C"]: row["validation_accuracy"] for row in linear_summary
    }
    ax.plot(
        list(linear_curve.keys()),
        list(linear_curve.values()),
        label="linear",
        marker="o",
    )

    for gamma in GAMMAS:
        gamma_curve = {
            row["C"]: row["validation_accuracy"]
            for row in rbf_summary if row["gamma"] == float(gamma)
        }
        ax.plot(
            list(gamma_curve.keys()),
            list(gamma_curve.values()),
            label=f"RBF gamma={gamma}",
            marker="o",
        )

    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("SVM Validation Accuracy vs C")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "svm_validation_accuracy_vs_C.png", dpi=160)
    plt.close(fig)

    for name, model_info in [("svm_linear", best_linear), ("svm_rbf", best_rbf)]:
        joblib.dump(model_info, RESULTS / f"{name}.pkl")
        print(f"\nBest {model_info['kernel']} SVM")
        print("C:", model_info["C"])
        if model_info["kernel"] == "rbf":
            print("gamma:", model_info["gamma"])
        print("Validation accuracy:", model_info["validation_accuracy"])
        print("Support vectors:", model_info["support_vectors"])
        print("Saved:", RESULTS / f"{name}.pkl")

    print("\nSaved validation table:", TABLES / "table_3_svm_validation_results.csv")
    print("Saved SVM plot:", FIGURES / "svm_validation_accuracy_vs_C.png")


if __name__ == "__main__":
    main()
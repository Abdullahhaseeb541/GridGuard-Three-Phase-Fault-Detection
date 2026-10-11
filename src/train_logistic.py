
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from logistic import (
    Q_FEATURE_ORDER,
    add_intercept,
    gradient_descent,
    make_set_q,
    predict,
)
from metrics import accuracy_score_manual
from prepare import prepare_data_raw, standardize_train_validation_test

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
TABLES.mkdir(exist_ok=True)

ALPHAS = [0.01, 0.1, 1, 10]
ITERATIONS = 2000


def save_table(rows, filename):
    pd.DataFrame(rows).to_csv(TABLES / filename, index=False)


def select_stable_alpha(X_train, y_train):
    stable_runs = []
    histories = {}

    for alpha in ALPHAS:
        theta, costs = gradient_descent(
            add_intercept(X_train), y_train, alpha, ITERATIONS
        )
        differences = np.diff(costs)
        stable = np.all(differences < 0)
        histories[alpha] = costs

        print(
            f"alpha={alpha}: initial cost={costs[0]:.6f}, "
            f"final cost={costs[-1]:.6f}, stable={stable}"
        )

        if stable:
            stable_runs.append((alpha, theta, costs))

    if not stable_runs:
        raise RuntimeError(
            "No alpha decreased cost at every iteration. "
            "Inspect cost curves and implementation."
        )

    return max(stable_runs, key=lambda item: item[0]), histories


def plot_cost_histories(set_name, histories):
    fig, ax = plt.subplots(figsize=(10, 6))

    for alpha, costs in histories.items():
        ax.plot(
            np.arange(len(costs)),
            costs,
            label=f"alpha={alpha}",
            linewidth=1.8,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Binary cross-entropy cost")
    ax.set_title(f"Logistic Regression Cost History — Set {set_name}")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / f"logistic_cost_history_set_{set_name}.png", dpi=160)
    plt.close(fig)


def main():
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = prepare_data_raw()

    raw_sets = {
        "R": (X_train_raw, X_val_raw, X_test_raw),
        "Q": (make_set_q(X_train_raw), make_set_q(X_val_raw), make_set_q(X_test_raw)),
    }

    standardized_sets = {}
    means = {}
    stds = {}

    for set_name, (train_raw, val_raw, test_raw) in raw_sets.items():
        X_train_s, X_val_s, X_test_s, mean, std = standardize_train_validation_test(
            train_raw, val_raw, test_raw
        )
        standardized_sets[set_name] = {
            "train": X_train_s,
            "val": X_val_s,
            "test": X_test_s,
        }
        means[set_name] = mean
        stds[set_name] = std

    results = []
    set_q_weight_table_rows = []

    for set_name in ["R", "Q"]:
        X_train = standardized_sets[set_name]["train"]
        X_val = standardized_sets[set_name]["val"]
        X_test = standardized_sets[set_name]["test"]

        print(f"\n========== LOGISTIC SET {set_name} ==========")
        (alpha, theta, costs), histories = select_stable_alpha(X_train, y_train)

        plot_cost_histories(set_name, histories)

        train_pred = predict(add_intercept(X_train), theta)
        train_acc = accuracy_score_manual(y_train, train_pred)

        X_val_bias = add_intercept(X_val)
        val_pred = predict(X_val_bias, theta)
        val_acc = accuracy_score_manual(y_val, val_pred)

        model = {
            "theta": theta,
            "alpha": alpha,
            "feature_set": set_name,
            "mean": means[set_name],
            "std": stds[set_name],
            "feature_order": Q_FEATURE_ORDER if set_name == "Q" else [
                "intercept", "Ia", "Ib", "Ic", "Va", "Vb", "Vc"
            ],
            "iterations": ITERATIONS,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        }

        model_path = RESULTS / f"logistic_{set_name}.pkl"
        joblib.dump(model, model_path)
        np.save(RESULTS / f"logistic_{set_name}_costs.npy", costs)

        results.append({
            "set": set_name,
            "selected_alpha": alpha,
            "training_accuracy": float(train_acc),
            "validation_accuracy": float(val_acc),
            "final_cost": float(costs[-1]),
        })

        print("Chosen alpha:", alpha)
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print("Saved:", model_path)

        if set_name == "Q":
            weights = theta[1:]
            for name, value in zip(Q_FEATURE_ORDER, weights):
                set_q_weight_table_rows.append({
                    "feature": name,
                    "weight": float(value),
                })

    save_table(results, "table_2_logistic_regression.csv")
    save_table(set_q_weight_table_rows, "table_5_set_q_weights.csv")

    print("\n========== VALIDATION SUMMARY ==========")
    for row in results:
        print(
            f"Set {row['set']}: alpha={row['selected_alpha']}, "
            f"validation accuracy={row['validation_accuracy']:.4f}, "
            f"final cost={row['final_cost']:.6f}"
        )


if __name__ == "__main__":
    main()
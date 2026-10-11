
import numpy as np


def accuracy_score_manual(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    return np.mean(y_true == y_pred)


def confusion_matrix_manual(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    matrix = np.zeros((2, 2), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1

    return matrix


def missed_faults_manual(y_true, y_pred):
    matrix = confusion_matrix_manual(y_true, y_pred)
    # Row = actual class, column = predicted class
    # Actual fault (1), predicted no fault (0)
    return int(matrix[1, 0])


def print_metrics(y_true, y_pred, name="Model"):
    acc = accuracy_score_manual(y_true, y_pred)
    cm = confusion_matrix_manual(y_true, y_pred)
    missed = missed_faults_manual(y_true, y_pred)

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)
    print("Missed faults (actual 1, predicted 0):", missed)

    return acc, cm, missed
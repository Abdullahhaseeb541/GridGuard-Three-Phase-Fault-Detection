
import numpy as np


FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
Q_FEATURE_ORDER = FEATURES + [f"{name}^2" for name in FEATURES]


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X):
    X = np.asarray(X, dtype=float)
    return np.column_stack((np.ones(X.shape[0]), X))


def compute_cost(X, y, theta):
    probabilities = sigmoid(X @ theta)
    eps = 1e-12

    probabilities = np.clip(probabilities, eps, 1 - eps)

    return -np.mean(
        y * np.log(probabilities)
        + (1 - y) * np.log(1 - probabilities)
    )


def gradient_descent(X, y, alpha, iterations=2000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    theta = np.zeros(X.shape[1], dtype=float)
    costs = [compute_cost(X, y, theta)]

    for _ in range(iterations):
        probabilities = sigmoid(X @ theta)
        gradient = (X.T @ (probabilities - y)) / len(y)
        theta = theta - alpha * gradient
        costs.append(compute_cost(X, y, theta))

    return theta, np.asarray(costs)


def predict_probability(X, theta):
    return sigmoid(X @ theta)


def predict(X, theta, threshold=0.5):
    return (predict_probability(X, theta) >= threshold).astype(int)


def standardize_features(X, mean=None, std=None):
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std


def make_set_q(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[1] != 6:
        raise ValueError("Set Q requires exactly six raw measurement features.")

    return np.column_stack((X, X ** 2))
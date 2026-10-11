# AI4003 Assignment 2: Transmission-Line Fault Detection

This project implements a small fault-detection workflow for the provided three-phase transmission-line dataset. It trains a manually implemented logistic regression model, evaluates SVM alternatives, compares the final models on a held-out test split, and provides a Streamlit dashboard for inference.

## Environment and dependencies

Use the project virtual environment located at:

`d:\SEMESTER 07\Applied Machine Learning\AI4003_Assignment_2\AI4003_Assignment_2\.venv\Scripts\python.exe`

If the environment needs to be recreated, use the following command from the project root:

```powershell
& "C:\Users\ZAKKI BOSS TRADERS\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m venv .venv
```

Install the required Python packages in the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scikit-learn streamlit joblib
```

## Project structure

- `data/` — raw dataset files.
- `src/` — data preparation, EDA, training, and evaluation scripts.
- `app/` — Streamlit interface for live inference.
- `results/figures/` — saved EDA, logistic-cost, and SVM plots.
- `results/tables/` — CSV tables summarizing preprocessing, logistic results, SVM validation, and test performance.

## Execution order

Run the scripts in this order from the project root:

```powershell
.\.venv\Scripts\python.exe src\prepare.py
.\.venv\Scripts\python.exe src\train_logistic.py
.\.venv\Scripts\python.exe src\train_svm.py
.\.venv\Scripts\python.exe src\test_models.py
```

## Launch the Streamlit application

From the project root, run:

```powershell
.\.venv\Scripts\streamlit.exe run app\app.py
```

The app uses the same preprocessing and feature ordering used during model training so that predictions remain consistent with the saved model artifacts.

### Input measurement units

Enter `Ia`, `Ib`, and `Ic` in the raw units used to produce the dataset. The CSV does not include metadata identifying the physical current unit, so do not assume that the values are amperes or kiloamperes. Enter `Va`, `Vb`, and `Vc` as per-unit (p.u.) values, as specified by the assignment. The app does not convert units before inference.

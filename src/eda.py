
from pathlib import Path
import matplotlib.pyplot as plt
from load_data import load_raw

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def main():
    df = load_raw()

    if "Output (S)" in df.columns:
        df = df.rename(columns={"Output (S)": "fault"})

    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df[["Ia", "Ib", "Ic", "Va", "Vb", "Vc", "fault"]].dropna()
    df["fault"] = df["fault"].astype(int)

    # Graph 1: First 2000 current samples with fault regions shaded
    part = df.iloc[:2000].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in ["Ia", "Ib", "Ic"]:
        ax.plot(part.index, part[col], label=col, linewidth=0.8)

    fault = part["fault"].to_numpy()
    start = None

    for i, value in enumerate(fault):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            ax.axvspan(start, i - 1, alpha=0.15)
            start = None

    if start is not None:
        ax.axvspan(start, len(fault) - 1, alpha=0.15)

    ax.set_title("Three-Phase Currents: First 2000 Samples")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Current")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "current_timeseries.png", dpi=160)
    plt.close(fig)

    # Graph 2: Ia versus Ib by fault class
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, name in [(0, "No fault"), (1, "Fault")]:
        subset = df[df["fault"] == label]
        ax.scatter(
            subset["Ia"], subset["Ib"],
            s=10, alpha=0.45, label=name
        )

    ax.set_title("Ia vs Ib by Fault Class")
    ax.set_xlabel("Ia")
    ax.set_ylabel("Ib")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "ia_ib_scatter.png", dpi=160)
    plt.close(fig)

    # Graph 3: Ia distributions for each class
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        df.loc[df["fault"] == 0, "Ia"],
        bins=50, alpha=0.6, label="No fault"
    )
    ax.hist(
        df.loc[df["fault"] == 1, "Ia"],
        bins=50, alpha=0.6, label="Fault"
    )
    ax.set_title("Ia Distribution by Fault Class")
    ax.set_xlabel("Ia")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "ia_histograms.png", dpi=160)
    plt.close(fig)

    print("Saved three EDA plots to:", FIGURES)


if __name__ == "__main__":
    main()
"""Figures from the saved results tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES, FIG = ROOT / "results", ROOT / "figures"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    b = pd.read_csv(RES / "by_snsp_band.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(range(len(b)), b["wind_capture"], color="#2a6f97", label="wind capture price")
    ax.plot(range(len(b)), b["mean_price"], "o-", color="#c1121f", label="mean price")
    ax.set_xticks(range(len(b)), b.index, rotation=20)
    ax.set_xlabel("SNSP band (%, reconstructed)"); ax.set_ylabel("EUR/MWh")
    ax.set_title("IE(SEM) 2023-2026: prices fall as non-synchronous penetration rises"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "snsp_bands.png", dpi=150); plt.close(fig)
    v = pd.read_csv(RES / "ccgt_value_of_forecast.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(v.index, v["share_of_perfect_%"], color="#2a6f97")
    for i, x in enumerate(v["share_of_perfect_%"]):
        ax.text(i, x + 0.8, f"{x:.1f}", ha="center")
    ax.set_ylim(0, 112); ax.set_ylabel("% of perfect-foresight profit")
    ax.set_title("Stylised 400 MW CCGT, 2025: value of the price forecast"); fig.tight_layout()
    fig.savefig(FIG / "ccgt_value.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()

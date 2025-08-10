import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

# ------------------------------------------------------------------
# ① One‑time Nature-ish styling
# ------------------------------------------------------------------
def setup_nature():
    mpl.rcParams.update({
        # ---------- Fonts ----------
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,  # global body text size

        # ---------- Figure ----------
        "figure.figsize": (3, 2.2),  # ~90 mm × 56 mm
        "figure.dpi": 300,

        # ---------- Axes ----------
        "axes.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.labelpad": 2,

        # ---------- Ticks ----------
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # ---------- Legend ----------
        "legend.frameon": False,
        "legend.fontsize": 6,

        # ---------- Misc ----------
        "savefig.bbox": "tight",
    })

# ② Color palette (colorblind‑friendly)
_palette = {
    "GT":   "#3778C2",  # blue
    "PCAD": "#F17CB0",  # pink
    "DRF":  "#9370DB",  # purple
    "DNN":  "#F49C44",  # orange
}

# ------------------------------------------------------------------
# ③ Plotting function
# ------------------------------------------------------------------
def plot_1_fig(gt, pred_pcad, pred_drf, pred_dnn, scenario_id: str, event_id: str, dt: float = 0.1):
    """
    Draw a single comparison panel (Nature single‑column style).

    Parameters
    ----------
    gt, pred_* : 1‑D array‑like
        Time series of perceived risk and model predictions.
    scenario_id, event_id : str
        Used in output filename.
    """
    gt = np.asarray(gt)
    pred_pcad = np.asarray(pred_pcad)
    pred_drf = np.asarray(pred_drf)
    pred_dnn = np.asarray(pred_dnn)

    # Align lengths if needed (truncate to the shortest to avoid shape errors)
    n = min(len(gt), len(pred_pcad), len(pred_drf), len(pred_dnn))
    gt, pred_pcad, pred_drf, pred_dnn = gt[:n], pred_pcad[:n], pred_drf[:n], pred_dnn[:n]

    fig, ax = plt.subplots()

    # Generate time axis: start at 0.1 s, step 0.1 s
    x = np.arange(1, n + 1) * dt

    ax.plot(x, gt, label="Ground truth", lw=1.0, color=_palette["GT"])
    ax.plot(x, pred_pcad, label="PCAD", lw=0.9, color=_palette["PCAD"])
    ax.plot(x, pred_drf, label="DRF", lw=0.9, color=_palette["DRF"])
    ax.plot(x, pred_dnn, label="DNN", lw=0.9, color=_palette["DNN"])

    # Absolute errors against GT on a secondary y‑axis
    error_pcad = np.abs(gt - pred_pcad)
    error_drf = np.abs(gt - pred_drf)
    error_dnn = np.abs(gt - pred_dnn)

    ax2 = ax.twinx()
    ax2.plot(x, error_pcad, label="PCAD error", lw=0.5, color=_palette["PCAD"], linestyle="--")
    ax2.plot(x, error_drf, label="DRF error", lw=0.5, color=_palette["DRF"], linestyle="--")
    ax2.plot(x, error_dnn, label="DNN error", lw=0.5, color=_palette["DNN"], linestyle="--")

    ax2.set_ylim(0, 10)
    ax2.set_ylabel("Prediction error")

    ax.set_xlabel("Time / s")
    ax.set_ylabel("Perceived risk")

    # Merge legends from both axes (deduplicate while preserving order)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    uniq = dict(zip(l1 + l2, h1 + h2))  # Python ≥3.7 preserves insertion order
    handles, labels = uniq.values(), uniq.keys()
    ax.legend(handles, labels, loc="upper right", ncol=1, handlelength=1, handletextpad=0.4,
              columnspacing=0.8, frameon=False)

    # Axes limits
    ax.set_xlim(0, n * dt)
    ax.set_ylim(-4, 10)

    plt.tight_layout()
    outdir = Path("./outputs_ts")
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"time_series_error_{scenario_id}_{event_id}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    return fig, ax

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    setup_nature()

    parser = argparse.ArgumentParser(description="Plot GT vs model predictions and absolute errors.")
    parser.add_argument("--scenario", nargs="+", default=["SVM"],
                        help="One or more scenario IDs, e.g., SVM HB MB LC")
    parser.add_argument("--events", type=int, default=27, help="Number of events (sheets) per scenario")
    parser.add_argument("--sheet-template", default="{scn}{i}",
                        help="Sheet name pattern, default '{scn}{i}' -> e.g., SVM1..SVM27")
    parser.add_argument("--gt-col", type=int, default=2, help="0-based column index of Ground Truth")
    parser.add_argument("--pcad-col", type=int, default=6, help="0-based column index of PCAD prediction")
    parser.add_argument("--drf-col", type=int, default=10, help="0-based column index of DRF prediction")
    parser.add_argument("--dnn-col", type=int, default=14, help="0-based column index of DNN prediction")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step in seconds (default: 0.1)")
    args = parser.parse_args()

    for scn in args.scenario:
        file_path = HERE / "raw_data" / f"{scn}_DRF.xlsx"
        if not file_path.exists():
            print(f"[WARN] Excel not found for {scn}: {file_path}")
            continue

        print(f"[INFO] Processing {scn} from {file_path}")
        for event_id in range(1, args.events + 1):
            sheet = args.sheet_template.format(scn=scn, i=event_id)
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
            except Exception as e:
                print(f"[WARN] Skip {scn}#{event_id}: cannot read sheet '{sheet}' ({e})")
                continue

            gt = pd.to_numeric(df.iloc[:, args.gt_col], errors="coerce").to_numpy()
            pred_pcad = pd.to_numeric(df.iloc[:, args.pcad_col], errors="coerce").to_numpy()
            pred_drf = pd.to_numeric(df.iloc[:, args.drf_col], errors="coerce").to_numpy()
            pred_dnn = pd.to_numeric(df.iloc[:, args.dnn_col], errors="coerce").to_numpy()

            plot_1_fig(gt, pred_pcad, pred_drf, pred_dnn, scn, event_id, dt=args.dt)

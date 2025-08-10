# -*- coding: utf-8 -*-
"""
boxplot & density (KDE) for 3-column RMSE data per Excel sheet.
Each sheet should have its first three columns as PCAD / DRF / DNN.
Outputs: <scene>_boxplot.pdf and <scene>_density.pdf under ./outputs_distribution
"""
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

# Anchor all paths to the script directory so running CWD does not matter
HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# 1) One-time Nature-ish styling
# ----------------------------------------------------------------------
def setup_nature():
    mpl.rcParams.update({
        # ---------- Fonts ----------
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
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
        # ---------- Legend ----------
        "legend.frameon": False,
        "legend.fontsize": 7,
        # ---------- Misc ----------
        "savefig.bbox": "tight",
    })

    # Make seaborn inherit the thin-frame style
    sns.set_style("ticks", {
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": 0.6,
    })


# ----------------------------------------------------------------------
# 2) Palette (colorblind-friendly, high contrast)
# ----------------------------------------------------------------------
_PALETTE = ["#F17CB0", "#9370DB", "#F49C44"]  # pink / purple / orange
_LABELS = ["PCAD", "DRF", "DNN"]              # match the first 3 columns


def plot_error_data(df_raw: pd.DataFrame, scene: str, outdir: Path, xlim: float = 6.0, fill: bool = True):
    """Plot a boxplot and a KDE density curve for one scene.

    Parameters
    ----------
    df_raw : DataFrame
        First three columns must be PCAD / DRF / DNN RMSE.
    scene : str
        Scene name used in output filenames.
    outdir : Path
        Output directory (will be created if it does not exist).
    xlim : float
        X-axis upper limit for density plot.
    fill : bool
        Whether to fill the KDE curves.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure the first 3 columns are numeric and labeled
    df = df_raw.copy()
    for i in range(min(3, df.shape[1])):
        df.iloc[:, i] = pd.to_numeric(df.iloc[:, i], errors="coerce")
    df.columns = _LABELS + list(df.columns[len(_LABELS):])  # only rename first three

    cols = _LABELS  # ["PCAD", "DRF", "DNN"]

    # -------- A) Boxplot --------
    fig, ax = plt.subplots()
    sns.boxplot(
        data=df,
        order=cols,
        palette=_PALETTE,
        width=0.3,
        showfliers=False,
        linewidth=0.6,
        ax=ax,
    )
    # Harmonize edge/fill colors and alpha
    for i, patch in enumerate(ax.artists[: len(cols)]):
        c = _PALETTE[i]
        patch.set_edgecolor(c)
        patch.set_facecolor(mpl.colors.to_rgba(c, 0.45))
        patch.set_linewidth(0.6)

    ax.set_ylabel("Error (RMSE)")
    ax.set_xticklabels(cols)
    ax.legend(
        handles=[Line2D([0], [0], color=_PALETTE[i], lw=1.5, label=cols[i]) for i in range(3)],
        loc="upper right",
    )
    plt.tight_layout()
    outpath = outdir / f"{scene}_boxplot.pdf"
    fig.savefig(outpath)
    print(f"Saved: {outpath}")
    plt.close(fig)

    # -------- B) Density (KDE) --------
    fig, ax = plt.subplots()
    for label, color in zip(cols, _PALETTE):
        sns.kdeplot(
            df[label],
            bw_adjust=0.5,
            label=label,
            color=color,
            lw=0.6,
            clip=(0, None),
            fill=fill,
            ax=ax,
        )
    ax.set_xlabel("Error (RMSE)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, xlim)
    ax.legend(loc="upper right")
    plt.tight_layout()
    outpath = outdir / f"{scene}_density.pdf"
    fig.savefig(outpath)
    print(f"Saved: {outpath}")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3) Load Excel and plot per-scene
# ----------------------------------------------------------------------
def main():
    setup_nature()

    parser = argparse.ArgumentParser(description="Plot RMSE distributions for PCAD/DRF/DNN from Excel.")
    parser.add_argument("--excel", type=str, default=str(HERE / "raw_data" / "error_data.xlsx"),
                        help="Path to the Excel file (default: ./raw_data/error_data.xlsx relative to this script)")
    parser.add_argument("--scenes", nargs="+", default=["MB", "HB", "LC", "SVM"],
                        help="Scene names to render (must match sheet names in the Excel)")
    parser.add_argument("--xlim", type=float, default=6.0, help="X-axis max for the KDE plot (default: 6)")
    parser.add_argument("--no-fill", action="store_true", help="Do not fill the KDE curves")
    args = parser.parse_args()

    path_excel = Path(args.excel)
    if not path_excel.exists():
        raise FileNotFoundError(f"Excel not found: {path_excel}")

    # Load all sheets once
    df_dict = pd.read_excel(path_excel, sheet_name=None)

    outdir = HERE / "outputs_distribution"
    outdir.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        # Prefer exact sheet name match; warn if missing
        if scene not in df_dict:
            print(f"[WARN] Sheet '{scene}' not found in {path_excel.name}. Available: {list(df_dict.keys())}")
            continue
        print(f"[INFO] Plotting scene '{scene}' from sheet '{scene}'")
        plot_error_data(df_dict[scene], scene, outdir=outdir, xlim=args.xlim, fill=(not args.no_fill))


if __name__ == "__main__":
    main()
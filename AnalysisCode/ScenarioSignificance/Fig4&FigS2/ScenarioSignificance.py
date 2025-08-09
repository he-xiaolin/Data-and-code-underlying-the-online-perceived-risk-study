import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations
from matplotlib.patches import Patch  # for legend handles

# ------------------------------------------------------------
# 0. Configure unified plotting style for journals
# ------------------------------------------------------------
_PALETTE = {
    "1": "#4C72B0",  # calm blue
    "2": "#D55E00",  # vivid orange
    "3": "#009E73",  # fresh teal
    "4": "#CC79A7",  # soft magenta
}

# ------------------------------------------------------------
# 1. Define all scenario-feature configurations
#    Each tuple: (scenario_id, feature_id, y_base, legend_elems, cond_list, offset)
# ------------------------------------------------------------
CONFIGS = [
    # HB scenarios (3 conditions)
    ("HB", "BI",
     {1:10.5, 2:10.5, 3:10.5, 4:8.5, 5:3.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label=r"$-2\,m/s^2$", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label=r"$-5\,m/s^2$", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label=r"$-8\,m/s^2$", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    ("HB", "distance",
     {1:10.5, 2:10.5, 3:10.5, 4:8.5, 5:3.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="5 m", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="15 m", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="25 m", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    ("HB", "speed",
     {1:10.5, 2:10.5, 3:10.5, 4:8.5, 5:3.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="80 m/s", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="100 m/s", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="120 m/s", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    # MB scenarios (3 conditions)
    ("MB", "BI",
     {1:3.5, 2:8.0, 3:10.5, 4:8.5, 5:3.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label=r"$-2\,m/s^2$", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label=r"$-5\,m/s^2$", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label=r"$-8\,m/s^2$", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    ("MB", "distance",
     {1:3.5, 2:9.5, 3:10.5, 4:9.5, 5:3.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="5 m", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="15 m", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="25 m", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    ("MB", "speed",
     {1:3.5, 2:5.0, 3:5.0, 4:8.5, 5:3.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="80 m/s", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="100 m/s", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="120 m/s", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    # SVM scenarios (3 conditions)
    ("SVM", "BI",
     {1:3.5, 2:8.0, 3:8.5, 4:9.5, 5:5.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label=r"$-2\,m/s^2$", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label=r"$-5\,m/s^2$", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label=r"$-8\,m/s^2$", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    ("SVM", "distance",
     {1:5.5, 2:5.5, 3:9.5, 4:9.5, 5:5.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="5 m", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="15 m", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="25 m", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    ("SVM", "speed",
     {1:2.5, 2:5.5, 3:9.5, 4:7.5, 5:2.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="80 m/s", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="100 m/s", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="120 m/s", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    # LC scenarios
    ("LC", "Distance",
     {1:10.5, 2:10.5, 3:10.5, 4:10.5, 5:9.5, 6:6.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="5 m",  alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="25 m", alpha=0.8),
     ],
     ["1", "2"],
     {"1": -0.18, "2": 0.18}),

    ("LC", "Driving style",
     {1:10.5, 2:10.5, 3:10.5, 4:10.5, 5:9.5, 6:5.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="Cautious",   alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="Mild",       alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="Aggressive", alpha=0.8),
     ],
     ["1", "2", "3"],
     {"1": -0.20, "2": 0.0, "3": 0.20}),

    ("LC", "Lateral behaviour",
     {1:9.5, 2:9.0, 3:9.0, 4:9.5, 5:9.5, 6:9.5},
     [
         Patch(facecolor=_PALETTE["1"], edgecolor=".3", label="Lat vel (1 m/s)", alpha=0.8),
         Patch(facecolor=_PALETTE["2"], edgecolor=".3", label="Lat vel (3 m/s)", alpha=0.8),
         Patch(facecolor=_PALETTE["3"], edgecolor=".3", label="Frag. lane ch.", alpha=0.8),
         Patch(facecolor=_PALETTE["4"], edgecolor=".3", label="Abort lane ch.", alpha=0.8),
     ],
     ["1", "2", "3", "4"],
     {"1": -0.20, "2": -0.10, "3": 0.10, "4": 0.20}),
]

# ------------------------------------------------------------
# Utility to draw significance brackets
# ------------------------------------------------------------
def _add_sig(ax, x1, x2, y, txt, lw=0.45, star_pad=0.30):
    ax.plot([x1, x1, x2, x2], [y, y+dy_line, y+dy_line, y], lw=lw, c="k", clip_on=False)
    ax.text((x1+x2)/2, y + dy_line - star_pad, txt, ha="center", va="bottom", fontsize=8)

# ------------------------------------------------------------
# Apply publication-style rcParams
# ------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "figure.dpi": 300,
    "figure.figsize": (5, 3.0),
    "axes.linewidth": 0.6,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
})

sns.set_theme(style="ticks")

# ------------------------------------------------------------
# Main: loop over CONFIGS and produce each plot
# ------------------------------------------------------------
if __name__ == '__main__':
    for scenario_id, feature_id, y_base, legend_elems, cond_list, offset in CONFIGS:
        # 1) Load and tidy data
        dfs = []
        max_clip = max(y_base.keys())
        for clip in range(1, max_clip + 1):
            for cond in cond_list:
                fp = Path("./process_data") / scenario_id / f"{feature_id}_clip_{clip}_condition_{cond}.csv"
                df = pd.read_csv(fp)
                if "value" not in df.columns:
                    if df.shape[1] == 1:
                        df.columns = ["value"]
                    else:
                        num_cols = df.select_dtypes(include="number").columns
                        df = df.rename(columns={num_cols[0]: "value"})
                df = df[["value"]].copy()
                df["clip"] = clip
                df["condition"] = cond
                dfs.append(df)
        tidy = pd.concat(dfs, ignore_index=True)

        # 2) Compute Welch's t-test p-values
        pvals = {}
        for clip in tidy["clip"].unique():
            group = tidy[tidy["clip"] == clip]
            for a, b in combinations(cond_list, 2):
                _, p = stats.ttest_ind(
                    group[group["condition"] == a]["value"],
                    group[group["condition"] == b]["value"],
                    equal_var=False)
                pvals[(clip, a, b)] = p

        # 3) Plotting
        fig, ax = plt.subplots()
        sns.boxplot(x="clip", y="value", hue="condition", data=tidy, ax=ax,
                    palette=_PALETTE, width=0.5, linewidth=0.6, dodge=True,
                    boxprops={"alpha": 0.8, "edgecolor": ".3"},
                    whiskerprops={"linewidth": .6}, capprops={"linewidth": .6},
                    medianprops={"linewidth": .8, "color": ".15"},
                    showcaps=True, showfliers=False, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": ".2", "markersize": 1,
                               "markeredgewidth": 0.3})
        ax.legend(handles=legend_elems, frameon=True, fontsize=7, title_fontsize=7, loc="best")

        dy_line = 0.2
        step_between = 0.8
        for clip in sorted(tidy["clip"].unique()):
            star_count = 0
            base_y = y_base[clip]
            for a, b in combinations(cond_list, 2):
                p = pvals[(clip, a, b)]
                if p < 0.05:
                    star_count += 1
                    stars = "***" if p < .001 else "**" if p < .01 else "*"
                    x1 = clip + offset[a] - 1
                    x2 = clip + offset[b] - 1
                    _add_sig(ax, x1, x2, base_y + star_count * step_between, stars)

        ax.set_xlabel("Clip", labelpad=2)
        ax.set_ylabel("Perceived risk", labelpad=2)
        ax.set_ylim(-1, 10)
        sns.despine(trim=True)
        plt.tight_layout()

        outdir = Path("./outputs")
        outdir.mkdir(parents=True, exist_ok=True)
        filename = f"correlation_check_{scenario_id}_{feature_id}.pdf"
        fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

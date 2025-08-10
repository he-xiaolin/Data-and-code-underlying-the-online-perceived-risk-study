# -*- coding: utf-8 -*-
"""
Nature‑style boxplot & histogram for 3‑column RMSE data in each Excel sheet
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path


# ----------------------------------------------------------------------
# 1. 一次性设置 Nature 统一 rcParams
# ----------------------------------------------------------------------
def setup_nature():
    mpl.rcParams.update({
        # ---------- 字体 ----------
        "font.family"      : "sans-serif",
        "font.sans-serif"  : ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset" : "dejavusans",
        "font.size"        : 8,           # 正文 8 pt
        # ---------- 画布 ----------
        "figure.figsize"   : (3, 2.2), # 90 mm × 56 mm
        "figure.dpi"       : 300,
        # ---------- 坐标轴 ----------
        "axes.linewidth"   : .6,
        "axes.spines.top"  : True,
        "axes.spines.right": True,
        "axes.labelpad"    : 2,
        # ---------- 刻度 ----------
        "xtick.direction"  : "in",
        "ytick.direction"  : "in",
        "xtick.major.width": .6,
        "ytick.major.width": .6,
        "xtick.major.size" : 2.5,
        "ytick.major.size" : 2.5,
        # ---------- 图例 ----------
        "legend.frameon"   : False,
        "legend.fontsize"  : 7,
        # ---------- 其他 ----------
        "savefig.bbox"     : "tight",
    })

    # 让 seaborn 继承细框风格
    sns.set_style("ticks", {
        "axes.spines.top":    True,
        "axes.spines.right":  True,
        "axes.linewidth":     .6,
    })


# ----------------------------------------------------------------------
# 2. 色板（色盲安全，对比度高）
# ----------------------------------------------------------------------
_PALETTE = ["#F17CB0", "#9370DB", "#F49C44"]  # 蓝 / 粉 / 紫
_LABELS  = ["PCAD", "DRF", "DNN"]             # 与列顺序保持一致




def plot_error_data(df_raw, scene):
    """
    Parameters
    ----------
    df_raw : pd.DataFrame
        前 3 列依次为 PCAD / DRF / DNN 的 RMSE
    """
    # ---- 给缺省列名的数据补列名 ----
    df = df_raw.copy()
    df.columns = _LABELS + list(df.columns[len(_LABELS):])  # 只改前 3 列名

    cols = _LABELS                      # ["PCAD", "DRF", "DNN"]

    # -------- A. 箱形图 --------
    fig, ax = plt.subplots()
    sns.boxplot(
        data=df,
        order=cols,                     # 强制顺序
        palette=_PALETTE,
        width=0.3,
        showfliers=False,
        linewidth=0.6,
        ax=ax
    )
    # 统一边缘/填充颜色、透明度
    for i, patch in enumerate(ax.artists):
        c = _PALETTE[i]
        patch.set_edgecolor(c)
        patch.set_facecolor(mpl.colors.to_rgba(c, 0.45))
        patch.set_linewidth(0.6)

    ax.set_ylabel("Error")
    ax.set_xticklabels(cols)
    ax.legend(
        handles=[Line2D([0], [0], color=_PALETTE[i], lw=1.5, label=cols[i])
                 for i in range(3)],
        loc="upper right"
    )
    plt.tight_layout()
    fig.savefig(f"./outputs_distribution/{scene}_boxplot.pdf")
    plt.close(fig)

    # -------- B. 仅密度曲线 --------
    fig, ax = plt.subplots()
    for label, color in zip(cols, _PALETTE):
        sns.kdeplot(
            df[label],
            bw_adjust=0.5,
            label=label,
            color=color,
            lw=0.6,
            clip=(0, None),
            fill=True,          # 填充可改为 False
            ax=ax
        )
    ax.set_xlabel("Error")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(f"./outputs_distribution/{scene}_density.pdf")
    plt.close(fig)



# ----------------------------------------------------------------------
# 4. 批量读取 Excel 并绘图
# ----------------------------------------------------------------------
def main():
    setup_nature()

    path_excel = Path("./raw_data/error_data.xlsx")
    df_dict    = pd.read_excel(path_excel, sheet_name=None)  # 全部表
    scene_list = ["MB", "HB", "LC", "SVM"]

    for scene, (sheet, df) in zip(scene_list, df_dict.items()):
        print(f"绘制 {sheet} → {scene}")
        plot_error_data(df, scene)


if __name__ == "__main__":
    main()
# 真值 PCAD DRF DNN
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# ① 一次性设置 Nature 样式
# ------------------------------------------------------------------
def setup_nature():
    mpl.rcParams.update({
        # ---------- 字体 ----------
        "font.family"      : "sans-serif",
        "font.sans-serif"  : ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset" : "dejavusans",
        "font.size"        : 8,          # 全局正文字号

        # ---------- 画布 ----------
        "figure.figsize"   : (3, 2.2),# 90 mm × 56 mm
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
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # ---------- 图例 ----------
        "legend.frameon"   : False,
        "legend.fontsize"  : 6,

        # ---------- 其他 ----------
        "savefig.bbox"     : "tight",
    })

# ② 调色盘（色盲安全）
_palette = {
    "GT"  : "#3778C2",  # 蓝色
    "PCAD": "#F17CB0",  # 粉色
    "DRF" : "#9370DB",  # 紫色
    "DNN" : "#F49C44",  # 橙色
}

# ------------------------------------------------------------------
# ③ 绘图函数
# ------------------------------------------------------------------
def plot_1_fig(gt, pred_pcad, pred_drf, pred_dnn,
               scenario_id: str, event_id: str):
    """
    画单幅对比曲线图（Nature 单栏风格）
    Parameters
    ----------
    gt, pred_* : 1‑D array‑like
        序列数据
    scenario_id, event_id : str
        用于标题
    """
    fig, ax = plt.subplots()
    # generate x values, start from 0 , delta = 0.1
    x = np.arange(0.1, len(gt)*0.1+0.1, 0.1)   

    ax.plot(x, gt, label="Ground truth", lw=1.0,
            color=_palette["GT"])
    ax.plot(x, pred_pcad, label="PCAD",  lw=0.9,
            color=_palette["PCAD"])
    ax.plot(x, pred_drf,  label="DRF",   lw=0.9,
            color=_palette["DRF"])
    ax.plot(x, pred_dnn,  label="DNN",   lw=0.9,
            color=_palette["DNN"])
    
    # calculate the error between gt and pred
    error_pcad = np.abs(gt - pred_pcad)
    error_drf  = np.abs(gt - pred_drf)
    error_dnn  = np.abs(gt - pred_dnn)

    # plot the error in the second y axis
    ax2 = ax.twinx()
    ax2.plot(x, error_pcad, label="PCAD error", lw=0.5,
             color=_palette["PCAD"], linestyle='--')
    ax2.plot(x, error_drf,  label="DRF error",  lw=0.5,
             color=_palette["DRF"], linestyle='--')
    ax2.plot(x, error_dnn,  label="DNN error",  lw=0.5,
             color=_palette["DNN"], linestyle='--')
    # 设置第二个 y 轴的范围
    ax2.set_ylim(0, 10)
    # 设置第二个 y 轴的标签
    ax2.set_ylabel("Prediction error")

    # 轴标签
    ax.set_xlabel("Time/s")
    ax.set_ylabel("Perceived risk")



    # 先分别取出 handle 和 label
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    # 若怕 label 重复，可用 dict 去重（保留顺序）
    uniq = dict(zip(l1 + l2, h1 + h2))        # Python ≥3.7 顺序稳定
    handles, labels = uniq.values(), uniq.keys()

    # 重新绘制统一图例（放在主轴或整张图都可）
    ax.legend(handles, labels,
              loc="upper right", ncol=1,     # 或 ncol=1 按需调整
              handlelength=1, handletextpad=0.4,
              columnspacing=0.8, frameon=False)




    # set x, y 轴范围
    ax.set_xlim(0, 30)
    ax.set_ylim(-4, 10)
    plt.tight_layout()        # 去多余留白
    plt.savefig(f"./outputs_ts/time_series_error_{scenario_id}_{event_id}.pdf",dpi=300, bbox_inches="tight")
    plt.close()
    return fig, ax

# ------------------------------------------------------------------
# ④ 典型用法示例
# ------------------------------------------------------------------
if __name__ == "__main__":
    setup_nature()  # 只需调用一次
    # 1. 拼出文件名
    scenario_id  = "SVM"
    file_path = Path(f"./raw_data/{scenario_id}_DRF.xlsx")
    event_nb = 27


    dfs = pd.read_excel(
        file_path,
        sheet_name=list(range(event_nb))         # 读索引 0‑2 这 3 张表
    )


    # the sheet name is the like HB1, please loop for it from HB1 to HB27
    # 2. 读取数据
    for event_id in range(1, event_nb+1):
        # 读取数据
        df = pd.read_excel(file_path, sheet_name=f"{scenario_id}{event_id}")
        # extract the second column and convert to numpy array
        gt = df.iloc[:, 2].to_numpy()
        pred_pcad = df.iloc[:, 6].to_numpy()
        pred_drf  = df.iloc[:, 10].to_numpy()
        pred_DNN  = df.iloc[:, 14].to_numpy()

     
        fig, ax = plot_1_fig(gt, pred_pcad, pred_drf, pred_DNN, scenario_id, event_id)



    # plt.show()
    

    





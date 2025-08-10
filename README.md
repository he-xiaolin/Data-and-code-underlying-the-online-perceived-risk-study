# Data & Code for the Online Perceived Risk Study

This repository contains data and code underlying the publication: `Reading minds on the road: decoding perceived risk in automated vehicles through 140k+ ratings`. It includes:

- **Statistical analysis** of subjective ratings (per-clip boxplots with significance marks).
- **Perceived risk prediction** using physics-based models (PCAD, DRF) and deep neurual networks (DNN).
- **Model interpretation** with SHAP (global & local views).
- **Result visualizations** for time-series comparisons and error distributions.

> **Scenarios**: `HB`, `LC` (split as `LC_1`, `LC_2`, `LC_3` where applicable), `MB`, and `SVM`.

---

## 1) Repository layout (key folders)

Your exact paths may differ slightly; the layout below reflects the intended organization.

```
AnalysisCode/
├─ ScenarioSignificance/                   # Stats & significance on subjective ratings
│  ├─ data_process.py                      # Split raw Excel into per-clip CSVs
│  ├─ ScenarioSignificance.py              # Welch t-tests per clip; export p-values/annotations
│  └─ README.md                            # Folder-level documentation (how to run)
│
├─ PerceivedRiskPrediction/
│  ├─ ModelCalibration_Training/
│  │  └─ DNN_Training/
│  │     ├─ step_2_NN_train/
│  │     │  ├─ modules/                    # MC-Dropout model, training/eval utilities
│  │     │  ├─ data/                       # *Generated* .npy features (e.g., HB_feature_reg.npy)
│  │     │  └─ models/<exp>/<SCN>/         # Trained weights: best_model_<SCN>.pth
│  │     └─ (legacy) step_3_shap/          # Original SHAP location (now moved to AnalysisCode/SHAP)
│  │
│  └─ PerceivedRiskPredictionResults/
│     ├─ raw_data/                         # Per-scenario Excel for plotting (e.g., SVM_DRF.xlsx)
│     ├─ outputs_ts/                       # Time-series comparison PDFs
│     ├─ outputs_distribution/             # Error distribution PDFs
│     ├─ time_series_error_comparison.py   # GT vs PCAD/DRF/DNN + |error| curves
│     └─ error_distribution_plot.py        # Boxplot + KDE for RMSE of the three models
│
└─ SHAP/                                   # SHAP analysis (current, recommended location)
   ├─ shap_HB.py
   ├─ shap_MB.py
   ├─ shap_SVM.py
   ├─ shap_LC_1.py
   ├─ shap_LC_2.py
   └─ shap_LC_3.py
```

**External data (example):**
```
Data/
├─ SubjectiveRatings_clip/                 # Raw subjective rating Excels (HB.xlsx, LC.xlsx, MB.xlsx, SVM.xlsx)
└─ parameter_list.xlsx                     # Scenario/parameter mapping used by the scripts
```

---

## 2) Environment

- **Python**: 3.9–3.11  
- **Recommended packages**: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `openpyxl`, `torch`, `shap`, `tqdm`

Install (pip):
```bash
pip install numpy pandas matplotlib seaborn scipy openpyxl torch shap tqdm
```

> **MATLAB** is needed if you (re)run PCAD/DRF calibration or MATLAB-based helpers.

---

## 3) Subjective ratings → tidy CSVs → significance

Scripts under `AnalysisCode/ScenarioSignificance/`.

### 3.1 Split raw Excel into per-clip CSVs — `data_process.py`

Converts each scenario Excel (e.g., `HB.xlsx`, `LC.xlsx`) into per-clip per-condition CSV files (a single column named `value`):
```
AnalysisCode/ScenarioSignificance/process_data/<SCENARIO>/<FEATURE>_clip_<i>_condition_<j>.csv
```
Features are sheets (e.g., `BI`, `distance`, `speed`; LC uses `Distance`, `Driving style`, `Lateral behaviour`). Adjust the scenario block or use the multi-scenario loop version.

### 3.2 Welch t-tests per clip — `ScenarioSignificance.py`

Performs **pairwise Welch t-tests** between conditions **within each clip**, saving:
- `pvalues.csv` (columns: `clip`, `cond_A`, `cond_B`, `p_value`, `significant`)
- `annotations.json` (mapping to `""`, `"*"`, `"**"`, `"***"`)

These are used to place bracket/asterisk annotations on grouped boxplots.

---

## 4) DNN training & evaluation

Under `AnalysisCode/PerceivedRiskPrediction/ModelCalibration_Training/DNN_Training/step_2_NN_train/`:

- **Training**: `train_<SCENARIO>.py`  
  Args: `--num_epochs`, `--experiment_index` (used in `models/<exp>/<SCN>/`)
- **Evaluation**: `evaluation_<SCENARIO>.py`  
  Loads best weights and produces predictions/metrics.

**Conventions**
- Data: `<SCENARIO>_feature_reg.npy` in `step_2_NN_train/data/` (last column = label).
- Models: `step_2_NN_train/models/<experiment_index>/<SCENARIO>/best_model_<SCENARIO>.pth`.

Quick smoke test:
```bash
# from step_2_NN_train/
python train_MB.py --num_epochs 2 --experiment_index 2025
python evaluation_MB.py --experiment_index 2025
```

> Some legacy scripts used defaults like `2024` or `202404-complete_data`. For consistency, pass a single tag everywhere (e.g., `--experiment_index 2025`).

---

## 5) SHAP (global & local)

Scripts in `AnalysisCode/SHAP/`. Each script (`shap_<SCENARIO>.py`) loads the trained DNN and computes:

- **Global importance bar**: `shap_bar_all_random.pdf`
- **Local, per-event**:  
  - `shap_ts_heatmap_event<N>.pdf`  
  - `shap_lines_event<N>.pdf`  
  - `shap_waterfall_event<N>_t<T>.pdf`
- **Cache**: `shap_data_all_random.pkl` (reused if `K` unchanged)

Usage (from `AnalysisCode/SHAP/`):
```bash
python shap_SVM.py --scenario SVM --experiment_index 2025 --K 30 --event_id 1
python shap_SVM.py --scenario SVM --experiment_index 2025 --K 30 --per_event   # all events
```

Expected artifacts:
```
../PerceivedRiskPrediction/ModelCalibration_Training/DNN_Training/step_2_NN_train/models/<exp>/<SCENARIO>/best_model_<SCENARIO>.pth
../PerceivedRiskPrediction/ModelCalibration_Training/DNN_Training/step_2_NN_train/data/<SCENARIO>_feature_reg.npy
```

---

## 6) Result figures

### 6.1 Time-series comparison (GT vs predictions + |error|)

Script: `AnalysisCode/PerceivedRiskPrediction/PerceivedRiskPredictionResults/time_series_error_comparison.py`

- Reads per-scenario Excel (e.g., `SVM_DRF.xlsx`) from `PerceivedRiskPredictionResults/raw_data/`
- Expects sheet names like `SVM1..SVM27` (template is configurable)
- Default columns (0-based): GT=`2`, PCAD=`6`, DRF=`10`, DNN=`14`
- Saves to `PerceivedRiskPredictionResults/outputs_ts/`

Examples:
```bash
python time_series_error_comparison.py --scenario SVM --events 27 --dt 0.1
python time_series_error_comparison.py --scenario LC  --events 24 --dt 0.1
python time_series_error_comparison.py --scenario HB MB LC SVM --events 27 --dt 0.1
```

### 6.2 Error distributions (RMSE of PCAD/DRF/DNN)

Script: `AnalysisCode/PerceivedRiskPrediction/PerceivedRiskPredictionResults/error_distribution_plot.py`

- Reads `error_data.xlsx` from `PerceivedRiskPredictionResults/raw_data/` (or `--excel <path>`)
- Requires first three columns to be **PCAD / DRF / DNN**
- Matches **sheet names** to scenes (e.g., `MB`, `HB`, `LC`, `SVM`)
- Outputs: `<scene>_boxplot.pdf`, `<scene>_density.pdf` under `outputs_distribution/`

Example:
```bash
python error_distribution_plot.py --scenes MB HB LC SVM --xlim 6
```

---

## 7) Figure assembly (multi-panel)

Most multi-panel figures (e.g., Fig. 2–5) are composed by exporting **individual panels** (PDF) and assembling in `Figma`. Keep styling consistent (fonts, line widths, colors), and export as **PDF** to preserve vector quality.

---

## 8) Reproducibility

```python
import random, numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
```

- Log the `experiment_index` used for each trained model.
- Keep SHAP background size `K` consistent for comparability.
- Archive `requirements.txt`/`environment.yml` for camera-ready package.

---

## 9) License & citation

### License

- **Code**: MIT License (recommended). 

- **Data**: Licensed **as specified on the 4TU.ResearchData record** of the dataset below.  
  Please follow the license shown on the dataset landing page (DOI). If uncertain, check the DOI page or contact the corresponding author / repository support.

### How to cite

Please cite the dataset and (once available) the paper using the entries below.

**Dataset**
- He, Xiaolin; Li, Zirui; Wang, Xinwei; R. (Riender) Happee; Wang, Meng (2025): *Data and code underlying the online perceived risk study.* Version 2. 4TU.ResearchData. dataset. https://doi.org/10.4121/242d9474-e522-4518-8917-8f284fc8a7a8

**BibTeX (dataset)**
```bibtex
@dataset{he_li_wangxinwei_happee_wang_2025_online_perceived_risk,
  author    = {He, Xiaolin and Li, Zirui and Wang, Xinwei and Happee, Riender and Wang, Meng},
  title     = {Data and code underlying the online perceived risk study},
  year      = {2025},
  version   = {1},
  publisher = {4TU.ResearchData},
  doi       = {10.4121/242d9474-e522-4518-8917-8f284fc8a7a8},
  url       = {https://doi.org/10.4121/242d9474-e522-4518-8917-8f284fc8a7a8},
  type      = {dataset}
}
```

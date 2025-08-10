# AnalysisCode

This folder contains all scripts used to **process subjective ratings**, **train/evaluate a DNN perceived risk model**, **compute SHAP explanations**, and **render publication figures**.

> Scenarios covered across scripts: `HB`, `LC` (sometimes split as `LC_1`, `LC_2`, `LC_3`), `MB`, `SVM`.

---

## Folder map (key parts)

```
AnalysisCode/
├─ ScenarioSignificance/                   # Stats on subjective ratings (per-clip significance)
│  ├─ data_process.py                      # Split raw Excel → per-clip CSVs
│  ├─ ScenarioSignificance.py              # Welch t-tests per clip + annotations
│  └─ README.md                            # Folder-level how-to (optional)
│
├─ PerceivedRiskPrediction/
│  ├─ ModelCalibration_Training/
│  │  └─ DNN_Training/
│  │     ├─step_1_Feature_extraction
│  │     └─ step_2_NN_train/
│  │        ├─ modules/                    # MC-Dropout model + utilities
│  │        ├─ data/                       # *.npy features (e.g., SVM_feature_reg.npy)
│  │        └─ models/<exp>/<SCN>/         # best_model_<SCN>.pth and logs
│  │     
│  │
│  └─ PerceivedRiskPredictionResults/
│     ├─ raw_data/                         # Excel for plotting (e.g., SVM_DRF.xlsx, error_data.xlsx)
│     ├─ outputs_ts/                       # Time-series comparison PDFs (auto-created)
│     ├─ outputs_distribution/             # Error distribution PDFs (auto-created)
│     ├─ time_series_error_comparison.py   # GT vs PCAD/DRF/DNN + |error| curves
│     └─ error_distribution_plot.py        # RMSE boxplot + density per model
│
└─ SHAP/                                   # Current SHAP scripts (recommended location)
   ├─ shap_HB.py
   ├─ shap_MB.py
   ├─ shap_SVM.py
   ├─ shap_LC_1.py
   ├─ shap_LC_2.py
   └─ shap_LC_3.py
```

**External data (examples):**
```
Data/
├─ SubjectiveRatings_clip/                 # HB.xlsx, LC.xlsx, MB.xlsx, SVM.xlsx
└─ parameter_list.xlsx                     # Scenario/parameter mapping
```

---

## 1) Environment

- Python 3.9–3.11
- Install (pip):
  ```bash
  pip install numpy pandas matplotlib seaborn scipy openpyxl torch shap tqdm
  ```
- MATLAB only needed if (re)running PCAD/DRF calibration or MATLAB helper functions.

---

## 2) Subjective ratings → tidy CSVs → significance

### 2.1 Split raw Excel into per-clip CSVs — `ScenarioSignificance/data_process.py`

- **Input**: `Data/SubjectiveRatings_clip/<SCENARIO>.xlsx`  
  Sheets correspond to features (e.g., `BI`, `distance`, `speed`; for LC: `Distance`, `Driving style`, `Lateral behaviour`).
- **Output**: one-column CSVs named `value` under:
  ```
  AnalysisCode/ScenarioSignificance/process_data/<SCENARIO>/<FEATURE>_clip_<i>_condition_<j>.csv
  ```
- **Run** (edit the scenario block or use the multi-scenario loop inside the script):
  ```bash
  cd AnalysisCode/ScenarioSignificance
  python data_process.py
  ```

### 2.2 Pairwise Welch t-tests per clip — `ScenarioSignificance/ScenarioSignificance.py`

- Loads the per-clip CSVs for each feature and performs pairwise **Welch’s t-tests** across conditions **within each clip**.
- **Outputs** (used by plotting scripts to add asterisks/brackets):
  - `pvalues.csv` with columns: `clip, cond_A, cond_B, p_value, significant`
  - `annotations.json` mapping pairs to `""`, `"*"`, `"**"`, `"***"`

---

## 3) DNN training & evaluation

Path: `PerceivedRiskPrediction/ModelCalibration_Training/DNN_Training/step_2_NN_train/`

### 3.1 Train

- **Data**: `data/<SCENARIO>_feature_reg.npy` (**last column is label**).
- **Models**: `models/<EXPERIMENT_INDEX>/<SCENARIO>/best_model_<SCENARIO>.pth`

Run examples:
```bash
cd AnalysisCode/PerceivedRiskPrediction/ModelCalibration_Training/DNN_Training/step_2_NN_train

# quick smoke tests (2 epochs) — pick a unified experiment tag, e.g. 2025
python train_MB.py --num_epochs 2 --experiment_index 2025
python train_HB.py --num_epochs 2 --experiment_index 2025
python train_AMB.py --num_epochs 2 --experiment_index 2025   # AMB == SVM in older naming
```

### 3.2 Evaluate

```bash
python evaluation_MB.py --experiment_index 2025
python evaluation_HB.py --experiment_index 2025
# ... and so on for other scenarios
```

---

## 4) SHAP (global & local explanations)

Current scripts live in `AnalysisCode/SHAP/` (moved from legacy `step_3_shap/`).  
Each `shap_<SCENARIO>.py` expects:

- **Trained model**:
  ```
  ../PerceivedRiskPrediction/ModelCalibration_Training/DNN_Training/step_2_NN_train/models/<EXPERIMENT_INDEX>/<SCENARIO>/best_model_<SCENARIO>.pth
  ```
- **Feature data**:
  ```
  ../PerceivedRiskPrediction/ModelCalibration_Training/DNN_Training/step_2_NN_train/data/<SCENARIO>_feature_reg.npy
  ```

Examples:
```bash
cd AnalysisCode/SHAP

# Global bar + local (selected event)
python shap_SVM.py --scenario SVM --experiment_index 2025 --K 30 --event_id 1

# All events (creates per-event heatmaps/lines/waterfalls)
python shap_SVM.py --scenario SVM --experiment_index 2025 --K 30 --per_event
```

**Outputs (per scenario, under `AnalysisCode/SHAP/`)**  
- `shap_bar_all_random.pdf` (global)  
- `shap_ts_heatmap_event<N>.pdf`, `shap_lines_event<N>.pdf`, `shap_waterfall_event<N>_t<T>.pdf` (local)  
- `shap_data_all_random.pkl` (cache; reused if the same `K`)

---

## 5) Result figures

### 5.1 Time-series comparison — `PerceivedRiskPredictionResults/time_series_error_comparison.py`

- **Input**: per-scenario Excel in `PerceivedRiskPredictionResults/raw_data/` (e.g., `SVM_DRF.xlsx`)  
  Sheets typically follow `SVM1..SVM27` etc. (template is configurable).
- **Columns (0-based, default)**: GT=`2`, PCAD=`6`, DRF=`10`, DNN=`14`  
- **Outputs**: PDFs to `PerceivedRiskPredictionResults/outputs_ts/` (auto-created)
- **Run**:
  ```bash
  cd AnalysisCode/PerceivedRiskPrediction/PerceivedRiskPredictionResults
  # SVM has 27 events, dt=0.1 s
  python time_series_error_comparison.py --scenario SVM --events 27 --dt 0.1
  # LC often has 24 events (each ~361 samples):
  python time_series_error_comparison.py --scenario LC --events 24 --dt 0.1
  # Multiple scenarios in one command (same event count):
  python time_series_error_comparison.py --scenario HB MB LC SVM --events 27 --dt 0.1
  ```

### 5.2 Error distributions — `PerceivedRiskPredictionResults/error_distribution_plot.py`

- **Input**: `raw_data/error_data.xlsx` (or `--excel <path>`)  
  First **three columns** = PCAD, DRF, DNN. Sheets named by scene (`MB`, `HB`, `LC`, `SVM`).
- **Outputs**: `<scene>_boxplot.pdf`, `<scene>_density.pdf` to `outputs_distribution/` (auto-created)
- **Run**:
  ```bash
  python error_distribution_plot.py --scenes MB HB LC SVM --xlim 6
  ```

---

## 6) Reproducibility

```python
import random, numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
```
- Use a single `--experiment_index` (e.g., `2025`) across scenarios.
- Keep SHAP background size `K` constant when comparing global bars.
- Prefer vector outputs (PDF) for figure panels.

---

## 7) Dataset citation & license pointers

**Dataset to cite**
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

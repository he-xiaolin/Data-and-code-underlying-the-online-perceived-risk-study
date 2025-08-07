# ScenarioSignificance

This folder contains scripts and data to compute within-clip statistical significance of perceived risk ratings across four scenarios (**HB**, **LC**, **MB**, **SVM**). It:

1. Merges per-clip/condition CSVs into tidy tables  
2. Runs pairwise Welch’s t-tests within each clip  
3. Outputs significance levels and boxplots

ScenarioSignificance/
├── data_process.py           # Merge per-clip CSVs into tidy_.csv
├── ScenarioSignificance.py   # Run statistical test and output significance level and boxxplots 
├── process_data/             # Input: per-clip CSVs by scenario
│   ├── HB/
│   ├── LC/
│   ├── MB/
│   └── SVM/
└── outputs/         # boxplots
---


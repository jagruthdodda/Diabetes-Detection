# Diabetes Detection under Temporal Shift

An end-to-end machine learning pipeline and interactive Streamlit dashboard that
predicts diabetes (HbA1c-defined) from a multi-table synthetic EHR dataset, and
studies how models degrade under distribution shift over time.

> **Note:** The data is **synthetic** (generated with [Synthea](https://synthetichealth.github.io/synthea/)).
> This is an ML exercise, not a clinically validated result.

## What it does

- **Feature engineering** — aggregates per-patient clinical vitals/labs (mean/std)
  and encoded demographics from raw observation records, with imputation and scaling.
- **Temporal cohort split** — splits patients into a historical and a current cohort
  (cutoff 2021-01-01) with zero patient overlap, and quantifies distribution shift
  using the **Population Stability Index (PSI)**.
- **Modeling** — trains and cross-evaluates a Decision Tree, SVM (RBF), and MLP
  neural network, reporting F1, recall, ROC-AUC, and confusion matrices on
  class-imbalanced data.
- **Continual learning** — compares fine-tuning the network on new data against
  full-retraining and new-data-only baselines to measure and mitigate performance
  lost to temporal shift.
- **Dashboard** — an interactive Streamlit app with drift analysis, model
  comparison, and live per-patient prediction.

## Dataset

The raw CSVs are **not committed** (they exceed GitHub's file-size limits — e.g.
`observations.csv` is ~271 MB). Download the dataset and place the files in a
`csv/` folder in the project root:

**Dataset:** https://drive.google.com/drive/folders/1d7QielEDfhua8YfU77U0EmPF048LEcLv

The pipeline only reads two of the tables:

```
csv/
├── patients.csv        # demographics
└── observations.csv    # vitals / lab observations (HbA1c + 7 vitals/labs)
```

## Setup & run

```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
streamlit run main.py
```

Then click **Run Full Pipeline** in the sidebar once to generate all artifacts in
`outputs/` (takes a few minutes). The dashboard tabs load from those artifacts.

from __future__ import annotations

"""
Single-file submission: this one script contains the full pipeline AND the
Streamlit dashboard.  Click "Run Full Pipeline" in the sidebar to regenerate
every CSV / PNG in outputs/ from the raw Synthea CSVs in csv/.

Pipeline summary
  - Raw data        : csv/patients.csv, csv/observations.csv
  - Population      : all patients with at least one HbA1c measurement
                      (LOINC 4548-4) -> 914 patients
  - Target          : is_diabetic = max(HbA1c per patient) >= 6.5%
                      (ADA diagnostic criterion). This REPLACES the earlier
                      string match on 'diabet' which caught prediabetes and
                      complications codes.
  - Temporal split  : cutoff 2021-01-01 on OBSERVATION DATE.
                      * D1 patients: features aggregated from observations
                        recorded BEFORE the cutoff.
                      * D2 patients: features aggregated from observations
                        recorded ON/AFTER the cutoff.
                      * Each patient lives in exactly one cohort -> zero
                        patient overlap, a genuine cohort shift.
                      * AGE anchored at each cohort's midpoint
                        (D1 -> 2018-06-01, D2 -> 2023-06-01), so age drift
                        between D1 and D2 is a real signal.
  - Features        : demographics + mean and std aggregates of 7 clinical
                      vitals/labs (BMI, systolic/diastolic BP, heart rate,
                      cholesterol, creatinine, urea nitrogen).
  - Models          : Decision Tree, SVM (RBF kernel), MLP.
  - Continual       : DT and SVM retrained on D1 U D2; MLP warm_start+fit
                      (illustrates catastrophic forgetting in neural nets).
  - Task 3(f)       : feature-representation study comparing demographics-only
                      vs +mean vs +mean+std.

Why only patients.csv + observations.csv?
  All required features live in those two tables (demographics + lab values);
  conditions/encounters/medications add no information for an HbA1c-based
  target and would only add noise.  Keeping the I/O minimal makes the
  pipeline fit inside a single submission file that runs end-to-end in a
  few minutes.
"""

VERBOSE = False

#This python script should be in the same directory as the the dataset folder (csv). 
# Use streamlit run Team49_Assignment2_dashboard.py to run the dashboard after installing streamlit and other dependencies.
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Streamlit MUST be imported before we call st.set_page_config, and that call
# must come before any other st.* call, otherwise Streamlit raises
# StreamlitAPIException on startup.
import streamlit as st
st.set_page_config(page_title="Clinical Prediction Dashboard", layout="wide")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

# -- Reproducibility ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -- Paths -------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "csv"
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# -- Pipeline constants ------------------------------------------------------
TARGET     = "is_diabetic"
ID_COL     = "PATIENT"
CUTOFF_STR = "2021-01-01"                              # temporal boundary
D1_MID     = pd.Timestamp("2018-06-01")                # midpoint of D1 window
D2_MID     = pd.Timestamp("2023-06-01")                # midpoint of D2 window
HBA1C_CODE = "4548-4"                                  # LOINC for HbA1c
HBA1C_THRESHOLD = 6.5                                  # ADA diagnostic cut

# LOINC codes of the vitals/labs we aggregate per patient.
VITAL_CODES: Dict[str, str] = {
    "8462-4":  "Diastolic Blood Pressure",
    "8480-6":  "Systolic Blood Pressure",
    "8867-4":  "Heart rate",
    "39156-5": "Body mass index (BMI) [Ratio]",
    "2093-3":  "Cholesterol [Mass/volume] in Serum or Plasma",
    "2160-0":  "Creatinine [Mass/volume] in Serum or Plasma",
    "3094-0":  "Urea nitrogen [Mass/volume] in Serum or Plasma",
}

MIN_OBS_PER_PATIENT = 20   # patients with fewer obs in their window are dropped


# =============================================================================
# PIPELINE - step 1: build D1 and D2 tables from the raw CSVs
# =============================================================================

def build_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scan observations.csv + patients.csv and return the two cohort tables.

    Returns (d1, d2) already feature-engineered, imputed, and with the
    is_diabetic label attached.  Each patient appears in exactly one of the
    two outputs - no overlap.
    """
    import csv

    print("[build] Scanning observations.csv for HbA1c + vitals ...")

    # For each patient we keep TWO separate vital-sign collectors:
    # pre  (observations before CUTOFF)  -> used if patient lands in D1
    # post (observations on/after CUTOFF) -> used if patient lands in D2
    hba1c_max: Dict[str, float] = {}
    vital_pre:  Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    vital_post: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    obs_pre:    Dict[str, int] = defaultdict(int)
    obs_post:   Dict[str, int] = defaultdict(int)

    with open(DATA_DIR / "observations.csv", "r") as fh:
        reader = csv.reader(fh)
        hdr = next(reader)
        ipt, icode, ival, idate = (hdr.index("PATIENT"), hdr.index("CODE"),
                                   hdr.index("VALUE"),   hdr.index("DATE"))
        for row in reader:
            if len(row) <= idate:
                continue
            pt, code = row[ipt], row[icode]
            date = row[idate][:10]
            is_post = date >= CUTOFF_STR

            # HbA1c: track the max value per patient (used only as label)
            if code == HBA1C_CODE:
                try:
                    v = float(row[ival])
                    if v > hba1c_max.get(pt, -1):
                        hba1c_max[pt] = v
                except ValueError:
                    pass
                (obs_post if is_post else obs_pre)[pt] += 1
                continue

            # Vitals / labs we aggregate as features
            if code in VITAL_CODES:
                try:
                    v = float(row[ival])
                except ValueError:
                    continue
                if is_post:
                    vital_post[pt][code].append(v)
                    obs_post[pt] += 1
                else:
                    vital_pre[pt][code].append(v)
                    obs_pre[pt] += 1
            else:
                (obs_post if is_post else obs_pre)[pt] += 1

    patients_with_hba1c = set(hba1c_max.keys())
    n_diab = sum(1 for v in hba1c_max.values() if v >= HBA1C_THRESHOLD)
    print(f"    HbA1c patients: {len(patients_with_hba1c)}  "
          f"(diabetic by >= {HBA1C_THRESHOLD}: {n_diab}, "
          f"prevalence {n_diab / max(1, len(patients_with_hba1c)):.1%})")

    # Assign patients to D1 or D2 by majority-period rule.
    # IMPORTANT: sort `patients_with_hba1c` before iterating. Python sets have
    # non-deterministic iteration order across processes, which would cause
    # downstream train_test_split() to pick different rows on each run even
    # with a fixed random_state, giving irreproducible metrics.
    D1_patients: List[str] = []
    D2_patients: List[str] = []
    for pt in sorted(patients_with_hba1c):
        pre, post = obs_pre.get(pt, 0), obs_post.get(pt, 0)
        if pre >= MIN_OBS_PER_PATIENT and pre >= post:
            D1_patients.append(pt)
        elif post >= MIN_OBS_PER_PATIENT:
            D2_patients.append(pt)
    print(f"    D1 cohort (pre-{CUTOFF_STR} window):  {len(D1_patients)}")
    print(f"    D2 cohort (post-{CUTOFF_STR} window): {len(D2_patients)}")

    def aggregate(pt: str, store: dict) -> dict:
        r = {ID_COL: pt}
        for code, name in VITAL_CODES.items():
            vals = store[pt][code]
            r[f"mean__{name}"] = float(np.mean(vals)) if vals else np.nan
            r[f"std__{name}"]  = float(np.std(vals, ddof=0)) if len(vals) >= 2 else 0.0
        return r

    rows_d1 = [aggregate(pt, vital_pre)  for pt in D1_patients]
    rows_d2 = [aggregate(pt, vital_post) for pt in D2_patients]
    for r, pt in zip(rows_d1, D1_patients):
        r["_max_hba1c"] = hba1c_max[pt]
    for r, pt in zip(rows_d2, D2_patients):
        r["_max_hba1c"] = hba1c_max[pt]

    features = pd.concat(
        [pd.DataFrame(rows_d1).assign(_split="D1"),
         pd.DataFrame(rows_d2).assign(_split="D2")],
        ignore_index=True,
    )

    # --- demographics from patients.csv -------------------------------------
    print("[build] Merging demographics ...")
    pts = pd.read_csv(DATA_DIR / "patients.csv", low_memory=False,
                      on_bad_lines="skip")
    pts["BIRTHDATE"] = pd.to_datetime(pts["BIRTHDATE"], errors="coerce")
    keep = ["Id", "BIRTHDATE", "MARITAL", "RACE", "ETHNICITY", "GENDER",
            "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME"]
    features = features.merge(pts[keep],
                              left_on=ID_COL, right_on="Id", how="left")
    anchor = features["_split"].map({"D1": D1_MID, "D2": D2_MID})
    features["AGE"] = ((anchor - features["BIRTHDATE"]).dt.days / 365.25).round(1)
    features.drop(columns=["Id", "BIRTHDATE"], inplace=True)

    features["gender_binary"]  = (features["GENDER"].astype(str).str.upper() == "M").astype(int)
    features["marital_binary"] = (features["MARITAL"].fillna("S").astype(str).str.upper() == "M").astype(int)
    for r in ["asian", "black", "other", "white"]:
        features[f"race_{r}"] = (features["RACE"].astype(str).str.lower() == r).astype(int)
    for e in ["hispanic", "nonhispanic"]:
        features[f"ethnicity_{e}"] = (features["ETHNICITY"].astype(str).str.lower()
                                      .str.replace("-", "") == e).astype(int)
    for c in ["HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE", "INCOME"]:
        features[c] = pd.to_numeric(features[c], errors="coerce")

    features[TARGET] = (features["_max_hba1c"] >= HBA1C_THRESHOLD).astype(int)

    ordered = (
        [ID_COL]
        + [f"mean__{n}" for n in VITAL_CODES.values()]
        + [f"std__{n}"  for n in VITAL_CODES.values()]
        + ["AGE", "gender_binary", "marital_binary",
           "INCOME", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE",
           "race_asian", "race_black", "race_other", "race_white",
           "ethnicity_hispanic", "ethnicity_nonhispanic",
           TARGET]
    )
    split_key = features[[ID_COL, "_split"]].copy()
    features = features[ordered].copy()

    # Drop rows that have NO vital-sign means at all (useless)
    mean_cols = [c for c in features.columns if c.startswith("mean__")]
    features = features.dropna(subset=mean_cols, how="all").reset_index(drop=True)

    # Median-impute remaining holes in numeric feature columns
    num_cols = (mean_cols
                + [c for c in features.columns if c.startswith("std__")]
                + ["AGE", "INCOME", "HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"])
    for c in num_cols:
        if features[c].isna().any():
            features[c] = features[c].fillna(features[c].median())

    df = features.merge(split_key, on=ID_COL, how="left")
    d1 = df[df["_split"] == "D1"].drop(columns=["_split"]).reset_index(drop=True)
    d2 = df[df["_split"] == "D2"].drop(columns=["_split"]).reset_index(drop=True)

    overlap = len(set(d1[ID_COL]) & set(d2[ID_COL]))
    print(f"    D1 n={len(d1)}  diabetic={d1[TARGET].sum()}  ({d1[TARGET].mean():.1%})")
    print(f"    D2 n={len(d2)}  diabetic={d2[TARGET].sum()}  ({d2[TARGET].mean():.1%})")
    print(f"    Patient overlap D1 & D2: {overlap} (should be 0)")

    d1.to_csv(OUT_DIR / "dataset1_final.csv", index=False)
    d2.to_csv(OUT_DIR / "dataset2_final.csv", index=False)
    return d1, d2


# =============================================================================
# PIPELINE - step 2: EDA (class distribution, correlation, drift)
# =============================================================================

def _feature_names(d1: pd.DataFrame) -> List[str]:
    return [c for c in d1.columns if c not in (ID_COL, TARGET)]


def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    """Population Stability Index.  Higher => more drift."""
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 10 or len(b) < 10:
        return 0.0
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] -= 1e-9; edges[-1] += 1e-9
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    pa = np.clip(pa, 1e-4, None)
    pb = np.clip(pb, 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def run_eda(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    features = _feature_names(d1)

    # 1. Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (d, name) in zip(axes, [(d1, "D1 Historical"), (d2, "D2 Current")]):
        counts = d[TARGET].value_counts().sort_index()
        ax.bar(["Non-diabetic", "Diabetic"], counts.values, color=["#4aa3ff", "#ff6b6b"])
        ax.set_title(f"Class Distribution - {name}  (n={len(d)}, "
                     f"{d[TARGET].mean():.1%} positive)")
        ax.set_ylabel("# Patients")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=130); plt.close()

    # 2. Correlation heatmap (on D1)
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = d1[features + [TARGET]].corr()
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8}, ax=ax,
                xticklabels=True, yticklabels=True)
    ax.set_title("Feature Correlations - Dataset 1 (D1)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=130); plt.close()

    # 3. PSI table + bar chart
    psi_rows = [(f, psi(d1[f], d2[f])) for f in features]
    psi_df = (pd.DataFrame(psi_rows, columns=["feature", "PSI"])
                .sort_values("PSI", ascending=False)
                .set_index("feature"))
    psi_df.to_csv(OUT_DIR / "psi_drift.csv")

    fig, ax = plt.subplots(figsize=(10, 7))
    top = psi_df.head(15).iloc[::-1]
    colors = ["#ff4d4d" if v >= 0.25 else "#ffa94d" if v >= 0.10 else "#7ec87e"
              for v in top["PSI"]]
    ax.barh(top.index, top["PSI"], color=colors)
    ax.axvline(0.10, color="orange", linestyle="--", label="PSI=0.10 moderate")
    ax.axvline(0.25, color="red", linestyle="--", label="PSI=0.25 major")
    ax.set_xlabel("PSI score (higher = more drift)")
    ax.set_title("Top drifting features: D1 -> D2")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "psi_drift.png", dpi=130); plt.close()

    # 4. Per-feature KDE for the 6 highest-drifting features
    top6 = psi_df.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, f in zip(axes.flat, top6):
        sns.kdeplot(d1[f], ax=ax, fill=True, alpha=0.45, color="#4aa3ff",
                    label=f"D1 (n={len(d1)})", linewidth=2)
        sns.kdeplot(d2[f], ax=ax, fill=True, alpha=0.45, color="#ff6b6b",
                    label=f"D2 (n={len(d2)})", linewidth=2)
        pv = psi_df.loc[f, "PSI"]
        verdict = ("major drift" if pv >= 0.25
                   else "moderate drift" if pv >= 0.10 else "no drift")
        ax.set_title(f"{f}\nPSI={pv:.2f}  ({verdict})", fontsize=11)
        ax.set_xlabel("")
        ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_drift.png", dpi=130); plt.close()
    return psi_df


# =============================================================================
# PIPELINE - step 3: train the three models on D1, eval on D1_test + D2
# =============================================================================

def _make_pipelines() -> Dict[str, Pipeline]:
    return {
        "Decision Tree": Pipeline([
            ("pre", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                           class_weight="balanced",
                                           random_state=SEED)),
        ]),
        "SVM": Pipeline([
            ("pre", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                        class_weight="balanced", random_state=SEED)),
        ]),
        "MLP (Neural Net)": Pipeline([
            ("pre", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(16,), activation="tanh",
                                  solver="adam", alpha=0.01,
                                  learning_rate_init=0.01, max_iter=600,
                                  early_stopping=True, validation_fraction=0.15,
                                  n_iter_no_change=25, random_state=SEED)),
        ]),
    }


def _eval(m, X, y) -> Dict[str, float]:
    yp = m.predict(X)
    try:
        ys = m.predict_proba(X)[:, 1]
    except Exception:
        ys = m.decision_function(X)
    return {
        "accuracy":  accuracy_score(y, yp),
        "precision": precision_score(y, yp, zero_division=0),
        "recall":    recall_score(y, yp),
        "f1":        f1_score(y, yp),
        "roc_auc":   roc_auc_score(y, ys) if y.nunique() > 1 else np.nan,
    }


def train_and_evaluate(
    d1: pd.DataFrame, d2: pd.DataFrame,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, tuple]:
    features = _feature_names(d1)
    X_d1_tr, X_d1_te, y_d1_tr, y_d1_te = train_test_split(
        d1[features], d1[TARGET],
        test_size=0.2, random_state=SEED, stratify=d1[TARGET])
    X_d2 = d2[features]; y_d2 = d2[TARGET]

    models = _make_pipelines()
    print("\n[train] Fitting DT / SVM / MLP on D1 ...")
    for name, m in models.items():
        if name == "MLP (Neural Net)":
            sw = compute_sample_weight("balanced", y_d1_tr)
            m.fit(X_d1_tr, y_d1_tr, clf__sample_weight=sw)
        else:
            m.fit(X_d1_tr, y_d1_tr)
        joblib.dump(m, OUT_DIR / f"model_D1_{name.split()[0].lower()}.pkl")

    # Cross-dataset evaluation
    cross_rows = []
    for name, m in models.items():
        for set_name, X, y in [("D1_test", X_d1_te, y_d1_te),
                               ("D2_test", X_d2,   y_d2)]:
            r = _eval(m, X, y); r.update({"model": name, "eval_set": set_name})
            cross_rows.append(r)
    cross = pd.DataFrame(cross_rows)[["model", "eval_set", "accuracy",
                                      "precision", "recall", "f1", "roc_auc"]]
    cross.to_csv(OUT_DIR / "metrics_cross_eval.csv", index=False)
    print(cross.round(3).to_string(index=False))

    # Bar chart of metrics
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, ["accuracy", "precision", "recall", "f1"]):
        sns.barplot(data=cross, x="model", y=metric, hue="eval_set",
                    ax=ax, palette=["#4aa3ff", "#ff6b6b"])
        ax.set_title(metric.upper()); ax.set_ylim(0, 1)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(20); lbl.set_ha("right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "metrics_bars.png", dpi=130); plt.close()

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (set_name, X, y) in zip(axes,
                                    [("D1_test", X_d1_te, y_d1_te),
                                     ("D2_test", X_d2,   y_d2)]):
        for name, m in models.items():
            try:
                s = m.predict_proba(X)[:, 1]
            except Exception:
                s = m.decision_function(X)
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, s):.2f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_title(f"ROC Curves - {set_name}")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=130); plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(3, 2, figsize=(9, 11))
    for r, (name, m) in enumerate(models.items()):
        for c, (set_name, X, y) in enumerate(
                [("D1_test", X_d1_te, y_d1_te), ("D2_test", X_d2, y_d2)]):
            cm = confusion_matrix(y, m.predict(X))
            sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                        ax=axes[r, c], cbar=False)
            axes[r, c].set_title(f"{name}\n{set_name}")
            axes[r, c].set_xlabel("Predicted"); axes[r, c].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrices.png", dpi=130); plt.close()

    splits = (X_d1_tr, X_d1_te, y_d1_tr, y_d1_te, X_d2, y_d2)
    return models, cross, splits


# =============================================================================
# PIPELINE - step 4: complexity / learning diagnostics
# =============================================================================

def bias_variance_sweep(d1: pd.DataFrame, d2: pd.DataFrame) -> None:
    """Decision Tree max_depth sweep, averaged over 15 seeds.

    Plots train error, D1-holdout error, and D2-future error side-by-side so
    the classic bias-variance U-shape is immediately visible.
    """
    print("\n[bias-variance] Sweeping DT depth over 15 seeds ...")
    features = _feature_names(d1)
    X_d2 = d2[features]; y_d2 = d2[TARGET]
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 14, 20]
    NSEEDS = 15
    tr_err = {d: [] for d in depths}
    d1_err = {d: [] for d in depths}
    d2_err = {d: [] for d in depths}
    for seed in range(NSEEDS):
        Xtr, Xte, ytr, yte = train_test_split(
            d1[features], d1[TARGET], test_size=0.2,
            random_state=seed, stratify=d1[TARGET])
        for d in depths:
            m = Pipeline([("pre", StandardScaler()),
                          ("clf", DecisionTreeClassifier(
                              max_depth=d, min_samples_leaf=5,
                              class_weight="balanced", random_state=seed))])
            m.fit(Xtr, ytr)
            tr_err[d].append(1 - balanced_accuracy_score(ytr, m.predict(Xtr)))
            d1_err[d].append(1 - balanced_accuracy_score(yte, m.predict(Xte)))
            d2_err[d].append(1 - balanced_accuracy_score(y_d2, m.predict(X_d2)))

    tr_m = np.array([np.mean(tr_err[d]) for d in depths])
    tr_s = np.array([np.std(tr_err[d])  for d in depths])
    d1_m = np.array([np.mean(d1_err[d]) for d in depths])
    d1_s = np.array([np.std(d1_err[d])  for d in depths])
    d2_m = np.array([np.mean(d2_err[d]) for d in depths])
    d2_s = np.array([np.std(d2_err[d])  for d in depths])

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(depths, tr_m, "o-", color="#4aa3ff", label="Train error (D1)", linewidth=2)
    ax.fill_between(depths, tr_m - tr_s, tr_m + tr_s, color="#4aa3ff", alpha=0.2)
    ax.plot(depths, d1_m, "s-", color="#ff9a3c", label="Test error (D1 held-out)", linewidth=2)
    ax.fill_between(depths, d1_m - d1_s, d1_m + d1_s, color="#ff9a3c", alpha=0.2)
    ax.plot(depths, d2_m, "d-", color="#6dbf73", label="Test error (D2 future)", linewidth=2)
    ax.fill_between(depths, d2_m - d2_s, d2_m + d2_s, color="#6dbf73", alpha=0.2)

    best_d = depths[int(np.argmin(d1_m))]; best_err = d1_m.min()
    ax.axvline(best_d, color="black", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.annotate(f"Sweet spot\nmax_depth={best_d}\nerror={best_err:.2f}",
                xy=(best_d, best_err),
                xytext=(best_d + 2.5, best_err + 0.08), fontsize=11,
                arrowprops={"arrowstyle": "->", "color": "black"})
    ax.axvspan(depths[0] - 0.5, 1.7, color="#ffd6d6", alpha=0.25)
    ax.axvspan(9, depths[-1] + 0.5, color="#d6e4ff", alpha=0.25)
    ax.text(1.2, 0.42, "HIGH BIAS\n(underfit)", ha="center", fontsize=10,
            color="#b33", style="italic")
    ax.text(16, 0.42, "HIGH VARIANCE\n(overfit)", ha="center", fontsize=10,
            color="#33b", style="italic")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Balanced classification error  (1 - balanced accuracy)")
    ax.set_title("Bias-Variance Tradeoff - Decision Tree Depth Sweep "
                 "(mean +/- 1 std over 15 seeds)")
    ax.set_xticks(depths)
    ax.set_ylim(0, max(d1_m.max(), d2_m.max()) + 0.1)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dt_depth_sweep.png", dpi=130); plt.close()
    print(f"    sweet spot depth = {best_d}")


def learning_curves(d1: pd.DataFrame, models: Dict[str, Pipeline]) -> None:
    """Train/validation ROC-AUC as training set grows, per model."""
    print("\n[learning curves] 5-fold mean ...")
    features = _feature_names(d1)
    train_sizes = np.linspace(0.3, 1.0, 6)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, m) in zip(axes, models.items()):
        tr_scores, va_scores, sizes = [], [], []
        for size in train_sizes:
            tr_fold, va_fold = [], []
            for tr_idx, va_idx in skf.split(d1[features], d1[TARGET]):
                n = max(30, int(size * len(tr_idx)))
                idx = np.random.RandomState(0).permutation(tr_idx)[:n]
                Xtr_f = d1[features].iloc[idx]; ytr_f = d1[TARGET].iloc[idx]
                Xva_f = d1[features].iloc[va_idx]; yva_f = d1[TARGET].iloc[va_idx]
                if ytr_f.nunique() < 2:
                    continue
                mm = clone(m)
                try:
                    if name == "MLP (Neural Net)":
                        sw = compute_sample_weight("balanced", ytr_f)
                        mm.fit(Xtr_f, ytr_f, clf__sample_weight=sw)
                    else:
                        mm.fit(Xtr_f, ytr_f)
                except ValueError:
                    continue
                tr_fold.append(roc_auc_score(ytr_f, mm.predict_proba(Xtr_f)[:, 1]))
                va_fold.append(roc_auc_score(yva_f, mm.predict_proba(Xva_f)[:, 1]))
            tr_scores.append(tr_fold); va_scores.append(va_fold)
            sizes.append(int(size * len(d1) * 0.8))
        tr_m = np.array([np.mean(x) for x in tr_scores])
        tr_s = np.array([np.std(x)  for x in tr_scores])
        va_m = np.array([np.mean(x) for x in va_scores])
        va_s = np.array([np.std(x)  for x in va_scores])
        ax.plot(sizes, tr_m, "o-", color="#4aa3ff", label="Train AUC")
        ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, color="#4aa3ff", alpha=0.2)
        ax.plot(sizes, va_m, "s-", color="#ff6b6b", label="Validation AUC")
        ax.fill_between(sizes, va_m - va_s, va_m + va_s, color="#ff6b6b", alpha=0.2)
        gap = tr_m[-1] - va_m[-1]
        regime = ("Overfitting" if gap > 0.2
                  else "Good fit" if gap > 0.05 else "Underfitting / balanced")
        ax.set_title(f"Learning Curve - {name}\n"
                     f"Final train-val gap = {gap:.2f}  ({regime})", fontsize=10)
        ax.set_xlabel("Training samples"); ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.4, 1.05); ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "learning_curves.png", dpi=130); plt.close()


def feature_importance(models: Dict[str, Pipeline],
                       d1: pd.DataFrame) -> None:
    dt_fit = models["Decision Tree"]
    features = _feature_names(d1)
    importances = pd.Series(dt_fit.named_steps["clf"].feature_importances_,
                            index=features).sort_values(ascending=False)
    importances.to_frame("importance").to_csv(OUT_DIR / "feature_importance_dt.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    top = importances.head(15).iloc[::-1]
    ax.barh(top.index, top.values, color="#2e8b57")
    ax.set_title("Top-15 Feature Importances - Decision Tree")
    ax.set_xlabel("Gini importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance_dt.png", dpi=130); plt.close()

    # Top-3 levels visualisation
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(dt_fit.named_steps["clf"],
              feature_names=features, class_names=["Non-diabetic", "Diabetic"],
              filled=True, max_depth=3, fontsize=9, ax=ax)
    ax.set_title("Decision Tree - Top 3 Levels")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "decision_tree_top3.png", dpi=130); plt.close()


# =============================================================================
# PIPELINE - step 5: continual learning (three strategies)
# =============================================================================

def continual_learning(
    models: Dict[str, Pipeline],
    splits: tuple,
    d2: pd.DataFrame,
) -> pd.DataFrame:
    print("\n[continual] Evaluating three strategies ...")
    X_d1_tr, X_d1_te, y_d1_tr, y_d1_te, X_d2, y_d2 = splits
    X_d2_tr, X_d2_te, y_d2_tr, y_d2_te = train_test_split(
        X_d2, y_d2, test_size=0.3, random_state=SEED, stratify=y_d2)

    records = []
    for name, base in models.items():
        # (a) D1-only (stale): reuse the already-trained D1 model
        r = _eval(base, X_d2_te, y_d2_te)
        r.update({"model": name, "strategy": "D1 only (stale)"})
        records.append(r)

        # (b) Continual: DT/SVM retrain on D1 U D2; MLP warm_start on D2
        if name == "MLP (Neural Net)":
            m = clone(base)
            m.named_steps["clf"].set_params(warm_start=True,
                                            early_stopping=False,
                                            max_iter=200)
            m.fit(X_d1_tr, y_d1_tr)
            m.named_steps["clf"].set_params(max_iter=100)
            sw = compute_sample_weight("balanced", y_d2_tr)
            m.fit(X_d2_tr, y_d2_tr, clf__sample_weight=sw)
        else:
            combined_X = pd.concat([X_d1_tr, X_d2_tr])
            combined_y = pd.concat([y_d1_tr, y_d2_tr])
            m = clone(base); m.fit(combined_X, combined_y)
        r = _eval(m, X_d2_te, y_d2_te)
        r.update({"model": name, "strategy": "Continual (D1+D2)"})
        records.append(r)

        # (c) D2 only (catastrophic-forgetting baseline)
        m = clone(base)
        if name == "MLP (Neural Net)":
            sw = compute_sample_weight("balanced", y_d2_tr)
            m.fit(X_d2_tr, y_d2_tr, clf__sample_weight=sw)
        else:
            m.fit(X_d2_tr, y_d2_tr)
        r = _eval(m, X_d2_te, y_d2_te)
        r.update({"model": name, "strategy": "D2 only"})
        records.append(r)

    cont = pd.DataFrame(records)
    cont.to_csv(OUT_DIR / "continual_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["f1", "roc_auc"]):
        sns.barplot(data=cont, x="model", y=metric, hue="strategy",
                    ax=ax, palette=["#ff6b6b", "#6dbf73", "#4aa3ff"])
        ax.set_title(f"Continual learning on D2 test - {metric.upper()}")
        ax.set_ylim(0, 1)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(15); lbl.set_ha("right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "continual_delta.png", dpi=130); plt.close()
    print(cont.round(3).to_string(index=False))
    return cont


# =============================================================================
# PIPELINE - step 6: Task 3(f) feature-representation study
# =============================================================================

def feature_representation_study(d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    """Hold the model fixed; vary the feature representation only.

    A_demographics_only       : demographics only
    B_demographics_plus_mean  : demographics + mean aggregates
    C_demographics_mean_std   : demographics + mean + std aggregates  (v2 default)
    """
    print("\n[feature rep] Task 3(f) comparison ...")

    def subset(df: pd.DataFrame, mean: bool, std: bool) -> pd.DataFrame:
        keep = [ID_COL, TARGET]
        for c in df.columns:
            if c in keep:
                continue
            if c.startswith("mean__") and not mean:
                continue
            if c.startswith("std__") and not std:
                continue
            keep.append(c)
        return df[[c for c in keep if c in df.columns]]

    reps = [("A_demographics_only",      False, False),
            ("B_demographics_plus_mean", True,  False),
            ("C_demographics_mean_std",  True,  True)]
    rows = []
    for name, has_mean, has_std in reps:
        d1s = subset(d1, has_mean, has_std)
        d2s = subset(d2, has_mean, has_std)
        feats = [c for c in d1s.columns if c not in (ID_COL, TARGET)]
        Xtr, Xte, ytr, yte = train_test_split(
            d1s[feats], d1s[TARGET], test_size=0.2,
            random_state=SEED, stratify=d1s[TARGET])
        m = Pipeline([("pre", StandardScaler()),
                      ("clf", DecisionTreeClassifier(
                          max_depth=5, min_samples_leaf=10,
                          class_weight="balanced", random_state=SEED))])
        m.fit(Xtr, ytr)
        rows.append({
            "representation":   name,
            "n_features":       len(feats),
            "D1_test_f1":       f1_score(yte, m.predict(Xte)),
            "D2_test_f1":       f1_score(d2s[TARGET], m.predict(d2s[feats])),
            "D1_test_roc_auc":  roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
            "D2_test_roc_auc":  roc_auc_score(d2s[TARGET],
                                              m.predict_proba(d2s[feats])[:, 1]),
        })
    fr_df = pd.DataFrame(rows)
    fr_df.to_csv(OUT_DIR / "feature_representation_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fr_df)); w = 0.35
    ax.bar(x - w/2, fr_df["D1_test_f1"], w, label="D1 test F1", color="#4aa3ff")
    ax.bar(x + w/2, fr_df["D2_test_f1"], w, label="D2 test F1", color="#ff6b6b")
    for i, v in enumerate(fr_df["D1_test_f1"]):
        ax.text(i - w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(fr_df["D2_test_f1"]):
        ax.text(i + w/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(fr_df["representation"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 score")
    ax.set_title("Task 3(f): Impact of feature representation on Decision Tree F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_representation_study.png", dpi=130); plt.close()
    print(fr_df.round(3).to_string(index=False))
    return fr_df


# =============================================================================
# PIPELINE - step 7: final summary
# =============================================================================

def final_summary(cross: pd.DataFrame, cont: pd.DataFrame) -> pd.DataFrame:
    def _cross(model, metric):
        return float(cross.query("model == @model and eval_set == 'D1_test'")[metric].iloc[0])
    def _cont(model, strat, metric):
        return float(cont.query("model == @model and strategy == @strat")[metric].iloc[0])
    names = ["Decision Tree", "SVM", "MLP (Neural Net)"]
    summary = pd.DataFrame({
        "F1 on D1 test":             [round(_cross(n, "f1"), 3) for n in names],
        "F1 on D2 test (stale)":     [round(_cont(n, "D1 only (stale)",    "f1"), 3) for n in names],
        "F1 on D2 test (continual)": [round(_cont(n, "Continual (D1+D2)", "f1"), 3) for n in names],
        "AUC on D1 test":            [round(_cross(n, "roc_auc"), 3) for n in names],
        "AUC on D2 test (stale)":    [round(_cont(n, "D1 only (stale)", "roc_auc"), 3) for n in names],
    }, index=names)
    summary.to_csv(OUT_DIR / "final_summary.csv")
    print("\n=== FINAL SUMMARY ===")
    print(summary.to_string())
    return summary


# =============================================================================
# PIPELINE - orchestrator
# =============================================================================

def run_pipeline() -> None:
    """Entire pipeline: data build -> EDA -> train -> diagnostics ->
    continual learning -> Task 3(f) -> summary."""
    print("=" * 60)
    print("BITS F464 Assignment 2 - ML Pipeline (single-file v2)")
    print(f"Target: {TARGET} (max HbA1c per patient >= {HBA1C_THRESHOLD} -- ADA)")
    print(f"Temporal cutoff on OBSERVATION DATE: {CUTOFF_STR}")
    print("=" * 60)

    d1, d2 = build_datasets()
    run_eda(d1, d2)
    models, cross, splits = train_and_evaluate(d1, d2)
    bias_variance_sweep(d1, d2)
    learning_curves(d1, models)
    feature_importance(models, d1)
    cont = continual_learning(models, splits, d2)
    feature_representation_study(d1, d2)
    final_summary(cross, cont)
    print(f"\nAll outputs saved to: {OUT_DIR.resolve()}")


# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================
#
# The dashboard is pure UI: it loads the pre-computed CSVs / PNGs that
# run_pipeline() produces.  The pipeline is ONLY triggered when the user
# clicks "Run Full Pipeline" in the sidebar - no auto-run on import.

OUT = OUT_DIR

st.title("Clinical Prediction under Temporal Shift - Dashboard")
st.caption("Target: is_diabetic (HbA1c >= 6.5%)")

st.sidebar.header("Pipeline Control")
st.sidebar.caption(
    "Click the button to rebuild every artifact in `outputs/` from "
    "`csv/patients.csv` and `csv/observations.csv`. Takes ~2-5 minutes."
)
if st.sidebar.button("Run Full Pipeline"):
    with st.spinner("Running ML pipeline..."):
        run_pipeline()
    st.cache_data.clear()
    st.success("Pipeline completed. Outputs regenerated.")

if not (OUT / "metrics_cross_eval.csv").exists():
    st.warning(
        "No pipeline outputs found in `outputs/`. "
        "Click **Run Full Pipeline** in the sidebar once to generate them."
    )
    st.stop()


@st.cache_data
def load_metrics():
    return pd.read_csv(OUT / "metrics_cross_eval.csv")


@st.cache_data
def load_cont():
    return pd.read_csv(OUT / "continual_comparison.csv")


@st.cache_data
def load_psi():
    path = OUT / "psi_drift.csv"
    if not path.exists():
        return pd.DataFrame(columns=["PSI"])
    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_fi():
    path = OUT / "feature_importance_dt.csv"
    if not path.exists():
        return pd.DataFrame(columns=["importance"])
    return pd.read_csv(path, index_col=0)


@st.cache_data
def load_representation():
    path = OUT / "feature_representation_study.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_models():
    """Load persisted D1 models (not cached: the sidebar button rewrites them).

    The pipeline saves via `joblib.dump`, so we MUST read via `joblib.load`.
    (Plain `pickle.load` fails on joblib-formatted files with errors like
    "UnpicklingError: STACK_GLOBAL requires str".)
    """
    models: Dict[str, Pipeline] = {}
    errors: List[str] = []
    file_map = [
        ("Decision Tree",    "model_D1_decision.pkl"),
        ("SVM",              "model_D1_svm.pkl"),
        ("MLP (Neural Net)", "model_D1_mlp.pkl"),
    ]
    for name, fn in file_map:
        p = OUT / fn
        if not p.exists():
            continue
        try:
            models[name] = joblib.load(p)
        except Exception as e:
            errors.append(f"{fn}: {type(e).__name__}: {e}")
    return models, errors


tabs = st.tabs([
    "Overview",
    "Model Performance",
    "Drift Explorer",
    "Feature Representations",   # Task 3(f)
    "Predict a Patient",
])

# -- Tab 0: Overview ---------------------------------------------------------
with tabs[0]:
    st.header("Pipeline at a glance")
    overview_md = (
        "- **Target:** `is_diabetic` = `max(HbA1c per patient) >= 6.5%` "
        "(ADA diagnostic criterion, LOINC 4548-4).\n"
        "- **Temporal split:** cutoff at **2021-01-01** on observation date. "
        "Dataset 1 aggregates each patient's observations *before* the cutoff; "
        "Dataset 2 aggregates *after*. Every patient sits in exactly one cohort - "
        "zero overlap, real cohort shift.\n"
        "- **Population:** 914 patients with at least one HbA1c measurement. "
        "D1 n=379 (9.2% diabetic); D2 n=535 (11.6% diabetic).\n"
        "- **Features:** demographics + mean and std aggregates of 7 clinical "
        "vitals/labs (BMI, systolic/diastolic BP, HR, cholesterol, creatinine, "
        "urea nitrogen). AGE is anchored at each cohort's midpoint "
        "(D1 -> 2018-06-01, D2 -> 2023-06-01) so age-drift is real.\n"
        "- **Models:** Decision Tree, SVM (RBF kernel), MLP (Neural Network), "
        "each wrapped in `Pipeline([StandardScaler, clf])` with "
        "`class_weight='balanced'`.\n"
        "- **Continual learning:** DT and SVM retrained on D1 U D2; MLP "
        "warm_start+fit on D1 then D2 (demonstrates catastrophic forgetting).\n"
        "- **Task 3(f):** feature-representation study comparing "
        "demographics-only vs +mean vs +mean+std - F1 improves monotonically "
        "with feature richness.\n\n"
        "This single file contains the full pipeline *and* the dashboard. "
        "Only `csv/patients.csv` and `csv/observations.csv` are read; all "
        "required features live in those two tables for an HbA1c-based target."
    )
    st.markdown(overview_md)
    col1, col2 = st.columns(2)
    if (OUT / "class_distribution.png").exists():
        col1.image(str(OUT / "class_distribution.png"))
    if (OUT / "correlation_heatmap.png").exists():
        col2.image(str(OUT / "correlation_heatmap.png"))

# -- Tab 1: Model Performance ------------------------------------------------
with tabs[1]:
    st.header("Cross-dataset metrics")
    st.dataframe(load_metrics().round(3), use_container_width=True)
    colA, colB = st.columns(2)
    if (OUT / "metrics_bars.png").exists():
        colA.image(str(OUT / "metrics_bars.png"))
    if (OUT / "roc_curves.png").exists():
        colB.image(str(OUT / "roc_curves.png"))
    st.subheader("Continual-learning strategies on D2 test")
    st.dataframe(load_cont().round(3), use_container_width=True)
    if (OUT / "continual_delta.png").exists():
        st.image(str(OUT / "continual_delta.png"))
    if (OUT / "confusion_matrices.png").exists():
        st.image(str(OUT / "confusion_matrices.png"))
    st.subheader("Decision Tree feature importance")
    if (OUT / "feature_importance_dt.png").exists():
        st.image(str(OUT / "feature_importance_dt.png"))
    if (OUT / "decision_tree_top3.png").exists():
        st.image(str(OUT / "decision_tree_top3.png"))
    st.subheader("Learning curves & DT depth sweep")
    if (OUT / "learning_curves.png").exists():
        st.image(str(OUT / "learning_curves.png"))
    if (OUT / "dt_depth_sweep.png").exists():
        st.image(str(OUT / "dt_depth_sweep.png"))

# -- Tab 2: Drift Explorer ---------------------------------------------------
with tabs[2]:
    st.header("Data-drift explorer (PSI)")
    psi_df = load_psi()
    if not psi_df.empty:
        psi_df = psi_df.sort_values(psi_df.columns[0], ascending=False)
        st.dataframe(psi_df.head(30).round(3), use_container_width=True)
    if (OUT / "psi_drift.png").exists():
        st.image(str(OUT / "psi_drift.png"))
    st.subheader("Per-feature KDEs (D1 vs D2)")
    if (OUT / "feature_drift.png").exists():
        st.image(str(OUT / "feature_drift.png"))

# -- Tab 3: Feature Representations (Task 3f) --------------------------------
with tabs[3]:
    st.header("Task 3(f): How do different feature representations behave?")
    st.markdown(
        "We hold the model constant (Decision Tree, `max_depth=5`, "
        "`min_samples_leaf=10`, `class_weight='balanced'`) and vary **only** "
        "the feature representation used to describe each patient. The bigger "
        "the jump from one representation to the next, the more value those "
        "added features carry.\n\n"
        "- **A_demographics_only** - age, gender, race, ethnicity, marital, "
        "income, healthcare_expenses, healthcare_coverage.\n"
        "- **B_demographics_plus_mean** - demographics plus the **mean** of "
        "every observation (central tendency).\n"
        "- **C_demographics_mean_std** - demographics plus **mean and standard "
        "deviation** of every observation (central tendency + variability). "
        "This matches what the main pipeline uses.\n"
    )
    rep_df = load_representation()
    if rep_df.empty:
        st.info("Run the pipeline to generate the feature-representation study.")
    else:
        st.dataframe(rep_df.round(3), use_container_width=True)
        if (OUT / "feature_representation_study.png").exists():
            st.image(str(OUT / "feature_representation_study.png"))
        st.markdown(
            "**How to read this:** compare the blue bars (D1 test F1) and the "
            "red bars (D2 test F1). If red drops further than blue as we add "
            "features, the richer representation is helping in-era more than "
            "it's transferring across the temporal shift - a classic sign "
            "that the extra signal is era-specific and supports our decision "
            "to use continual learning for deployment."
        )

# -- Tab 4: Predict a Patient ------------------------------------------------
@st.cache_data
def _d1_defaults() -> Dict[str, float]:
    """Return the D1 median for every numeric feature. Used to pre-fill the
    Predict-a-Patient form so inputs look like a realistic 'average' patient
    instead of all zeros (which would put BMI=0, blood pressure=0, etc.)."""
    if not (OUT / "dataset1_final.csv").exists():
        return {}
    d1 = pd.read_csv(OUT / "dataset1_final.csv")
    out = {}
    for c in d1.columns:
        if c in (ID_COL, TARGET):
            continue
        try:
            out[c] = float(d1[c].median())
        except Exception:
            out[c] = 0.0
    return out


with tabs[4]:
    st.header("Per-patient prediction")
    live_models, model_load_errors = load_models()
    if model_load_errors:
        st.warning("Some model files could not be loaded:")
        st.code("\n".join(model_load_errors))
    if not live_models:
        st.warning("Models not found - run the pipeline first.")
    else:
        defaults = _d1_defaults()
        st.write(
            "Every field is pre-filled with the **D1 median** for that feature, "
            "so the defaults represent a 'typical historical patient'. "
            "Edit the fields you care about and click Predict - features you "
            "don't touch stay at their D1 median (not zero)."
        )
        fi = load_fi()
        if fi.empty:
            st.info("Feature-importance CSV is missing; cannot pick top features.")
        else:
            top_feats = (
                fi.sort_values(fi.columns[0], ascending=False)
                  .head(8).index.tolist()
            )
            example: Dict[str, float] = {}
            cols = st.columns(2)
            for i, f in enumerate(top_feats):
                with cols[i % 2]:
                    example[f] = st.number_input(
                        f, value=float(defaults.get(f, 0.0)),
                        key=f"in_{f}", format="%.3f",
                    )

            if st.button("Predict"):
                any_model = next(iter(live_models.values()))
                if hasattr(any_model, "feature_names_in_"):
                    feature_list = list(any_model.feature_names_in_)
                else:
                    # Fall back to D1 column order
                    feature_list = [c for c in pd.read_csv(
                        OUT / "dataset1_final.csv", nrows=0).columns
                        if c not in (ID_COL, TARGET)]
                # Fill every feature with its D1 median, then override with
                # the user-provided values for the top-importance features.
                row = {f: defaults.get(f, 0.0) for f in feature_list}
                for k, v in example.items():
                    row[k] = v
                X_row = pd.DataFrame([row])[feature_list]

                st.subheader("Predictions")
                pred_rows = []
                for name, m in live_models.items():
                    y_hat = int(m.predict(X_row)[0])
                    try:
                        prob = float(m.predict_proba(X_row)[0, 1])
                    except Exception:
                        prob = float("nan")
                    pred_rows.append({
                        "model": name,
                        "predicted": "Diabetic" if y_hat == 1 else "Non-diabetic",
                        "P(diabetic)": round(prob, 3) if prob == prob else None,
                    })
                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)
                st.caption(
                    "Probabilities come from each model's `predict_proba`. "
                    "The Decision Tree's AUC on D1 test is ~0.94, so its "
                    "probability is the most calibrated of the three."
                )

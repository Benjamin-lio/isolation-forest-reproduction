
from pathlib import Path
from time import perf_counter
import csv

import numpy as np
import scipy
import sklearn
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

# -------------------------------------------------
# 1. Experiment configuration
# -------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "BreastW": ("breastw.mat", 0.99),
    "Mammography": ("mammography.mat", 0.86),
}

N_TREES = 100
SAMPLE_SIZE = 256
N_RUNS = 10

all_trials = []
summary = []

# -------------------------------------------------
# 2. Load datasets and reproduce experiments
# -------------------------------------------------

for name, (filename, paper_auc) in DATASETS.items():

    # Support files in the root or datasets/ folder
    path = PROJECT_DIR / filename
    if not path.exists():
        path = PROJECT_DIR / "datasets" / filename

    data = loadmat(path)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"]).ravel().astype(int)

    assert np.isfinite(X).all(), "Non-finite values found"
    assert set(np.unique(y)) == {0, 1}, "Unexpected labels"

    aucs = []

    print(f"\n=== {name} ===")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Anomalies: {np.sum(y == 1)}")

    for seed in range(N_RUNS):

        model = IsolationForest(
            n_estimators=N_TREES,
            max_samples=SAMPLE_SIZE,
            max_features=1.0,
            contamination="auto",
            bootstrap=False,
            random_state=seed,
            n_jobs=1
        )

        start = perf_counter()
        model.fit(X)
        fit_time = perf_counter() - start

        start = perf_counter()
        # sklearn scores are the opposite of the anomaly scores in the original paper.
        anomaly_scores = -model.score_samples(X)
        score_time = perf_counter() - start

        auc = roc_auc_score(y, anomaly_scores)
        aucs.append(auc)

        all_trials.append({
            "dataset": name,
            "seed": seed,
            "auc": auc,
            "fit_seconds": fit_time,
            "score_seconds": score_time
        })

        print(f"Run {seed + 1:02d}: AUC = {auc:.5f}")

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs, ddof=1))
    difference = mean_auc - paper_auc

    summary.append({
        "dataset": name,
        "paper_auc": paper_auc,
        "reproduced_mean_auc": mean_auc,
        "std_auc": std_auc,
        "difference": difference
    })

    print(f"\nPaper AUC:    {paper_auc:.5f}")
    print(f"Mean AUC:     {mean_auc:.5f}")
    print(f"Std AUC:      {std_auc:.5f}")
    print(f"Difference:   {difference:+.5f}")

# -------------------------------------------------
# 3. Export reproducible results
# -------------------------------------------------

for filename, rows in [
    ("individual_runs.csv", all_trials),
    ("summary.csv", summary)
]:
    output = RESULT_DIR / filename

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output}")

print("\n=== Environment ===")
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("Scikit-learn:", sklearn.__version__)
print("\nExperiment completed successfully.")

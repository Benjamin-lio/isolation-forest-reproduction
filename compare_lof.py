
from pathlib import Path
from time import perf_counter
import csv

import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "BreastW": ("breastw.mat", 0.37),
    "Mammography": ("mammography.mat", 0.67),
}

results = []

for name, (filename, paper_auc) in DATASETS.items():

    path = PROJECT_DIR / filename
    if not path.exists():
        path = PROJECT_DIR / "datasets" / filename

    data = loadmat(path)
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"]).ravel().astype(int)

    # LOF baseline from the original paper: k = 10
    model = LocalOutlierFactor(
        n_neighbors=10,
        metric="euclidean",
        contamination="auto",
        novelty=False,
        n_jobs=1
    )

    start = perf_counter()
    model.fit(X)
    elapsed = perf_counter() - start

    # Higher scores mean more anomalous.
    scores = -model.negative_outlier_factor_

    auc = roc_auc_score(y, scores)

    results.append({
        "dataset": name,
        "paper_lof_auc": paper_auc,
        "reproduced_lof_auc": auc,
        "difference": auc - paper_auc,
        "processing_seconds": elapsed
    })

    print(f"\n=== {name} ===")
    print(f"Paper LOF AUC:      {paper_auc:.5f}")
    print(f"Reproduced LOF AUC: {auc:.5f}")
    print(f"Difference:         {auc - paper_auc:+.5f}")
    print(f"Processing time:    {elapsed:.4f} s")

output = RESULT_DIR / "lof_results.csv"

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved: {output}")
print("LOF experiment completed.")

# isolation-forest-reproduction
Data Wrangling course, assignment 2, task 2.
requirements:
numpy==2.5.3
scipy==1.18.1
scikit-learn==1.9.1

This repository contains a partial reproduction of the benchmark
experiment in Section 5.1 of Liu et al. (2008), "Isolation Forest".

## Experiments

Two anomaly detection algorithms are evaluated:

- Isolation Forest (100 trees, 256 samples per tree)
- Local Outlier Factor (10 nearest neighbours)

The experiments use two datasets:
- BreastW: 683 observations, 9 features
- Mammography: 11,183 observations, 6 features

## Requirements

Install the required Python packages:

pip install -r requirements.txt

## How to Run

Run the Isolation Forest experiment:

python reproduce.py

Run the LOF comparison:

python compare_lof.py

Both scripts read the .mat datasets from the project directory
and save their outputs to the results/ folder.

## Results

Isolation Forest is repeated 10 times using random seeds 0-9.
LOF is run once per dataset.

The CSV files contain the individual runs, summary statistics
and LOF results.

The anomaly labels are used only for ROC-AUC evaluation.

## Notes

The LOF implementation may issue warnings about duplicate
observations. The original data and neighbour setting are retained.

This is a partial experimental reproduction using scikit-learn,
not a reimplementation of the original algorithms.

## Reference

Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008).
Isolation Forest. IEEE International Conference on Data Mining.
https://doi.org/10.1109/ICDM.2008.17

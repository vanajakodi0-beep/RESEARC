# Break-aware CausalImpact intervals: an equal-width benchmark

Code for "Break-Aware Causal Impact Intervals Using a Negative-Control Metric: An Equal-Width
Evaluation Against Fixed Widening". Data: Kaggle `mikhail1681/walmart-sales`, saved as
`Walmart_Sales.csv` (45 stores, first 143 weeks used). The file is not redistributed.

## Which file makes which table
| Paper tables | File | BSTS implementation |
|---|---|---|
| Sections 8.1-8.9 (Tables 8.1-8.9): coverage, width, MAE/bias, detection, point correction, DiD/SC, pooled equal-width test | `notebooks/sections_8_1_to_8_11.ipynb` | tfcausalimpact 0.0.19 |
| Sections 8.10-8.11 (BSTS with control store) | same notebook, `RUN = "full"` / `"all"` | tfcausalimpact 0.0.19 |
| Section 8.12 (Tables 8.10-8.15): scenarios A, B, C, D, D2 | `run_shift_scenarios.py` | local-level Kalman filter (maximum likelihood) |

The two implementations give different standard-BSTS numbers, so tables from the two files must not be compared directly.

## Install
```
pip install -r requirements.txt
```
The notebook needs a Python version that TensorFlow supports (Colab's default works).

## Run
Section 8.12 (about 10 minutes on a laptop):
```
python run_shift_scenarios.py --data Walmart_Sales.csv --out results
```
Outputs: `results/all_results.csv` (split x scenario x method: coverage, width, MAE, bias) and
`results/split_params.csv` (selected lambda, k, threshold, conformal quantiles per split). Tables 8.12-8.13 are
means of `all_results.csv` over the 20 splits; width ratios are width divided by the `standard` width.

Sections 8.1-8.11: open `notebooks/sections_8_1_to_8_11.ipynb`, put `Walmart_Sales.csv` in the working folder, set
`RUN = "pilot"` (20 pairs, quick check), then `"full"` (primary split) or `"all"` (all 1,980 pairs, needed for the
20-split BSTS-with-control summary). Tables are written to `paper_tables/`; progress is cached, so an interrupted run resumes.

## Known gaps (to resolve before submission)
- The paper's Section 8.12 numbers were produced on a synthetic stand-in; this script now requires the real file, so rerun
  it and update Tables 8.10-8.15.
- The notebook seeds the BSTS fits (`BSTS_SEED = 0`), while the paper says the reported fits were not seeded.
- The notebook's lambda/k grids are commented as reproducing lambda = 100, k = 8 on the primary split; the paper reports
  lambda = 2, k = 3 (Table 8.1). Confirm which run produced the paper.

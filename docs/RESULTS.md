# Verified experiment results

All numbers below are transcribed from the uploaded `2802519114(1).ipynb` notebook, rather than being reconstructed by rerunning model training.

## Dataset and evaluation

| Parameter | Notebook-recorded value |
| --- | ---: |
| Original dataset | 25,000 rows; 29 columns |
| Modeling rows after cleaning | 22,412 |
| Training rows | 17,835 |
| Test rows | 4,577 |
| Unique train customers | 7,212 |
| Unique test customers | 1,830 |
| Overlapping customer IDs | 0 |
| Target labels | Good, Poor, Standard |
| Best model | XGBoost, Original + Loan Flags |
| Best final macro F1 | 0.6898055617685017 |
| Test accuracy, best configuration | 0.7079 |
| Test macro OvR ROC-AUC, best configuration | 0.8591 |

## Full model comparison

| Feature set | Model | Accuracy | Balanced accuracy | Macro F1 | OvR ROC-AUC (macro) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original + Loan Flags | XGBoost | 0.7079 | 0.7040 | 0.6898 | 0.8591 |
| Original + Feature Engineering | XGBoost | 0.7044 | 0.7005 | 0.6860 | 0.8508 |
| Original + Feature Engineering | Random Forest | 0.6825 | 0.7155 | 0.6727 | 0.8475 |
| Original + Loan Flags | LightGBM | 0.6753 | 0.7258 | 0.6701 | 0.8552 |
| Original + Feature Engineering | LightGBM | 0.6771 | 0.7222 | 0.6701 | 0.8460 |
| Original + Loan Flags | Random Forest | 0.6771 | 0.7136 | 0.6685 | 0.8543 |
| Original + Loan Flags | Logistic Regression | 0.6589 | 0.7024 | 0.6509 | 0.8098 |
| Original + Feature Engineering | Logistic Regression | 0.6572 | 0.7017 | 0.6495 | 0.8065 |

## XGBoost confusion matrix (Original + Loan Flags)

| Actual / Predicted | Good | Poor | Standard |
| --- | ---: | ---: | ---: |
| Good | 566 | 20 | 204 |
| Poor | 93 | 896 | 349 |
| Standard | 346 | 325 | 1,778 |

Total: 4,577 evaluation observations; the matrix diagonal totals 3,240 correct predictions.

## Evaluation caveat

The notebook builds grouped training/test partitions and uses grouped cross-validation during tuning. However, in the final comparison (cell 680), it selects the final best model/feature set using **test-set macro F1**. This is exploratory model selection, not an independent final holdout result. For unbiased reporting, prefer validation/nested cross-validation selection followed by untouched grouped holdout evaluation.

Several dataset repair steps are executed before the split. Further leakage review and fold-specific preprocessing are advised before production use. No retraining or verification of the provided `.pkl` was performed for this documentation.

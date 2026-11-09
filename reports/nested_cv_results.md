# Nested cross-validation results

Mean ± std across outer folds. Model selected by ROC-AUC.

| Model | roc_auc | average_precision | f1 | precision | recall | accuracy |
|---|---|---|---|---|---|---|
| **random_forest** (selected) | 0.847 ± 0.012 | 0.659 ± 0.027 | 0.636 ± 0.012 | 0.530 ± 0.010 | 0.795 ± 0.016 | 0.758 ± 0.008 |
| gradient_boosting | 0.846 ± 0.010 | 0.655 ± 0.022 | 0.630 ± 0.007 | 0.517 ± 0.007 | 0.806 ± 0.011 | 0.749 ± 0.005 |
| logistic_regression | 0.845 ± 0.013 | 0.655 ± 0.026 | 0.625 ± 0.009 | 0.512 ± 0.007 | 0.803 ± 0.016 | 0.745 ± 0.005 |

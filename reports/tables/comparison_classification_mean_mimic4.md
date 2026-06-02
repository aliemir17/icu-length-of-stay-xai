# Model comparison — classification_mean  (mimic4)

`N_train = 38577`, `N_test = 9645`, `features = 152`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | auc_pr (CV mean ± std) | auc_roc (CV mean ± std) | f1 (CV mean ± std) | accuracy (CV mean ± std) | auc_pr (test) | auc_roc (test) | f1 (test) | accuracy (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 0.582 ± 0.008 | 0.776 ± 0.002 | 0.549 ± 0.007 | 0.745 ± 0.003 | 0.595 | 0.779 | 0.554 | 0.746 |
| linear | 0.505 ± 0.010 | 0.726 ± 0.004 | 0.507 ± 0.003 | 0.691 ± 0.003 | 0.518 | 0.734 | 0.516 | 0.699 |
| rf | 0.553 ± 0.008 | 0.757 ± 0.003 | 0.494 ± 0.006 | 0.772 ± 0.002 | 0.560 | 0.759 | 0.493 | 0.773 |
| xgb | 0.583 ± 0.009 | 0.776 ± 0.002 | 0.550 ± 0.004 | 0.748 ± 0.003 | 0.596 | 0.782 | 0.562 | 0.756 |

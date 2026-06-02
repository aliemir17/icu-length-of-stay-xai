# Model comparison — classification  (mimic4)

`N_train = 38577`, `N_test = 9645`, `features = 152`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | auc_pr (CV mean ± std) | auc_roc (CV mean ± std) | f1 (CV mean ± std) | accuracy (CV mean ± std) | auc_pr (test) | auc_roc (test) | f1 (test) | accuracy (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 0.435 ± 0.009 | 0.809 ± 0.009 | 0.460 ± 0.005 | 0.830 ± 0.002 | 0.455 | 0.820 | 0.473 | 0.826 |
| linear | 0.353 ± 0.019 | 0.766 ± 0.009 | 0.380 ± 0.008 | 0.722 ± 0.007 | 0.352 | 0.772 | 0.387 | 0.723 |
| rf | 0.403 ± 0.014 | 0.790 ± 0.009 | 0.382 ± 0.013 | 0.870 ± 0.002 | 0.424 | 0.801 | 0.398 | 0.870 |
| xgb | 0.441 ± 0.009 | 0.812 ± 0.009 | 0.462 ± 0.006 | 0.839 ± 0.002 | 0.462 | 0.820 | 0.473 | 0.836 |

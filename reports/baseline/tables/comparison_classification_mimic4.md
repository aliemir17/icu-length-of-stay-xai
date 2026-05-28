# Model comparison — classification  (mimic4)

`N_train = 38577`, `N_test = 9645`, `features = 110`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | auc_pr (CV mean ± std) | auc_roc (CV mean ± std) | f1 (CV mean ± std) | accuracy (CV mean ± std) | auc_pr (test) | auc_roc (test) | f1 (test) | accuracy (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 0.413 ± 0.013 | 0.793 ± 0.008 | 0.439 ± 0.012 | 0.813 ± 0.006 | 0.439 | 0.807 | 0.448 | 0.812 |
| linear | 0.302 ± 0.019 | 0.717 ± 0.013 | 0.339 ± 0.010 | 0.696 ± 0.005 | 0.300 | 0.725 | 0.349 | 0.704 |
| rf | 0.382 ± 0.015 | 0.773 ± 0.011 | 0.393 ± 0.009 | 0.853 ± 0.002 | 0.403 | 0.785 | 0.411 | 0.847 |
| xgb | 0.420 ± 0.010 | 0.799 ± 0.009 | 0.443 ± 0.007 | 0.843 ± 0.002 | 0.440 | 0.809 | 0.458 | 0.839 |

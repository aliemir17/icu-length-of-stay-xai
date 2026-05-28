# Model comparison — regression_raw  (mimic4)

`N_train = 38577`, `N_test = 9645`, `features = 110`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | mae (CV mean ± std) | rmse (CV mean ± std) | r2 (CV mean ± std) | mape (CV mean ± std) | mae (test) | rmse (test) | r2 (test) | mape (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 2.567 ± 0.023 | 4.982 ± 0.090 | 0.167 ± 0.017 | 0.828 ± 0.012 | 2.490 | 4.648 | 0.198 | 0.819 |
| linear | 2.983 ± 0.188 | 17.156 ± 13.174 | -14.793 ± 20.557 | 1.041 ± 0.084 | 2.749 | 4.926 | 0.099 | 0.951 |
| rf | 2.701 ± 0.030 | 5.165 ± 0.094 | 0.105 ± 0.018 | 0.915 ± 0.008 | 2.637 | 4.846 | 0.128 | 0.908 |
| xgb | 2.569 ± 0.023 | 4.988 ± 0.089 | 0.165 ± 0.017 | 0.828 ± 0.013 | 2.496 | 4.661 | 0.193 | 0.816 |

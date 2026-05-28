# Model comparison — regression_raw  (mimic4)

`N_train = 38577`, `N_test = 9645`, `features = 152`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | mae (CV mean ± std) | rmse (CV mean ± std) | r2 (CV mean ± std) | mape (CV mean ± std) | mae (test) | rmse (test) | r2 (test) | mape (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 2.519 ± 0.014 | 4.918 ± 0.076 | 0.189 ± 0.012 | 0.804 ± 0.014 | 2.456 | 4.592 | 0.217 | 0.793 |
| linear | 2.731 ± 0.034 | 5.525 ± 0.562 | -0.033 ± 0.211 | 0.920 ± 0.012 | 2.669 | 4.816 | 0.139 | 0.909 |
| rf | 2.591 ± 0.021 | 4.995 ± 0.084 | 0.163 ± 0.014 | 0.856 ± 0.011 | 2.531 | 4.675 | 0.188 | 0.848 |
| xgb | 2.547 ± 0.026 | 4.949 ± 0.090 | 0.178 ± 0.018 | 0.818 ± 0.011 | 2.483 | 4.642 | 0.200 | 0.806 |

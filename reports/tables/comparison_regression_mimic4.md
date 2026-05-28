# Model comparison — regression  (mimic4)

`N_train = 38577`, `N_test = 9645`, `features = 152`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | mae (CV mean ± std) | rmse (CV mean ± std) | r2 (CV mean ± std) | mape (CV mean ± std) | mae (test) | rmse (test) | r2 (test) | mape (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 2.231 ± 0.028 | 5.005 ± 0.078 | 0.159 ± 0.013 | 0.563 ± 0.006 | 2.168 | 4.680 | 0.187 | 0.553 |
| linear | 2.421 ± 0.057 | 6.720 ± 1.246 | -0.563 ± 0.528 | 0.628 ± 0.012 | 2.344 | 4.963 | 0.085 | 0.612 |
| rf | 2.302 ± 0.029 | 5.155 ± 0.073 | 0.108 ± 0.011 | 0.585 ± 0.004 | 2.254 | 4.847 | 0.128 | 0.580 |
| xgb | 2.227 ± 0.026 | 5.002 ± 0.069 | 0.161 ± 0.010 | 0.561 ± 0.006 | 2.169 | 4.684 | 0.185 | 0.553 |

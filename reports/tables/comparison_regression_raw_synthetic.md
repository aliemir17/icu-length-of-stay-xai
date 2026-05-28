# Model comparison — regression_raw  (synthetic)

`N_train = 1600`, `N_test = 400`, `features = 14`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | mae (CV mean ± std) | rmse (CV mean ± std) | r2 (CV mean ± std) | mape (CV mean ± std) | mae (test) | rmse (test) | r2 (test) | mape (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 2.252 ± 0.030 | 3.024 ± 0.066 | -0.088 ± 0.091 | 1.229 ± 0.127 | 2.213 | 3.014 | -0.108 | 1.261 |
| linear | 2.118 ± 0.045 | 2.832 ± 0.104 | 0.049 ± 0.048 | 1.221 ± 0.091 | 2.102 | 2.822 | 0.029 | 1.191 |
| rf | 2.168 ± 0.032 | 2.908 ± 0.073 | -0.006 ± 0.083 | 1.248 ± 0.086 | 2.193 | 2.943 | -0.057 | 1.264 |
| xgb | 2.253 ± 0.044 | 3.027 ± 0.085 | -0.089 ± 0.078 | 1.252 ± 0.084 | 2.269 | 3.014 | -0.108 | 1.288 |

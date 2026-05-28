# Model comparison — regression  (synthetic)

`N_train = 1600`, `N_test = 400`, `features = 14`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | mae (CV mean ± std) | rmse (CV mean ± std) | r2 (CV mean ± std) | mape (CV mean ± std) | mae (test) | rmse (test) | r2 (test) | mape (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 2.173 ± 0.078 | 3.027 ± 0.134 | -0.086 ± 0.037 | 1.043 ± 0.078 | 2.137 | 2.999 | -0.097 | 1.073 |
| linear | 2.046 ± 0.081 | 2.899 ± 0.162 | 0.006 ± 0.025 | 0.994 ± 0.067 | 1.988 | 2.871 | -0.006 | 0.965 |
| rf | 2.061 ± 0.083 | 2.936 ± 0.148 | -0.021 ± 0.048 | 1.001 ± 0.064 | 2.015 | 2.888 | -0.017 | 0.992 |
| xgb | 2.140 ± 0.069 | 3.008 ± 0.148 | -0.071 ± 0.028 | 1.034 ± 0.055 | 2.091 | 2.955 | -0.065 | 1.044 |

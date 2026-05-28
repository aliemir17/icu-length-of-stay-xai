# Model comparison — classification  (synthetic)

`N_train = 1600`, `N_test = 400`, `features = 14`. CV = 5-fold on the training split; test = single held-out 20 % split. Random seed = 42.

| Model | auc_pr (CV mean ± std) | auc_roc (CV mean ± std) | f1 (CV mean ± std) | accuracy (CV mean ± std) | auc_pr (test) | auc_roc (test) | f1 (test) | accuracy (test) |
|---|---|---|---|---|---|---|---|---|
| lgbm | 0.181 ± 0.032 | 0.575 ± 0.043 | 0.126 ± 0.030 | 0.835 ± 0.010 | 0.184 | 0.582 | 0.148 | 0.828 |
| linear | 0.239 ± 0.042 | 0.628 ± 0.044 | 0.277 ± 0.041 | 0.657 ± 0.020 | 0.233 | 0.654 | 0.303 | 0.677 |
| rf | 0.198 ± 0.038 | 0.572 ± 0.042 | 0.019 ± 0.023 | 0.874 ± 0.002 | 0.194 | 0.583 | 0.036 | 0.865 |
| xgb | 0.185 ± 0.029 | 0.577 ± 0.042 | 0.105 ± 0.055 | 0.856 ± 0.004 | 0.178 | 0.587 | 0.118 | 0.850 |

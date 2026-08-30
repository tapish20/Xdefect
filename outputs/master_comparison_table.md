# Master Performance Comparison Table

Statistically significant superior models ($p < 0.05$ vs. Logistic Regression via Wilcoxon signed-rank tests) are marked with **bold font (*)**.

| Evaluation Type | Feature Set | Model | AUC-ROC | F1 Score | MCC | Accuracy | Precision | Recall | Balanced Acc | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| within-project | hybrid | **lightgbm** (*) | `0.8216` | `0.5067` | `0.4059` | `0.8418` | `0.5219` | `0.5007` | `0.6981` | `0.5484` |
| within-project | hybrid | logistic_regression | `0.7617` | `0.4676` | `0.3422` | `0.7773` | `0.4084` | `0.5799` | `0.6954` | `0.5103` |
| within-project | hybrid | **random_forest** (*) | `0.8313` | `0.5221` | `0.4239` | `0.8443` | `0.5242` | `0.5305` | `0.7113` | `0.5478` |
| within-project | hybrid | **xgboost** (*) | `0.8229` | `0.5219` | `0.4133` | `0.8313` | `0.5050` | `0.5490` | `0.7119` | `0.5509` |
| within-project | static | **lightgbm** (*) | `0.7871` | `0.4500` | `0.3351` | `0.8223` | `0.4587` | `0.4595` | `0.6654` | `0.4818` |
| within-project | static | logistic_regression | `0.7600` | `0.4465` | `0.3242` | `0.7632` | `0.3777` | `0.6001` | `0.6963` | `0.4725` |
| within-project | static | **random_forest** (*) | `0.7971` | `0.4426` | `0.3256` | `0.8185` | `0.4364` | `0.4656` | `0.6642` | `0.4825` |
| within-project | static | **xgboost** (*) | `0.7900` | `0.4504` | `0.3296` | `0.8111` | `0.4269` | `0.4938` | `0.6732` | `0.4808` |
| cross-project | hybrid | **lightgbm** (*) | `0.6730` | `0.3134` | `0.1965` | `0.7018` | `0.3668` | `0.4141` | `0.5990` | `0.3696` |
| cross-project | hybrid | logistic_regression | `0.5579` | `0.2841` | `0.1164` | `0.6092` | `0.2738` | `0.4644` | `0.5627` | `0.3048` |
| cross-project | hybrid | **random_forest** (*) | `0.7057` | `0.2978` | `0.2160` | `0.7315` | `0.4302` | `0.3708` | `0.6008` | `0.3927` |
| cross-project | hybrid | **xgboost** (*) | `0.6725` | `0.3280` | `0.2026` | `0.6900` | `0.3603` | `0.4476` | `0.6047` | `0.3663` |
| cross-project | static | **lightgbm** (*) | `0.6630` | `0.3012` | `0.1797` | `0.7468` | `0.3344` | `0.3513` | `0.5869` | `0.3144` |
| cross-project | static | logistic_regression | `0.6179` | `0.3063` | `0.1705` | `0.7056` | `0.2932` | `0.4308` | `0.5961` | `0.2917` |
| cross-project | static | **random_forest** (*) | `0.6880` | `0.2898` | `0.1867` | `0.7584` | `0.3648` | `0.3216` | `0.5845` | `0.3255` |
| cross-project | static | **xgboost** (*) | `0.6656` | `0.3116` | `0.1839` | `0.7342` | `0.3298` | `0.3938` | `0.5938` | `0.3168` |
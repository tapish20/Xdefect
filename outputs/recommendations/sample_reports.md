# Refactoring Recommendation Reports (Sample High-Risk Modules)

This report demonstrates end-to-end traceability from **Defect Risk Prediction $\rightarrow$ SHAP Feature Attribution $\rightarrow$ Actionable Refactoring Recommendations** to address **Research Question 4 (RQ4)**.

---

## Module Report #1: `cm1_test_row_123`
- **Dataset**: `cm1` (static feature set)
- **Model Used**: `xgboost`
- **Predicted Risk Score**: `0.9079` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `LOC_COMMENTS` | `191.0000` | `+2.6531` |
| `LOC_CODE_AND_COMMENT` | `37.0000` | `-1.0710` |
| `MAINTENANCE_SEVERITY` | `0.3900` | `+0.4620` |
| `HALSTEAD_LEVEL` | `0.0100` | `-0.4088` |
| `DESIGN_DENSITY` | `0.6600` | `-0.3955` |

### Actionable Refactoring Recommendations
#### Category: Complexity & Size Management
> **Recommendation**: High complexity (CYCLOMATIC_COMPLEXITY=70.0 (P75=8.0)) combined with large module size (LOC_BLANK=164.0 (P75=20.0), LOC_EXECUTABLE=361.0 (P75=47.0), NUMBER_OF_LINES=764.0 (P75=90.5)). Consider splitting this module/class into smaller, decoupled units using 'Extract Method' or 'Extract Class' refactorings.

*Traceability (Triggered Features)*:
- Feature `CYCLOMATIC_COMPLEXITY` = `70.00` exceeding 75th percentile threshold (`8.00`) with SHAP impact `+0.0994`
- Feature `LOC_BLANK` = `164.00` exceeding 75th percentile threshold (`20.00`) with SHAP impact `-0.0422`
- Feature `LOC_EXECUTABLE` = `361.00` exceeding 75th percentile threshold (`47.00`) with SHAP impact `+0.2125`
- Feature `NUMBER_OF_LINES` = `764.00` exceeding 75th percentile threshold (`90.50`) with SHAP impact `-0.0002`

---

## Module Report #2: `kc1_test_row_8`
- **Dataset**: `kc1` (static feature set)
- **Model Used**: `random_forest`
- **Predicted Risk Score**: `0.92` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `sumHALSTEAD_PROG_TIME` | `3026.0600` | `+0.0346` |
| `sumHALSTEAD_EFFORT` | `54468.9100` | `+0.0252` |
| `maxNUM_UNIQUE_OPERANDS` | `39.0000` | `+0.0236` |
| `sumHALSTEAD_ERROR_EST` | `0.9000` | `+0.0218` |
| `avgESSENTIAL_COMPLEXITY` | `1.6667` | `+0.0167` |

### Actionable Refactoring Recommendations
#### Category: Coupling & Dependency Reduction
> **Recommendation**: Excessive inter-module coupling and dependency fan-out (COUPLING_BETWEEN_OBJECTS=18.0 (P75=14.0)). Reduce tight coupling by applying Dependency Inversion (DIP) or introducing Facade/Adapter patterns.

*Traceability (Triggered Features)*:
- Feature `COUPLING_BETWEEN_OBJECTS` = `18.00` exceeding 75th percentile threshold (`14.00`) with SHAP impact `+0.0054`

---

## Module Report #3: `pc1_test_row_229`
- **Dataset**: `pc1` (static feature set)
- **Model Used**: `random_forest`
- **Predicted Risk Score**: `0.885` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `LOC_BLANK` | `12.0000` | `+0.0698` |
| `HALSTEAD_CONTENT` | `44.5600` | `+0.0442` |
| `NORMALIZED_CYLOMATIC_COMPLEXITY` | `0.1100` | `+0.0419` |
| `NUM_UNIQUE_OPERANDS` | `27.0000` | `+0.0388` |
| `DESIGN_DENSITY` | `1.0000` | `-0.0357` |

### Actionable Refactoring Recommendations
#### Category: Complexity & Size Management
> **Recommendation**: Large module size (LOC_BLANK=12.0 (P75=10.0)). Decompose class into focused sub-components to adhere to Single Responsibility Principle (SRP).

*Traceability (Triggered Features)*:
- Feature `LOC_BLANK` = `12.00` exceeding 75th percentile threshold (`10.00`) with SHAP impact `+0.0698`

---

## Module Report #4: `pc3_test_row_182`
- **Dataset**: `pc3` (static feature set)
- **Model Used**: `random_forest`
- **Predicted Risk Score**: `0.92` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `LOC_BLANK` | `28.0000` | `+0.0687` |
| `NUM_UNIQUE_OPERANDS` | `26.0000` | `+0.0456` |
| `LOC_CODE_AND_COMMENT` | `4.0000` | `+0.0395` |
| `DECISION_DENSITY` | `2.5000` | `+0.0268` |
| `NUMBER_OF_LINES` | `78.0000` | `+0.0258` |

### Actionable Refactoring Recommendations
#### Category: Complexity & Size Management
> **Recommendation**: Large module size (LOC_BLANK=28.0 (P75=8.5), NUMBER_OF_LINES=78.0 (P75=44.0)). Decompose class into focused sub-components to adhere to Single Responsibility Principle (SRP).

*Traceability (Triggered Features)*:
- Feature `LOC_BLANK` = `28.00` exceeding 75th percentile threshold (`8.50`) with SHAP impact `+0.0687`
- Feature `NUMBER_OF_LINES` = `78.00` exceeding 75th percentile threshold (`44.00`) with SHAP impact `+0.0258`

---

## Module Report #5: `aeeem_eclipse_test_row_300`
- **Dataset**: `aeeem_eclipse` (static feature set)
- **Model Used**: `random_forest`
- **Predicted Risk Score**: `0.99` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `wmc` | `1042.0000` | `+0.1246` |
| `rfc` | `1095.0000` | `+0.1002` |
| `numberOfLinesOfCode` | `6113.0000` | `+0.0946` |
| `cbo` | `50.0000` | `+0.0691` |
| `fanOut` | `37.0000` | `+0.0481` |

### Actionable Refactoring Recommendations
#### Category: Complexity & Size Management
> **Recommendation**: High complexity (wmc=1042.0 (P75=50.0)) combined with large module size (numberOfLinesOfCode=6113.0 (P75=192.0)). Consider splitting this module/class into smaller, decoupled units using 'Extract Method' or 'Extract Class' refactorings.

*Traceability (Triggered Features)*:
- Feature `wmc` = `1042.00` exceeding 75th percentile threshold (`50.00`) with SHAP impact `+0.1246`
- Feature `numberOfLinesOfCode` = `6113.00` exceeding 75th percentile threshold (`192.00`) with SHAP impact `+0.0946`

#### Category: Coupling & Dependency Reduction
> **Recommendation**: Excessive inter-module coupling and dependency fan-out (cbo=50.0 (P75=14.0), fanOut=37.0 (P75=10.0), rfc=1095.0 (P75=70.0)). Reduce tight coupling by applying Dependency Inversion (DIP) or introducing Facade/Adapter patterns.

*Traceability (Triggered Features)*:
- Feature `cbo` = `50.00` exceeding 75th percentile threshold (`14.00`) with SHAP impact `+0.0691`
- Feature `fanOut` = `37.00` exceeding 75th percentile threshold (`10.00`) with SHAP impact `+0.0481`
- Feature `rfc` = `1095.00` exceeding 75th percentile threshold (`70.00`) with SHAP impact `+0.1002`

#### Category: API Surface & Hierarchy Refactoring
> **Recommendation**: Large method API surface / inherited complexity (numberOfMethods=62.0 (P75=14.0), numberOfPublicMethods=39.0 (P75=9.0)). Consider favoring composition over inheritance and segregating interfaces (Interface Segregation Principle).

*Traceability (Triggered Features)*:
- Feature `numberOfMethods` = `62.00` exceeding 75th percentile threshold (`14.00`) with SHAP impact `+0.0021`
- Feature `numberOfPublicMethods` = `39.00` exceeding 75th percentile threshold (`9.00`) with SHAP impact `-0.0144`

---

## Module Report #6: `aeeem_eclipse_hybrid_test_row_767`
- **Dataset**: `aeeem_eclipse_hybrid` (hybrid feature set)
- **Model Used**: `random_forest`
- **Predicted Risk Score**: `0.94` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `CvsWEntropy` | `0.3541` | `+0.0696` |
| `commit_count` | `150.0000` | `+0.0499` |
| `wmc` | `168.0000` | `+0.0495` |
| `CvsEntropy` | `27.3399` | `+0.0488` |
| `numberOfLinesOfCode` | `547.0000` | `+0.0394` |

### Actionable Refactoring Recommendations
#### Category: Complexity & Size Management
> **Recommendation**: High complexity (wmc=168.0 (P75=50.0)) combined with large module size (numberOfLinesOfCode=547.0 (P75=192.0)). Consider splitting this module/class into smaller, decoupled units using 'Extract Method' or 'Extract Class' refactorings.

*Traceability (Triggered Features)*:
- Feature `wmc` = `168.00` exceeding 75th percentile threshold (`50.00`) with SHAP impact `+0.0495`
- Feature `numberOfLinesOfCode` = `547.00` exceeding 75th percentile threshold (`192.00`) with SHAP impact `+0.0394`

#### Category: Coupling & Dependency Reduction
> **Recommendation**: Excessive inter-module coupling and dependency fan-out (cbo=20.0 (P75=14.0), fanOut=18.0 (P75=10.0), rfc=197.0 (P75=70.0)). Reduce tight coupling by applying Dependency Inversion (DIP) or introducing Facade/Adapter patterns.

*Traceability (Triggered Features)*:
- Feature `cbo` = `20.00` exceeding 75th percentile threshold (`14.00`) with SHAP impact `+0.0215`
- Feature `fanOut` = `18.00` exceeding 75th percentile threshold (`10.00`) with SHAP impact `+0.0163`
- Feature `rfc` = `197.00` exceeding 75th percentile threshold (`70.00`) with SHAP impact `+0.0184`

#### Category: Process Churn & Modification Risk
> **Recommendation**: High code churn and modification entropy (code_churn=214.00 (P75=188.00), commit_count=150.00 (P75=52.00), CvsLinEntropy=0.20 (P75=0.15), CvsLogEntropy=8.64 (P75=8.17), CvsExpEntropy=0.31 (P75=0.23)). This module experiences frequent, volatile changes; prioritize for mandatory senior code review and expanded regression test suites.

*Traceability (Triggered Features)*:
- Feature `code_churn` = `214.00` exceeding 75th percentile threshold (`188.00`) with SHAP impact `-0.0062`
- Feature `commit_count` = `150.00` exceeding 75th percentile threshold (`52.00`) with SHAP impact `+0.0499`
- Feature `CvsLinEntropy` = `0.20` exceeding 75th percentile threshold (`0.15`) with SHAP impact `+0.0145`
- Feature `CvsLogEntropy` = `8.64` exceeding 75th percentile threshold (`8.17`) with SHAP impact `+0.0228`
- Feature `CvsExpEntropy` = `0.31` exceeding 75th percentile threshold (`0.23`) with SHAP impact `+0.0173`

#### Category: API Surface & Hierarchy Refactoring
> **Recommendation**: Large method API surface / inherited complexity (numberOfMethods=22.0 (P75=14.0), numberOfMethodsInherited=85.0 (P75=73.0), numberOfPublicMethods=22.0 (P75=9.0)). Consider favoring composition over inheritance and segregating interfaces (Interface Segregation Principle).

*Traceability (Triggered Features)*:
- Feature `numberOfMethods` = `22.00` exceeding 75th percentile threshold (`14.00`) with SHAP impact `-0.0043`
- Feature `numberOfMethodsInherited` = `85.00` exceeding 75th percentile threshold (`73.00`) with SHAP impact `+0.0030`
- Feature `numberOfPublicMethods` = `22.00` exceeding 75th percentile threshold (`9.00`) with SHAP impact `-0.0065`

---

## Module Report #7: `aeeem_equinox_hybrid_test_row_323`
- **Dataset**: `aeeem_equinox_hybrid` (hybrid feature set)
- **Model Used**: `random_forest`
- **Predicted Risk Score**: `0.955` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `CvsExpEntropy` | `0.1127` | `+0.0513` |
| `CvsLinEntropy` | `0.0516` | `+0.0510` |
| `lines_added` | `889.0000` | `+0.0469` |
| `CvsLogEntropy` | `0.4353` | `+0.0376` |
| `CvsWEntropy` | `0.4348` | `+0.0285` |

### Actionable Refactoring Recommendations
#### Category: Complexity & Size Management
> **Recommendation**: High complexity (wmc=106.0 (P75=36.0)) combined with large module size (numberOfLinesOfCode=332.0 (P75=130.0)). Consider splitting this module/class into smaller, decoupled units using 'Extract Method' or 'Extract Class' refactorings.

*Traceability (Triggered Features)*:
- Feature `wmc` = `106.00` exceeding 75th percentile threshold (`36.00`) with SHAP impact `+0.0183`
- Feature `numberOfLinesOfCode` = `332.00` exceeding 75th percentile threshold (`130.00`) with SHAP impact `+0.0227`

#### Category: Coupling & Dependency Reduction
> **Recommendation**: Excessive inter-module coupling and dependency fan-out (cbo=37.0 (P75=13.2), fanOut=30.0 (P75=10.0), rfc=206.0 (P75=61.5)). Reduce tight coupling by applying Dependency Inversion (DIP) or introducing Facade/Adapter patterns.

*Traceability (Triggered Features)*:
- Feature `cbo` = `37.00` exceeding 75th percentile threshold (`13.25`) with SHAP impact `+0.0100`
- Feature `fanOut` = `30.00` exceeding 75th percentile threshold (`10.00`) with SHAP impact `+0.0039`
- Feature `rfc` = `206.00` exceeding 75th percentile threshold (`61.50`) with SHAP impact `+0.0051`

#### Category: Process Churn & Modification Risk
> **Recommendation**: High code churn and modification entropy (code_churn=798.00 (P75=110.75), commit_count=21.00 (P75=14.00), CvsLinEntropy=0.05 (P75=0.02), CvsLogEntropy=0.44 (P75=0.21), CvsExpEntropy=0.11 (P75=0.06)). This module experiences frequent, volatile changes; prioritize for mandatory senior code review and expanded regression test suites.

*Traceability (Triggered Features)*:
- Feature `code_churn` = `798.00` exceeding 75th percentile threshold (`110.75`) with SHAP impact `+0.0051`
- Feature `commit_count` = `21.00` exceeding 75th percentile threshold (`14.00`) with SHAP impact `+0.0282`
- Feature `CvsLinEntropy` = `0.05` exceeding 75th percentile threshold (`0.02`) with SHAP impact `+0.0510`
- Feature `CvsLogEntropy` = `0.44` exceeding 75th percentile threshold (`0.21`) with SHAP impact `+0.0376`
- Feature `CvsExpEntropy` = `0.11` exceeding 75th percentile threshold (`0.06`) with SHAP impact `+0.0513`

#### Category: API Surface & Hierarchy Refactoring
> **Recommendation**: Large method API surface / inherited complexity (numberOfMethods=40.0 (P75=12.0), numberOfMethodsInherited=30.0 (P75=17.0), numberOfPublicMethods=34.0 (P75=7.0)). Consider favoring composition over inheritance and segregating interfaces (Interface Segregation Principle).

*Traceability (Triggered Features)*:
- Feature `numberOfMethods` = `40.00` exceeding 75th percentile threshold (`12.00`) with SHAP impact `+0.0188`
- Feature `numberOfMethodsInherited` = `30.00` exceeding 75th percentile threshold (`17.00`) with SHAP impact `+0.0114`
- Feature `numberOfPublicMethods` = `34.00` exceeding 75th percentile threshold (`7.00`) with SHAP impact `+0.0050`

---

## Module Report #8: `aeeem_mylyn_hybrid_test_row_726`
- **Dataset**: `aeeem_mylyn_hybrid` (hybrid feature set)
- **Model Used**: `lightgbm`
- **Predicted Risk Score**: `0.9938` (High Risk)

### Top SHAP-Contributing Features
| Feature | Actual Value | SHAP Contribution |
|---|---|---|
| `fanOut` | `50.0000` | `+1.1440` |
| `cbo` | `79.0000` | `+0.8502` |
| `rfc` | `381.0000` | `+0.7263` |
| `dit` | `2.0000` | `-0.6714` |
| `numberOfLinesOfCode` | `829.0000` | `+0.6389` |

### Actionable Refactoring Recommendations
#### Category: Complexity & Size Management
> **Recommendation**: High complexity (wmc=127.0 (P75=19.0)) combined with large module size (numberOfLinesOfCode=829.0 (P75=88.0)). Consider splitting this module/class into smaller, decoupled units using 'Extract Method' or 'Extract Class' refactorings.

*Traceability (Triggered Features)*:
- Feature `wmc` = `127.00` exceeding 75th percentile threshold (`19.00`) with SHAP impact `+0.1375`
- Feature `numberOfLinesOfCode` = `829.00` exceeding 75th percentile threshold (`88.00`) with SHAP impact `+0.6389`

#### Category: Coupling & Dependency Reduction
> **Recommendation**: Excessive inter-module coupling and dependency fan-out (cbo=79.0 (P75=9.0), fanOut=50.0 (P75=6.0), rfc=381.0 (P75=37.0)). Reduce tight coupling by applying Dependency Inversion (DIP) or introducing Facade/Adapter patterns.

*Traceability (Triggered Features)*:
- Feature `cbo` = `79.00` exceeding 75th percentile threshold (`9.00`) with SHAP impact `+0.8502`
- Feature `fanOut` = `50.00` exceeding 75th percentile threshold (`6.00`) with SHAP impact `+1.1440`
- Feature `rfc` = `381.00` exceeding 75th percentile threshold (`37.00`) with SHAP impact `+0.7263`

#### Category: Process Churn & Modification Risk
> **Recommendation**: High code churn and modification entropy (code_churn=517.00 (P75=12.75), commit_count=101.00 (P75=12.00), CvsLinEntropy=0.25 (P75=0.17), CvsLogEntropy=9.48 (P75=9.28), CvsExpEntropy=0.55 (P75=0.30)). This module experiences frequent, volatile changes; prioritize for mandatory senior code review and expanded regression test suites.

*Traceability (Triggered Features)*:
- Feature `code_churn` = `517.00` exceeding 75th percentile threshold (`12.75`) with SHAP impact `+0.6011`
- Feature `commit_count` = `101.00` exceeding 75th percentile threshold (`12.00`) with SHAP impact `+0.1394`
- Feature `CvsLinEntropy` = `0.25` exceeding 75th percentile threshold (`0.17`) with SHAP impact `+0.1261`
- Feature `CvsLogEntropy` = `9.48` exceeding 75th percentile threshold (`9.28`) with SHAP impact `-0.1115`
- Feature `CvsExpEntropy` = `0.55` exceeding 75th percentile threshold (`0.30`) with SHAP impact `+0.0880`

#### Category: API Surface & Hierarchy Refactoring
> **Recommendation**: Large method API surface / inherited complexity (numberOfMethods=59.0 (P75=9.0), numberOfPublicMethods=40.0 (P75=7.0)). Consider favoring composition over inheritance and segregating interfaces (Interface Segregation Principle).

*Traceability (Triggered Features)*:
- Feature `numberOfMethods` = `59.00` exceeding 75th percentile threshold (`9.00`) with SHAP impact `-0.0406`
- Feature `numberOfPublicMethods` = `40.00` exceeding 75th percentile threshold (`7.00`) with SHAP impact `-0.3744`

---

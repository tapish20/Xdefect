# Counterfactual ('What-If?') Refactoring Reports

This report provides exact minimal metric modifications required to convert high-risk defect-prone modules into safe modules ($y_{\text{score}} < 0.35$).

---

## Counterfactual Case #1: `cm1_row_297`
- **Dataset**: `cm1`
- **Initial Predicted Defect Risk**: `0.8150` (HIGH RISK)
- **Target Defect Risk**: `0.3500` (SAFE)
- **Post-Counterfactual Risk**: `0.2900`

### Required Metric Modifications ('What-If' Recalibration)
| Feature | Original Value | Target Value | Reduction Required | % Reduction |
|---|---|---|---|---|
| `LOC_BLANK` | `39.00` | `31.20` | `7.80` | `20.0%` |
| `BRANCH_COUNT` | `37.00` | `29.60` | `7.40` | `20.0%` |
| `CALL_PAIRS` | `6.00` | `4.80` | `1.20` | `20.0%` |
| `LOC_CODE_AND_COMMENT` | `6.00` | `4.80` | `1.20` | `20.0%` |
| `LOC_COMMENTS` | `51.00` | `40.80` | `10.20` | `20.0%` |

---

## Counterfactual Case #2: `kc1_row_8`
- **Dataset**: `kc1`
- **Initial Predicted Defect Risk**: `0.9200` (HIGH RISK)
- **Target Defect Risk**: `0.3500` (SAFE)
- **Post-Counterfactual Risk**: `0.2050`

### Required Metric Modifications ('What-If' Recalibration)
| Feature | Original Value | Target Value | Reduction Required | % Reduction |
|---|---|---|---|---|
| `COUPLING_BETWEEN_OBJECTS` | `18.00` | `7.20` | `10.80` | `60.0%` |
| `DEPTH` | `2.00` | `0.80` | `1.20` | `60.0%` |
| `LACK_OF_COHESION_OF_METHODS` | `100.00` | `40.00` | `60.00` | `60.0%` |
| `NUM_OF_CHILDREN` | `1.00` | `0.40` | `0.60` | `60.0%` |
| `DEP_ON_CHILD` | `1.00` | `0.40` | `0.60` | `60.0%` |

---

## Counterfactual Case #3: `pc1_row_229`
- **Dataset**: `pc1`
- **Initial Predicted Defect Risk**: `0.8850` (HIGH RISK)
- **Target Defect Risk**: `0.3500` (SAFE)
- **Post-Counterfactual Risk**: `0.2200`

### Required Metric Modifications ('What-If' Recalibration)
| Feature | Original Value | Target Value | Reduction Required | % Reduction |
|---|---|---|---|---|
| `LOC_BLANK` | `12.00` | `9.60` | `2.40` | `20.0%` |
| `BRANCH_COUNT` | `7.00` | `5.60` | `1.40` | `20.0%` |
| `CALL_PAIRS` | `1.00` | `0.80` | `0.20` | `20.0%` |
| `CONDITION_COUNT` | `12.00` | `9.60` | `2.40` | `20.0%` |
| `CYCLOMATIC_COMPLEXITY` | `4.00` | `3.20` | `0.80` | `20.0%` |

---

## Counterfactual Case #4: `aeeem_eclipse_hybrid_row_300`
- **Dataset**: `aeeem_eclipse_hybrid`
- **Initial Predicted Defect Risk**: `0.9400` (HIGH RISK)
- **Target Defect Risk**: `0.3500` (SAFE)
- **Post-Counterfactual Risk**: `0.6550`

### Required Metric Modifications ('What-If' Recalibration)
| Feature | Original Value | Target Value | Reduction Required | % Reduction |
|---|---|---|---|---|
| `cbo` | `50.00` | `12.50` | `37.50` | `75.0%` |
| `fanIn` | `18.00` | `4.50` | `13.50` | `75.0%` |
| `fanOut` | `37.00` | `9.25` | `27.75` | `75.0%` |
| `lcom` | `1891.00` | `472.75` | `1418.25` | `75.0%` |
| `numberOfAttributes` | `22.00` | `5.50` | `16.50` | `75.0%` |

---

## Counterfactual Case #5: `aeeem_equinox_hybrid_row_151`
- **Dataset**: `aeeem_equinox_hybrid`
- **Initial Predicted Defect Risk**: `0.9550` (HIGH RISK)
- **Target Defect Risk**: `0.3500` (SAFE)
- **Post-Counterfactual Risk**: `0.2400`

### Required Metric Modifications ('What-If' Recalibration)
| Feature | Original Value | Target Value | Reduction Required | % Reduction |
|---|---|---|---|---|
| `cbo` | `23.00` | `2.30` | `20.70` | `90.0%` |
| `dit` | `1.00` | `0.10` | `0.90` | `90.0%` |
| `fanIn` | `3.00` | `0.30` | `2.70` | `90.0%` |
| `fanOut` | `22.00` | `2.20` | `19.80` | `90.0%` |
| `lcom` | `630.00` | `63.00` | `567.00` | `90.0%` |

---

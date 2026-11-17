# heart-failure-less-is-more
# Heart Failure Mortality Prediction: Proving That Fewer Features Build Stronger Models

![Summary Visual](assets/summary_chart.png)

## 📌 Project Overview & Modeling Philosophy
My core philosophy in machine learning is simple: **the best models don’t need more features—they need fewer, better ones.**

In healthcare datasets with small sample sizes, feeding every available clinical column into an algorithm often introduces noise, inflates cross-validation variance, and obscures real biological signals. The goal of this project was to test that principle on the **UCI Heart Failure Clinical Records Dataset** (`n = 299` patients) by predicting short-term patient mortality (`DEATH_EVENT`) using a minimal, high-signal feature set.

### Key Outcomes
1. **Prevented Real-World Data Leakage:** Identified and removed the `time` column (`r = -0.53` with `DEATH_EVENT`), which leaks survival duration and renders a model unusable for real-time hospital triage.
2. **Proved the "Less is More" Hypothesis:** Stripping the model down from **11 clinical features** to just **2 core biomarkers** (`serum_creatinine` and `ejection_fraction`) increased high-risk patient recall (`DEATH_EVENT = 1`) from **48% to 60%** and reduced 5-fold cross-validation variance by **57%**.
3. **Validated Peer-Reviewed Research:** Independently replicated the findings of Davide Chicco & Giuseppe Jurman (*BMC Medical Informatics and Decision Making*, 2020), confirming that kidney function and heart pumping efficiency alone carry the primary predictive signal for heart failure survival.

---

## 📊 Dataset Description
- **Source:** [UCI Machine Learning Repository — Heart Failure Clinical Records](https://uci.edu)
- **Cohort:** 299 hospitalized patients with left ventricular systolic dysfunction (NYHA Class III & IV).
- **Target Variable:** `DEATH_EVENT` (`0` = Survived follow-up period [203 patients], `1` = Died during follow-up period [96 patients]).

### Clinical Features

| Feature | Description | Unit / Encoding | Role in Final Model |
| :--- | :--- | :--- | :--- |
| **`serum_creatinine`** | Level of creatinine waste in the blood (kidney stress) | `mg/dL` | **Selected (Top 2)** |
| **`ejection_fraction`** | Percentage of blood leaving the heart per contraction | `%` | **Selected (Top 2)** |
| `age` | Age of the patient | `Years` | Baseline only |
| `anaemia` | Decrease of red blood cells or hemoglobin | `0` = No, `1` = Yes | Baseline only |
| `creatinine_phosphokinase` | Level of the CPK enzyme in the blood | `mcg/L` | Baseline only |
| `diabetes` | Whether the patient has diabetes | `0` = No, `1` = Yes | Baseline only |
| `high_blood_pressure` | Whether the patient has hypertension | `0` = No, `1` = Yes | Baseline only |
| `platelets` | Platelet count in the blood | `kiloplatelets/mL` | Baseline only |
| `serum_sodium` | Level of sodium in the blood | `mEq/L` | Baseline only |
| `sex` | Biological sex | `0` = Woman, `1` = Man | Baseline only |
| `smoking` | Whether the patient smokes | `0` = No, `1` = Yes | Baseline only |
| ~~`time`~~ | ~~Follow-up observation period~~ | ~~`Days`~~ | **Dropped (Data Leakage)** |

---

## ⚠️ Catching the `-0.53` Data Leakage Trap
During exploratory data analysis, `time` showed the strongest correlation with `DEATH_EVENT` (`-0.53`), nearly double that of any clinical biomarker. However, `time` records how many days a patient was monitored before either dying (`DEATH_EVENT = 1`) or the study ending (`DEATH_EVENT = 0`). 

Because mortality directly terminates follow-up tracking, `time` is a consequence of the target rather than a predictor. Excluding `time` ensures the model relies strictly on baseline admission tests available when a patient first arrives at the hospital.

---

## 🧪 Experimental Results: 11 Features vs. Top 2 Features
Because medical triage prioritizes catching at-risk patients (`DEATH_EVENT = 1`) on an imbalanced dataset, models were evaluated using **5-fold Cross-Validation (`f1_macro`)** and **Minority Class Recall**, rather than raw accuracy alone.

| Evaluation Metric | Baseline (All 11 Features) | Stripped-Down (Top 2 Features) | Impact |
| :--- | :--- | :--- | :--- |
| **Class `1` Recall (Sensitivity)** | `0.48` (12/25 caught) | **`0.60`** (15/25 caught) | **+12% boost** in catching high-risk patients |
| **Class `1` F1-Score** | `0.56` | **`0.61`** | **+0.05 improvement** on minority class |
| **Test Macro Avg F1** | `0.66` | **`0.67`** | Better class balance |
| **Overall Test Accuracy** | `0.68` | `0.68` | Identical (`41/60` correct) |
| **5-Fold CV `f1_macro` Mean** | `0.703` | `0.696` | Virtually identical (`-0.007`) |
| **5-Fold CV Spread (Max - Min)** | `0.138` (`0.625 – 0.763`) | **`0.059`** (`0.663 – 0.722`) | **57% reduction in variance** |

### Raw Classification Reports

#### 1. Baseline Model (All 11 Clinical Features)
![All Features Classification Report](assets/all_features_report.png)

#### 2. Minimal Model (`serum_creatinine` & `ejection_fraction` Only)
![Top 2 Features Classification Report](assets/top2_features_report.png)

---

## 🚀 How to Run This Project
1. Clone the repository:
   ```bash
   git clone https://github.com
   cd heart-failure-feature-selection
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook heart_failure_analysis.ipynb
   ```

---

## 📚 Academic Reference
- Chicco, D., & Jurman, G. (2020). *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* **BMC Medical Informatics and Decision Making**, 20(1), 16. [DOI: 10.1186/s12911-020-1023-5](https://doi.org)

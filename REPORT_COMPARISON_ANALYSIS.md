# Project Report Comparison Analysis

## Reference Report (Full Marks) vs Our Report

This document provides a detailed comparison between the reference project report (Obesity Classification - Group F) that received full marks and our Maternal Health Risk Classification report.

---

## 1. Executive Summary

| Aspect | Reference Report (Full Marks) | Our Report | Gap Analysis |
|--------|------------------------------|------------|--------------|
| **Pages** | 13 pages | ~12 pages | Similar |
| **Models** | CART + SVM | Random Forest + SVM | Different ML1 model |
| **Classification** | Multi-class (7 classes) | Binary (2 classes) | Simpler problem |
| **Data Split** | 60/20/20 (manual) | 80/20 with 10-fold CV | Different approach |
| **XAI Methods** | Confusion matrix heatmaps only | Feature Importance, PDPs, LIME | **We have MORE XAI** |
| **Mathematical Depth** | Detailed with formulas | Detailed with formulas | Similar |

---

## 2. Section-by-Section Comparison

### 2.1 Introduction (4 Points)

#### Reference Report:
- General introduction about obesity as a health problem
- Links to external resources for more information
- Dataset source citation with links
- **Shows first 10 rows of data** in tables
- Lists all 17 attributes with descriptions
- Explains data collection method (77% synthetic via SMOTE)

#### Our Report:
- Problem statement about maternal mortality
- Rationale for binary classification
- Dataset source citation [@UCI_maternal]
- Variable description table with ranges
- Abstract in YAML header

#### **GAP IDENTIFIED:**
- [ ] **Missing: Show sample data rows** (first 5-10 rows of dataset)
- [x] We have abstract (they don't have one in YAML)
- [x] We explain binary conversion rationale (good addition)

---

### 2.2 Mathematical Overview (4 Points)

#### Reference Report:
- **CART Model:**
  - Gini Impurity formula with explanation
  - Tree construction explanation
  - Prediction formula
  - Hyperparameter tuning (cp) explanation
  - Model evaluation (accuracy formula)

- **SVM Model:**
  - Linear SVM with margin formula
  - Objective function and constraints
  - Non-linear SVM with kernel trick
  - RBF and Linear kernel formulas
  - Hyperparameters explanation (C, gamma)

#### Our Report:
- **Random Forest:**
  - Bootstrap sampling explanation
  - Feature randomization
  - Prediction formula (mode)
  - Gini Impurity formula
  - Key hyperparameters listed

- **SVM:**
  - Optimization problem formula
  - Soft-margin constraints
  - RBF kernel formula
  - Hyperparameters explanation

#### **GAP IDENTIFIED:**
- [x] Both have similar mathematical depth
- [x] We include formulas with proper LaTeX
- [ ] **Consider adding: Accuracy formula explicitly**

---

### 2.3 Model Fitting Process (6 Points)

#### Reference Report:
- **Data Splitting Section:**
  - Explicit 60/20/20 split code shown
  - Manual splitting with sample() function
  - Clear explanation of purpose for each set

- **CART Training:**
  - Initial model with default cp = 0.01
  - Baseline accuracy: 87.2%
  - Hyperparameter tuning loop code shown
  - Best cp = 0.001, accuracy = 90.76%
  - Final model trained on train + validation

- **SVM Training:**
  - Initial model with default cost=1, gamma=0.1
  - Baseline accuracy: 89.57%
  - Grid search code shown
  - Best: Cost=100, Gamma=0.01, accuracy = 95.26%
  - Final model trained on train + validation

#### Our Report:
- **Data Splitting:**
  - 80/20 split with CV (professor approved)
  - Uses createDataPartition (stratified)
  - Explanation of professor's guidelines

- **Class Imbalance Handling:**
  - Oversampling code shown
  - Before/after class distribution

- **Model Training:**
  - 3 models trained (RF, DT, SVM)
  - 10-fold CV for hyperparameter tuning
  - Grid search with caret's train()
  - Model selection based on CV AUC-ROC

#### **GAP IDENTIFIED:**
- [ ] **Missing: Show baseline accuracy BEFORE tuning**
- [ ] **Missing: Show accuracy AFTER tuning (improvement)**
- [x] We show hyperparameter tuning plots (they don't)
- [x] We explain "Why RF and SVM?" (good addition)
- [x] We handle class imbalance (they don't mention this)

---

### 2.4 Assessment & Method Comparison (Part of Section 6P)

#### Reference Report:
- Test predictions code shown
- CART test accuracy: 93.85%
- SVM test accuracy: 94.80%
- Comparison paragraph for each model
- Strengths and weaknesses discussed

#### Our Report:
- Test predictions with confusionMatrix()
- Confusion matrices as heatmaps (side-by-side)
- Metrics table (Accuracy, Sensitivity, Specificity, Precision, F1)
- ROC curves with AUC values
- Bar plot comparison
- Final comparison table

#### **GAP IDENTIFIED:**
- [x] We have MORE metrics (Sensitivity, Specificity, F1, AUC)
- [x] We have ROC curves (they don't)
- [x] We have better visualizations
- [ ] **Consider: Add explicit comparison paragraph like theirs**

---

### 2.5 XAI / Interpretability (6 Points)

#### Reference Report:
- **Only confusion matrix heatmaps**
- Interpretation of heatmap (diagonal = correct, off-diagonal = errors)
- No feature importance
- No PDPs
- No LIME

#### Our Report:
- **Feature Importance:** RF Gini + SVM Permutation (side-by-side)
- **PDPs:** 6 plots (3 features x 2 models)
- **LIME:** Side-by-side comparison for same case
- Interpretation text for each XAI method

#### **GAP IDENTIFIED:**
- [x] **WE ARE SIGNIFICANTLY BETTER HERE**
- [x] We cover all 3 required XAI types (Feature Importance, Feature Effects, Microscopic)
- [x] We compare XAI results across both models
- The reference report is WEAK on XAI (only confusion matrices)

---

### 2.6 Graphical Presentation (4 Points)

#### Reference Report:
- Data tables (first 10 rows split into 3 tables)
- Confusion matrix heatmaps (2)
- Total: ~4-5 figures/tables

#### Our Report:
- Target distribution + Boxplots (combined)
- Correlation matrix
- Summary statistics table
- Hyperparameter tuning plots (2)
- Confusion matrices (2)
- ROC curves
- Performance bar chart
- Feature importance (2)
- PDPs (6)
- LIME plots (2)
- Final comparison table
- **Total: 10+ figures/tables**

#### **GAP IDENTIFIED:**
- [x] **WE HAVE SIGNIFICANTLY MORE VISUALIZATIONS**
- [x] All figures have captions
- [ ] **Consider: Add sample data table like theirs**

---

### 2.7 Bibliography (2 Points)

#### Reference Report:
- 3 references
- Numbered citation style [1], [2], [3]
- Dataset source, research paper, SVM paper

#### Our Report:
- references.bib file
- [@key] citation style
- Dataset source, textbooks, R packages

#### **GAP IDENTIFIED:**
- [x] Both have proper citations
- [x] We use BibTeX (more professional)

---

### 2.8 Code Quality (4 Points)

#### Reference Report:
- Code snippets shown inline in report
- Comments in code
- Clear variable names

#### Our Report:
- echo = FALSE (code hidden in PDF)
- Separate R chunks with labels
- Comments in code
- Well-structured

#### **GAP IDENTIFIED:**
- [ ] **Consider: Show some key code snippets** (data split, model training)
- [x] Our code is well-organized with chunk labels

---

## 3. What Reference Report Has That We're Missing

| Item | Priority | Action Needed |
|------|----------|---------------|
| Sample data rows table | Medium | Add table showing first 5-10 rows |
| Baseline vs tuned accuracy comparison | High | Show accuracy before/after tuning |
| Explicit code snippets | Medium | Consider showing key code in report |
| Accuracy formula in math section | Low | Optional - we have other metrics |

---

## 4. What We Have That Reference Report Doesn't

| Item | Value |
|------|-------|
| Abstract | Professional touch |
| Class imbalance handling | Critical for medical data |
| "Why these models?" rationale | Shows understanding |
| ROC curves with AUC | Standard evaluation metric |
| Feature Importance plots | XAI requirement |
| Partial Dependence Plots | XAI requirement |
| LIME explanations | XAI requirement |
| Cross-model XAI comparison | Shows depth |
| Clinical recommendations | Domain relevance |
| Multiple performance metrics | Comprehensive evaluation |

---

## 5. Grading Criteria Compliance

### Based on Professor's Requirements:

| Section | Points | Reference Report | Our Report |
|---------|--------|------------------|------------|
| a) Introduction + Descriptive Analysis | 4P | Good | Good |
| b) Mathematical Overview (2 methods) | 4P | Good | Good |
| c) Fitting Process + Hyperparameters + Comparison | 6P | Good | Good (missing baseline comparison) |
| d) XAI (Feature Imp, Effects, Microscopic) | 6P | **WEAK** (only confusion matrix) | **EXCELLENT** |
| e) Graphical Presentation | 4P | Adequate | Excellent |
| f) Bibliography | 2P | Good | Good |
| R Code Quality | 4P | Good (shown inline) | Good (hidden but structured) |

---

## 6. Recommendations for Improvement

### High Priority:
1. **Add baseline accuracy before tuning** - Show model performance with default hyperparameters, then show improvement after tuning
2. **Show sample data** - Add a table with first 5 rows of the dataset

### Medium Priority:
3. **Show key code snippets** - Data split and model training code could be shown (set echo=TRUE for specific chunks)
4. **Add explicit comparison paragraph** - Like reference report's "CART achieved 93.85%... SVM achieved 94.80%..."

### Low Priority (Already Strong):
5. XAI section is already better than reference
6. Visualizations are already comprehensive
7. Mathematical overview is adequate

---

## 7. Overall Assessment

| Criteria | Reference Report | Our Report | Winner |
|----------|------------------|------------|--------|
| Introduction | Good | Good | Tie |
| Math Overview | Good | Good | Tie |
| Model Fitting | Shows baseline | Uses CV | Reference (clearer progression) |
| XAI | Weak | Excellent | **Our Report** |
| Graphics | Adequate | Excellent | **Our Report** |
| Bibliography | Good | Good | Tie |
| Code | Shown inline | Hidden | Reference (more transparent) |

### Conclusion:
**Our report is STRONGER in XAI and visualizations**, which are worth 10 points combined (6P + 4P). The reference report's weakness in XAI (only confusion matrices) would likely lose points under the current grading criteria. Our main areas for improvement are showing baseline vs tuned accuracy comparison and potentially adding sample data rows.

---

## 8. Action Items Checklist

- [ ] Add table showing first 5 rows of dataset
- [ ] Add baseline accuracy (before hyperparameter tuning) for both models
- [ ] Add explicit accuracy improvement statement after tuning
- [ ] Consider showing data split code with echo=TRUE
- [ ] Add comparison paragraph summarizing model performance

---

*Analysis completed: This document should be reviewed manually to determine which improvements to implement.*

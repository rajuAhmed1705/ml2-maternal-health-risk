# Machine Learning 2 Project Implementation Plan
## Maternal Health Risk Classification

**Team Members:** Aisha, Mufaddal, Raju Ahmed
**Deadline:** Saturday, 3rd January 2026 - 11:55 pm
**Dataset:** Maternal Health Risk Data Set (UCI ML Repository)

---


Abstract

Maternal health is a major challenge in Bangladesh, particularly in rural areas where healthcare is hard to
access. Many pregnant women suffer from conditions like high blood pressure and infections that go
unnoticed due to a lack of medical facilities and trained professionals. Early marriages, limited education,
and poverty add to the problem. Women often cannot reach healthcare centers in time, leading to
complications which increases the risk associated with the same. Sometimes this can lead to deaths.
Addressing this issue requires improving access to healthcare and supporting women throughout their
pregnancies.

## 1. Project Overview

### 1.1 Problem Statement
Predict maternal health risk as a **binary classification** (High Risk vs. Not High Risk) during pregnancy using health indicators collected from hospitals and community clinics in rural Bangladesh.

**Important Update (Professor Feedback):** The original dataset contains three ordinal risk levels (low, mid, high). Since ordinal relationships are not optimally captured by standard multi-class classifiers, we aggregate mid and low risk into a single "Not High Risk" category. This binary framing directly addresses the clinical question: *"Is this pregnancy high-risk and requiring immediate attention?"*

### 1.2 Dataset Summary
| Attribute | Description |
|-----------|-------------|
| **Records** | 1,014 observations |
| **Features** | 6 predictor variables |
| **Target** | RiskLevel (Binary: HighRisk vs NotHighRisk) |
| **Source** | UCI ML Repository |
| **Missing Values** | None (0 in all columns) - **NO missing data handling required** |

> **Note on Missing Data:** This dataset has NO missing values. Verified by inspection - all 1,014 rows have complete data for all 7 columns. You do NOT need to implement imputation or missing data handling strategies.

> **Binary Target Conversion:** The original 3-class target (low risk, mid risk, high risk) is converted to binary: "high risk" → HighRisk, "low risk" + "mid risk" → NotHighRisk.

### 1.3 Variables Description

| Variable | Type | Description | Range |
|----------|------|-------------|-------|
| Age | Integer (Continuous) | Age of pregnant woman | 10-70 years |
| SystolicBP | Integer (Continuous) | Systolic blood pressure (mmHg) | 70-160 |
| DiastolicBP | Integer (Continuous) | Diastolic blood pressure (mmHg) | 49-100 |
| BS | Numeric (Continuous) | Blood sugar level (mmol/L) | 6.0-19.0 |
| BodyTemp | Numeric (Continuous) | Body temperature (°F) | 98-103 |
| HeartRate | Integer (Continuous) | Heart rate (bpm) | 7-90 |
| RiskLevel | Character (Binary) | **Target**: HighRisk vs NotHighRisk | 2 classes |

### 1.4 ML Methods Approach

**Strategy: Train 3 models, select best 2 for final comparison**

We initially train three machine learning models:
1. **Random Forest** - Ensemble tree-based method (ML1)
2. **Decision Tree** - Single tree classifier (interpretable baseline)
3. **Support Vector Machine (SVM)** - Kernel-based classifier (ML2)

After cross-validation and initial evaluation, we select the **best 2 models** based on AUC-ROC performance for detailed comparison and XAI analysis. This approach ensures we compare the most effective models while demonstrating understanding of multiple ML techniques.

---

## 2. Project Structure & Deliverables

### 2.1 Required Deliverables
- [ ] Report in PDF format (via R Markdown)
- [ ] Dataset (.csv file)
- [ ] R code (structured and commented .R or .Rmd file)

### 2.2 Report Structure (30 Points Total)

| Section | Points | Description |
|---------|--------|-------------|
| a) Introduction & EDA | 4P | Descriptive data analysis |
| b) Mathematical Overview | 4P | Theory of Random Forest & SVM |
| c) Model Fitting & Comparison | 6P | Training, hyperparameters, evaluation |
| d) Interpretable ML (XAI) | 6P | Feature importance, effects, local explanations |
| e) Graphical Presentation | 4P | Quality visualizations |
| f) Bibliography | 2P | Proper citations |
| R Code Quality | 4P | Structure & comments |
| **Total** | **30P** | |

---

## 3. Detailed Implementation Plan

### Phase 1: Data Preparation & Exploratory Data Analysis (Section a)

#### 1.1 Data Loading and Initial Inspection
```r
# Tasks:
- Load dataset with read.csv()
- Check structure with str()
- Summary statistics with summary()
- Verify no missing values
- Convert RiskLevel to factor with ordered levels
```

#### 1.2 Descriptive Statistics
- **Numerical summaries:** Mean, median, SD, min, max, quartiles for each predictor
- **Target distribution:** Frequency table and proportions of risk levels
- **Check class imbalance:** Visualize distribution of target variable

#### 1.3 Visualizations for EDA
| Plot Type | Purpose |
|-----------|---------|
| Histograms/Density plots | Distribution of each predictor |
| Box plots | Compare predictors across risk levels |
| Correlation matrix/heatmap | Relationships between predictors |
| Pair plots | Bivariate relationships colored by risk level |
| Bar chart | Target variable distribution |

#### 1.4 Data Preprocessing
```r
# Tasks:
# Step 1: Convert to binary classification
data$RiskLevel <- ifelse(data$RiskLevel == "high risk", "HighRisk", "NotHighRisk")
data$RiskLevel <- factor(data$RiskLevel, levels = c("NotHighRisk", "HighRisk"))

# Step 2: Feature scaling for SVM (standardization/normalization)
# No encoding needed (all predictors are numeric)
```

#### 1.5 Outlier Handling Strategy (IMPORTANT)
**Issue Identified:** HeartRate = 7 bpm appears in rows 501 and 910. This is physiologically impossible (normal resting heart rate is 60-100 bpm, minimum viable ~30 bpm).

**Decision Required - Choose ONE approach:**
```r
# Option A: Remove outlier rows (Recommended for small number of outliers)
data <- data[data$HeartRate >= 30, ]  # Remove physiologically impossible values

# Option B: Treat as data entry error and impute
# Replace with median HeartRate for that risk level
data$HeartRate[data$HeartRate < 30] <- median(data$HeartRate[data$HeartRate >= 30])

# Option C: Cap/Winsorize at minimum viable value
data$HeartRate[data$HeartRate < 30] <- 30
```

**Recommendation:** Use **Option A** (remove the 2 rows) since:
- Only 2 out of 1014 observations (~0.2%)
- Values are clearly erroneous, not just extreme
- Imputation would introduce artificial data

**Additional outlier checks to perform:**
```r
# Check all variables for physiologically implausible values
summary(data)
boxplot(data[, 1:6])  # Visual inspection

# Verify ranges are reasonable:
# - Age: 10-70 (some young ages like 10-12 may be valid in this context)
# - SystolicBP: 70-160 (acceptable range)
# - DiastolicBP: 49-100 (acceptable range)
# - BS: 6.0-19.0 (high values indicate gestational diabetes risk)
# - BodyTemp: 98-103°F (acceptable, fever range included)
# - HeartRate: Should be 60-100 normal, but 30-120 acceptable during pregnancy
```

---

### Phase 2: Data Splitting

#### 2.1 Train/Validation/Test Split
```r
# Split ratios as per project requirements:
# - Training: 60%
# - Validation: 20%
# - Test: 20%

set.seed(123)  # For reproducibility

# Method: Use caret::createDataPartition or manual splitting
# Ensure stratified sampling to maintain class proportions
```

#### 2.2 Cross-Validation Approach (RECOMMENDED)
```r
# For a dataset of 1,014 samples, k-fold CV is preferred over hold-out validation
# DECISION: Use k-fold CV for hyperparameter tuning

# Split strategy:
# - 80% for training + CV (811 samples)
# - 20% for final test (203 samples)

set.seed(123)
library(caret)

# Stratified split to maintain class proportions
train_index <- createDataPartition(data$RiskLevel, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Use 10-fold CV within training data for hyperparameter tuning
cv_control <- trainControl(method = "cv", number = 10, classProbs = TRUE)
```

**Why CV over hold-out validation:**
- More robust performance estimates
- Better use of limited data
- Reduces variance in model selection

#### 2.3 Class Imbalance Check and Handling
```r
# Step 1: Check class distribution
table(data$RiskLevel)
prop.table(table(data$RiskLevel))

# Expected approximate distribution (verify with actual data):
# low risk: ~400 (40%)
# mid risk: ~330 (33%)
# high risk: ~280 (27%)
```

**If class imbalance is significant (ratio > 1:3), consider:**
```r
# Option A: Stratified sampling (ALREADY INCLUDED - use createDataPartition)
# This ensures train/test splits maintain class proportions

# Option B: Class weights in models
# For Random Forest:
rf_model <- randomForest(RiskLevel ~ ., data = train_data,
                         classwt = c("low risk" = 1, "mid risk" = 1, "high risk" = 2))

# For SVM:
svm_model <- svm(RiskLevel ~ ., data = train_scaled,
                 class.weights = c("low risk" = 1, "mid risk" = 1, "high risk" = 2))

# Option C: SMOTE (Synthetic Minority Oversampling) - if severe imbalance
library(smotefamily)
# Generally NOT needed for this dataset as imbalance is mild
```

**Recommendation:** Use stratified sampling (already planned) and consider class weights if high-risk recall is poor during initial model evaluation.

---

### Phase 3: Mathematical Overview (Section b)

#### 3.1 Random Forest Theory
Write brief mathematical descriptions covering:
- **Decision Trees:** Recursive partitioning, Gini impurity, information gain
- **Bagging:** Bootstrap aggregating concept
- **Random Forest:** Feature subsampling at each split (mtry parameter)
- **Voting mechanism:** Majority voting for classification
- **Key hyperparameters:** ntree, mtry, nodesize, maxnodes

#### 3.2 Support Vector Machine Theory
Write brief mathematical descriptions covering:
- **Linear SVM:** Maximum margin classifier, support vectors
- **Optimization problem:** Minimize ||w||² subject to constraints
- **Soft margin:** Slack variables (ξ), cost parameter C
- **Kernel trick:** φ(x) mapping, kernel functions K(x,x')
- **Common kernels:** Linear, Polynomial, RBF (Radial Basis Function)
- **Key hyperparameters:** C (cost), gamma (for RBF), degree (for polynomial)

---

### Phase 4: Model Training & Hyperparameter Tuning (Section c)

#### 4.1 Random Forest Implementation
```r
# R packages: randomForest, ranger, or caret

# Step 1: Train baseline model
library(randomForest)
rf_model <- randomForest(RiskLevel ~ ., data = train_data, ntree = 500)

# Step 2: Hyperparameter tuning via validation set or CV
# Parameters to tune:
# - ntree: Number of trees (100, 300, 500, 1000)
# - mtry: Variables at each split (sqrt(p), p/3, etc.)
# - nodesize: Minimum node size (1, 5, 10)
# - maxnodes: Maximum terminal nodes

# Step 3: Use tuneRF() or caret::train() with tuneGrid
```

#### 4.2 SVM Implementation
```r
# R packages: e1071, kernlab, or caret

# Step 1: Feature scaling (CRITICAL for SVM)
# Standardize predictors: scale()
preProcValues <- preProcess(train_data[, -7], method = c("center", "scale"))
train_scaled <- predict(preProcValues, train_data)
test_scaled <- predict(preProcValues, test_data)  # Use SAME scaling parameters!

# Step 2: Kernel Comparison (BEFORE hyperparameter tuning)
library(e1071)
# Compare kernels using CV to select best kernel type
kernels <- c("linear", "radial", "polynomial")
kernel_results <- data.frame(kernel = kernels, accuracy = NA)

for (i in seq_along(kernels)) {
  set.seed(123)
  cv_model <- tune(svm, RiskLevel ~ ., data = train_scaled,
                   kernel = kernels[i],
                   tunecontrol = tune.control(cross = 10))
  kernel_results$accuracy[i] <- 1 - cv_model$best.performance
}
print(kernel_results)  # Select kernel with highest accuracy

# Step 3: Hyperparameter tuning for selected kernel
# For RBF kernel (typically best for this type of data):
# - C (cost): 0.01, 0.1, 1, 10, 100
# - gamma: 0.001, 0.01, 0.1, 1, "auto"

# Use tune() function or caret::train()
tune_result <- tune(svm, RiskLevel ~ ., data = train_scaled,
                    kernel = "radial",
                    ranges = list(cost = c(0.1, 1, 10, 100),
                                  gamma = c(0.01, 0.1, 1)))
best_svm <- tune_result$best.model
```

#### 4.3 Model Selection Process
1. Train multiple model configurations on training data
2. Evaluate on validation data (or use CV)
3. Select best hyperparameters based on validation performance
4. Retrain final model on training + validation data (if using hold-out validation)

---

### Phase 5: Model Evaluation & Comparison (Section c continued)

#### 5.1 Evaluation Metrics for Binary Classification
| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Confusion Matrix (2x2) | TP, TN, FP, FN |
| Sensitivity (Recall) | TP / (TP + FN) - Detection rate for HighRisk |
| Specificity | TN / (TN + FP) - Correct rejection of NotHighRisk |
| Precision (PPV) | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| F1-Score | Harmonic mean of precision & recall |
| AUC-ROC | Area under ROC curve |
| Cohen's Kappa | Agreement accounting for chance |

**CRITICAL: Emphasis on High-Risk Sensitivity**

For maternal health risk classification, **missing a high-risk patient is more costly than a false alarm**. Therefore:

```r
# Priority metrics to report and optimize:
# 1. Sensitivity for HighRisk class - MOST IMPORTANT
#    (What % of actual high-risk patients did we correctly identify?)
# 2. AUC-ROC - overall discrimination ability
# 3. F1-Score - balances precision and recall

# For binary classification, use twoClassSummary
cv_control <- trainControl(method = "cv", number = 10,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

# Train with ROC as metric
model <- train(RiskLevel ~ ., data = train_data, metric = "ROC", ...)

# Extract metrics from binary confusion matrix
cm <- confusionMatrix(predictions, actual, positive = "HighRisk")
sensitivity <- cm$byClass["Sensitivity"]
specificity <- cm$byClass["Specificity"]
```

**Discussion point:** If one model has higher overall accuracy but lower sensitivity, the model with better sensitivity may be preferred in a clinical setting.

#### 5.2 Evaluation on Validation Data
```r
# For model selection and hyperparameter tuning
val_pred_rf <- predict(rf_model, newdata = val_data)
val_pred_svm <- predict(svm_model, newdata = val_data_scaled)

confusionMatrix(val_pred_rf, val_data$RiskLevel)
confusionMatrix(val_pred_svm, val_data$RiskLevel)
```

#### 5.3 Final Comparison on TEST Data ONLY
```r
# IMPORTANT: Only use test data for final model comparison!
test_pred_rf <- predict(final_rf_model, newdata = test_data)
test_pred_svm <- predict(final_svm_model, newdata = test_data_scaled)

# Generate comprehensive comparison
# - Side-by-side confusion matrices
# - Comparative metrics table
# - Statistical significance test (McNemar's test)
```

#### 5.4 Visualizations for Model Comparison
- Confusion matrix heatmaps (side by side, 2x2)
- ROC curves (binary classification: single curve per model)
- AUC comparison
- Performance metrics bar chart comparison
- Prediction agreement between models

---

### Phase 6: Interpretable Machine Learning / XAI (Section d)

#### 6.1 Feature Importance (Global Explanation)

**Random Forest:**
```r
# Built-in importance measures
importance(rf_model)
varImpPlot(rf_model)

# Types: MeanDecreaseGini, MeanDecreaseAccuracy
```

**SVM:**
```r
# Model-agnostic approaches required
# Use permutation importance or SHAP

library(iml)
predictor <- Predictor$new(svm_model, data = X_train, y = y_train)
imp <- FeatureImp$new(predictor, loss = "ce")  # cross-entropy for classification
plot(imp)
```

**Comparison:** Create side-by-side feature importance plots for both models

#### 6.2 Feature Effects (Global Explanation)

**Partial Dependence Plots (PDP):**
```r
library(pdp)
# or library(iml)

# For top important features
partial(rf_model, pred.var = "BS", plot = TRUE)
partial(rf_model, pred.var = "SystolicBP", plot = TRUE)

# 2D PDP for interactions
partial(rf_model, pred.var = c("BS", "SystolicBP"), plot = TRUE)
```

**Accumulated Local Effects (ALE):**
```r
library(iml)
predictor <- Predictor$new(model, data = X, y = y)
ale <- FeatureEffect$new(predictor, feature = "BS", method = "ale")
plot(ale)
```

#### 6.3 Local/Microscopic Explanations

**LIME (Local Interpretable Model-agnostic Explanations):**
```r
library(lime)

# Create explainer
explainer_rf <- lime(train_data[, -7], rf_model)
explainer_svm <- lime(train_scaled[, -7], svm_model)

# SELECT SPECIFIC CASES TO EXPLAIN (2-3 per risk level recommended):
# 1. One correctly classified high-risk case
# 2. One misclassified high-risk case (if any)
# 3. One correctly classified low-risk case
# 4. One "borderline" case (where model was uncertain)

# Find interesting cases
test_predictions <- predict(rf_model, test_data, type = "prob")
# Borderline cases: where max probability < 0.5
borderline_idx <- which(apply(test_predictions, 1, max) < 0.5)

# High-risk cases
high_risk_idx <- which(test_data$RiskLevel == "high risk")

# Explain selected cases (show 2-3 examples in report)
explanation <- explain(test_data[high_risk_idx[1:2], -7],
                       explainer_rf,
                       n_labels = 1,      # Explain predicted class
                       n_features = 6)    # Show all 6 features
plot_features(explanation)
```

**SHAP Values:**
```r
library(iml)  # or fastshap, shapr

# Create predictor wrapper
predictor_rf <- Predictor$new(rf_model, data = train_data[, -7], y = train_data$RiskLevel)

# SHAP for individual predictions
shapley <- Shapley$new(predictor_rf, x.interest = test_data[1, -7])
plot(shapley)

# Show SHAP for 2-3 carefully selected cases (same cases as LIME for comparison)
```

**How many examples to show in report:**
- 2-3 LIME explanations (one per risk level, or focus on high-risk)
- 2-3 SHAP explanations (same cases for consistency)
- Include at least one misclassified case if available (interesting for discussion)

#### 6.4 XAI Comparison Between Models
- Compare feature importance rankings (RF vs SVM)
- Compare PDP shapes for key features
- Compare local explanations for same test cases
- Discuss: Do both models "reason" similarly?

---

### Phase 7: Report Writing & Visualization (Sections e, f)

#### 7.1 Required Visualizations Summary

| Section | Visualizations |
|---------|----------------|
| EDA | Histograms, box plots, correlation heatmap, pair plots |
| Model Training | Learning curves, hyperparameter tuning results |
| Evaluation | Confusion matrices, metrics comparison bar charts |
| XAI | Feature importance plots, PDPs, LIME/SHAP plots |

#### 7.2 Report Writing Guidelines
- Use provided R Markdown template
- Include all code in the .Rmd file
- Ensure code runs without errors
- Add comments explaining each major step
- Use proper citations (BibTeX format)

#### 7.3 Bibliography Requirements
Cite sources for:
- Dataset (UCI repository)
- R packages used (randomForest, e1071, iml, lime, etc.)
- Theoretical references (textbooks, papers)
- Any external code or tutorials used

---

## 4. R Packages Required

```r
# Data manipulation & visualization
install.packages(c("tidyverse", "ggplot2", "dplyr", "corrplot", "GGally"))

# Machine Learning
install.packages(c("caret", "randomForest", "ranger", "e1071", "kernlab"))

# Model evaluation
install.packages(c("pROC", "MLmetrics"))

# Interpretable ML / XAI
install.packages(c("iml", "pdp", "lime", "DALEX", "vip"))

# Reporting
install.packages(c("knitr", "rmarkdown", "kableExtra"))
```

---

## 5. Code Structure

```
project/
├── Maternal Health Risk Data Set.csv    # Dataset
├── ML2_Project_Report.Rmd               # Main R Markdown report
├── references.bib                        # Bibliography file
├── ML2_Project_Report.pdf               # Generated PDF report
└── IMPLEMENTATION_PLAN.md               # This file
```

### Suggested Code Organization in .Rmd

```r
# 1. SETUP & DATA LOADING
#    - Load packages
#    - Load and inspect data
#    - Data preprocessing

# 2. EXPLORATORY DATA ANALYSIS
#    - Summary statistics
#    - Visualizations
#    - Data quality checks

# 3. DATA SPLITTING
#    - Train/Validation/Test split
#    - Stratified sampling

# 4. RANDOM FOREST
#    - Baseline model
#    - Hyperparameter tuning
#    - Final model training
#    - Validation performance

# 5. SUPPORT VECTOR MACHINE
#    - Data scaling
#    - Kernel selection
#    - Hyperparameter tuning
#    - Final model training
#    - Validation performance

# 6. MODEL COMPARISON (TEST SET)
#    - Predictions on test data
#    - Confusion matrices
#    - Performance metrics
#    - Statistical comparison

# 7. INTERPRETABLE ML / XAI
#    - Feature importance (both models)
#    - Partial dependence plots
#    - LIME explanations
#    - Comparison of interpretations

# 8. CONCLUSIONS
#    - Summary of findings
#    - Model recommendations
#    - Limitations
```

---

## 6. Key Points to Remember

### 6.1 Project Requirements Checklist
- [x] Train 3 ML models (Random Forest, Decision Tree, SVM)
- [x] Select best 2 models based on AUC-ROC for final comparison
- [x] Use at least one ML2 method (SVM with RBF kernel)
- [x] Convert to binary classification (HighRisk vs NotHighRisk) per professor feedback
- [x] Handle outliers (HeartRate = 7 in rows 501, 910) - Remove these 2 rows
- [x] Check class distribution and apply stratified sampling
- [x] 80/20 train/test split with 10-fold CV for hyperparameter tuning
- [x] Only use test data for final comparison
- [x] Compare SVM kernels before hyperparameter tuning
- [x] Scale features for SVM (use same scaler for train and test!)
- [x] Include XAI: feature importance, feature effects, local explanations
- [x] Report sensitivity prominently (clinical relevance for binary classification)
- [x] Include ROC curves and AUC comparison for all 3 models
- [x] Show 2-3 LIME/SHAP examples per model
- [x] Include Decision Tree visualization for interpretability
- [x] All code must be reproducible (set.seed() before EVERY random operation)
- [ ] Code must run without errors on lecturer's computer
- [x] No missing data handling needed (dataset is complete)

### 6.2 Common Pitfalls to Avoid
1. **Don't use test data during model selection** - only for final comparison
2. **Scale data for SVM** - SVM is sensitive to feature scales
3. **Use SAME scaler** - fit on training data, apply to test data
4. **Handle class imbalance** - stratified sampling + consider class weights
5. **Don't overfit** - use proper CV validation
6. **Document everything** - comments in code, citations in report
7. **Don't ignore outliers** - the HeartRate = 7 values must be addressed
8. **Don't just report accuracy** - emphasize high-risk recall for clinical relevance
9. **Interpret XAI plots** - don't just show them, explain what they mean

### 6.3 Discussion Preparation
Each team member must be able to explain:
- The mathematical foundations of all three methods (RF, DT, SVM)
- Why specific hyperparameters were chosen
- How to interpret XAI outputs (feature importance, PDPs, LIME, SHAP)
- The model selection process (why best 2 were chosen)
- Decision Tree visualization and decision rules
- Code functionality and logic
- Results and conclusions

---

## 7. Timeline Suggestion

| Task | Status |
|------|--------|
| Data loading & EDA | ✅ Complete |
| Binary classification conversion | ✅ Complete |
| Mathematical overview writing | ✅ Complete |
| Random Forest implementation | ✅ Complete |
| Decision Tree implementation | ✅ Complete |
| SVM implementation | ✅ Complete |
| Train all 3 models with CV | ✅ Complete |
| Select best 2 models (by AUC-ROC) | ✅ Complete |
| Model comparison on test set | ✅ Complete |
| ROC curve analysis (all 3 models) | ✅ Complete |
| Decision Tree visualization | ✅ Complete |
| XAI analysis | ✅ Complete |
| Report writing & formatting | ✅ Complete |
| Code review & testing | To Do |
| Final submission | Deadline: Jan 3, 2026 |

---

## 8. References (to include in report)

```bibtex
@misc{UCI_maternal,
  author = {Ahmed, Marzia and Kashem, Mohammod Abul},
  title = {Maternal Health Risk Data Set},
  year = {2023},
  publisher = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/dataset/863/maternal+health+risk}
}

@book{james2021introduction,
  title={An Introduction to Statistical Learning},
  author={James, Gareth and Witten, Daniela and Hastie, Trevor and Tibshirani, Robert},
  year={2021},
  publisher={Springer}
}

@article{breiman2001random,
  title={Random forests},
  author={Breiman, Leo},
  journal={Machine learning},
  volume={45},
  pages={5--32},
  year={2001}
}

@book{molnar2020interpretable,
  title={Interpretable Machine Learning},
  author={Molnar, Christoph},
  year={2020},
  url={https://christophm.github.io/interpretable-ml-book/}
}
```

---

**Good luck with your project!**

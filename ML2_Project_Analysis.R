# ============================================================
# Machine Learning 2: Maternal Health Risk Classification
# Authors: Aisha, Mufaddal, Raju Ahmed
# Date: December 2025
# ============================================================

# Clear workspace
rm(list = ls())

# Set seed for reproducibility
set.seed(123)

# ============================================================
# 1. LOAD PACKAGES
# ============================================================

cat("Loading packages...\n")

# Install missing packages if needed
required_packages <- c("tidyverse", "caret", "randomForest", "e1071",
                       "corrplot", "GGally", "gridExtra", "iml", "pdp", "lime")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(tidyverse)
  library(caret)
  library(randomForest)
  library(e1071)
  library(corrplot)
  library(GGally)
  library(gridExtra)
  library(iml)
  library(pdp)
  library(lime)
})

cat("Packages loaded successfully!\n\n")

# ============================================================
# 2. LOAD AND INSPECT DATA
# ============================================================

cat("=== DATA LOADING ===\n")

data <- read.csv("Maternal Health Risk Data Set.csv")

cat("Dataset dimensions:", nrow(data), "observations,", ncol(data), "variables\n")
cat("\nStructure:\n")
str(data)

cat("\nSummary statistics:\n")
summary(data)

# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

cat("\n=== DATA PREPROCESSING ===\n")

# 3.1 Check for missing values
cat("\nMissing values per column:\n")
print(colSums(is.na(data)))

# 3.2 Outlier detection and removal
cat("\nOutlier detection:\n")
cat("HeartRate range:", min(data$HeartRate), "-", max(data$HeartRate), "\n")
cat("Observations with HeartRate < 30 (physiologically impossible):",
    sum(data$HeartRate < 30), "\n")

# Remove outliers
data_clean <- data[data$HeartRate >= 30, ]
cat("Observations after outlier removal:", nrow(data_clean), "\n")

# 3.3 Convert target to factor (use valid R names without spaces)
data_clean$RiskLevel <- factor(data_clean$RiskLevel,
                                levels = c("low risk", "mid risk", "high risk"),
                                labels = c("LowRisk", "MidRisk", "HighRisk"))

# ============================================================
# 4. EXPLORATORY DATA ANALYSIS
# ============================================================

cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# 4.1 Target distribution
cat("\nTarget variable distribution:\n")
print(table(data_clean$RiskLevel))
cat("\nProportions:\n")
print(round(prop.table(table(data_clean$RiskLevel)) * 100, 1))

# 4.2 Descriptive statistics by risk level
cat("\nDescriptive statistics by risk level:\n")
desc_stats <- data_clean %>%
  group_by(RiskLevel) %>%
  summarise(
    n = n(),
    Age_mean = round(mean(Age), 1),
    SystolicBP_mean = round(mean(SystolicBP), 1),
    DiastolicBP_mean = round(mean(DiastolicBP), 1),
    BS_mean = round(mean(BS), 2),
    BodyTemp_mean = round(mean(BodyTemp), 2),
    HeartRate_mean = round(mean(HeartRate), 1)
  )
print(desc_stats)

# 4.3 Correlation matrix
cat("\nCorrelation matrix:\n")
cor_matrix <- cor(data_clean[, 1:6])
print(round(cor_matrix, 3))

# 4.4 Visualizations
# Target distribution plot
p1 <- ggplot(data_clean, aes(x = RiskLevel, fill = RiskLevel)) +
  geom_bar(alpha = 0.8) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("LowRisk" = "#2ecc71",
                                "MidRisk" = "#f39c12",
                                "HighRisk" = "#e74c3c")) +
  labs(title = "Distribution of Maternal Health Risk Levels",
       x = "Risk Level", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

# Box plots
data_long <- data_clean %>%
  pivot_longer(cols = c(Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate),
               names_to = "Variable", values_to = "Value")

p2 <- ggplot(data_long, aes(x = RiskLevel, y = Value, fill = RiskLevel)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~Variable, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = c("LowRisk" = "#2ecc71",
                                "MidRisk" = "#f39c12",
                                "HighRisk" = "#e74c3c")) +
  labs(title = "Predictors by Risk Level") +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))

# Save plots
ggsave("plot_target_distribution.png", p1, width = 8, height = 5)
ggsave("plot_boxplots.png", p2, width = 10, height = 8)

# Correlation heatmap
png("plot_correlation.png", width = 800, height = 600)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.8,
         tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix", mar = c(0, 0, 2, 0))
dev.off()

cat("\nEDA plots saved!\n")

# ============================================================
# 5. DATA SPLITTING
# ============================================================

cat("\n=== DATA SPLITTING ===\n")

set.seed(123)
train_index <- createDataPartition(data_clean$RiskLevel, p = 0.8, list = FALSE)
train_data <- data_clean[train_index, ]
test_data <- data_clean[-train_index, ]

cat("Training set:", nrow(train_data), "observations\n")
cat("Test set:", nrow(test_data), "observations\n")

# Verify stratification
cat("\nClass proportions in training:\n")
print(round(prop.table(table(train_data$RiskLevel)) * 100, 1))
cat("\nClass proportions in test:\n")
print(round(prop.table(table(test_data$RiskLevel)) * 100, 1))

# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

cat("\n=== RANDOM FOREST MODEL ===\n")

# 6.1 Hyperparameter tuning with CV
cat("\nTuning mtry parameter with 10-fold CV...\n")

set.seed(123)
cv_control <- trainControl(method = "cv", number = 10, classProbs = TRUE)

rf_grid <- expand.grid(mtry = c(2, 3, 4, 5))

rf_model <- train(
  RiskLevel ~ .,
  data = train_data,
  method = "rf",
  trControl = cv_control,
  tuneGrid = rf_grid,
  ntree = 500,
  importance = TRUE
)

cat("\nRandom Forest CV Results:\n")
print(rf_model$results)
cat("\nBest mtry:", rf_model$bestTune$mtry, "\n")

# 6.2 Final model
rf_final <- rf_model$finalModel
cat("\nFinal Random Forest Model:\n")
print(rf_final)

# 6.3 Feature importance
cat("\nFeature Importance (Mean Decrease Gini):\n")
rf_importance <- importance(rf_final)
rf_imp_df <- data.frame(
  Feature = rownames(rf_importance),
  MeanDecreaseGini = rf_importance[, "MeanDecreaseGini"],
  MeanDecreaseAccuracy = rf_importance[, "MeanDecreaseAccuracy"]
)
rf_imp_df <- rf_imp_df[order(-rf_imp_df$MeanDecreaseGini), ]
print(rf_imp_df)

# Feature importance plot
p_rf_imp <- ggplot(rf_imp_df, aes(x = reorder(Feature, MeanDecreaseGini),
                                   y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "#3498db", alpha = 0.8) +
  coord_flip() +
  labs(title = "Random Forest: Feature Importance",
       x = "", y = "Mean Decrease in Gini") +
  theme_minimal()

ggsave("plot_rf_importance.png", p_rf_imp, width = 8, height = 5)

# ============================================================
# 7. SUPPORT VECTOR MACHINE MODEL
# ============================================================

cat("\n=== SUPPORT VECTOR MACHINE MODEL ===\n")

# 7.1 Feature scaling
cat("\nScaling features...\n")
preProcValues <- preProcess(train_data[, 1:6], method = c("center", "scale"))
train_scaled <- predict(preProcValues, train_data)
test_scaled <- predict(preProcValues, test_data)

cat("Scaling parameters (means):\n")
print(round(preProcValues$mean, 2))

# 7.2 Kernel comparison
cat("\nComparing SVM kernels with 10-fold CV...\n")

kernels <- c("linear", "radial", "polynomial")
kernel_results <- data.frame(Kernel = kernels, Accuracy = NA)

for (i in seq_along(kernels)) {
  set.seed(123)
  if (kernels[i] == "polynomial") {
    tune_result <- tune(svm, RiskLevel ~ ., data = train_scaled,
                        kernel = kernels[i],
                        ranges = list(cost = c(1, 10), degree = c(2, 3)),
                        tunecontrol = tune.control(cross = 10))
  } else {
    tune_result <- tune(svm, RiskLevel ~ ., data = train_scaled,
                        kernel = kernels[i],
                        ranges = list(cost = c(0.1, 1, 10)),
                        tunecontrol = tune.control(cross = 10))
  }
  kernel_results$Accuracy[i] <- round((1 - tune_result$best.performance) * 100, 2)
}

cat("\nKernel Comparison Results:\n")
print(kernel_results)

# 7.3 Hyperparameter tuning for RBF kernel
cat("\nTuning RBF kernel hyperparameters...\n")

set.seed(123)
svm_tune <- tune(svm, RiskLevel ~ ., data = train_scaled,
                 kernel = "radial",
                 ranges = list(cost = c(0.1, 1, 10, 100),
                               gamma = c(0.01, 0.1, 0.5, 1)),
                 tunecontrol = tune.control(cross = 10))

cat("\nBest SVM parameters:\n")
print(svm_tune$best.parameters)
cat("Best CV error:", round(svm_tune$best.performance * 100, 2), "%\n")

svm_final <- svm_tune$best.model

# ============================================================
# 8. MODEL COMPARISON ON TEST DATA
# ============================================================

cat("\n=== MODEL COMPARISON ON TEST DATA ===\n")

# 8.1 Predictions
rf_pred <- predict(rf_model, newdata = test_data)
svm_pred <- predict(svm_final, newdata = test_scaled)

# 8.2 Confusion matrices
cat("\n--- Random Forest Confusion Matrix ---\n")
rf_cm <- confusionMatrix(rf_pred, test_data$RiskLevel)
print(rf_cm$table)

cat("\n--- SVM Confusion Matrix ---\n")
svm_cm <- confusionMatrix(svm_pred, test_scaled$RiskLevel)
print(svm_cm$table)

# 8.3 Performance metrics comparison
cat("\n=== PERFORMANCE METRICS COMPARISON ===\n")

metrics_df <- data.frame(
  Metric = c("Overall Accuracy", "Kappa",
             "High Risk Recall", "High Risk Precision", "High Risk F1",
             "Mid Risk Recall", "Mid Risk Precision", "Mid Risk F1",
             "Low Risk Recall", "Low Risk Precision", "Low Risk F1"),
  RandomForest = c(
    round(rf_cm$overall["Accuracy"] * 100, 2),
    round(rf_cm$overall["Kappa"], 3),
    round(rf_cm$byClass["Class: HighRisk", "Recall"] * 100, 2),
    round(rf_cm$byClass["Class: HighRisk", "Precision"] * 100, 2),
    round(rf_cm$byClass["Class: HighRisk", "F1"] * 100, 2),
    round(rf_cm$byClass["Class: MidRisk", "Recall"] * 100, 2),
    round(rf_cm$byClass["Class: MidRisk", "Precision"] * 100, 2),
    round(rf_cm$byClass["Class: MidRisk", "F1"] * 100, 2),
    round(rf_cm$byClass["Class: LowRisk", "Recall"] * 100, 2),
    round(rf_cm$byClass["Class: LowRisk", "Precision"] * 100, 2),
    round(rf_cm$byClass["Class: LowRisk", "F1"] * 100, 2)
  ),
  SVM = c(
    round(svm_cm$overall["Accuracy"] * 100, 2),
    round(svm_cm$overall["Kappa"], 3),
    round(svm_cm$byClass["Class: HighRisk", "Recall"] * 100, 2),
    round(svm_cm$byClass["Class: HighRisk", "Precision"] * 100, 2),
    round(svm_cm$byClass["Class: HighRisk", "F1"] * 100, 2),
    round(svm_cm$byClass["Class: MidRisk", "Recall"] * 100, 2),
    round(svm_cm$byClass["Class: MidRisk", "Precision"] * 100, 2),
    round(svm_cm$byClass["Class: MidRisk", "F1"] * 100, 2),
    round(svm_cm$byClass["Class: LowRisk", "Recall"] * 100, 2),
    round(svm_cm$byClass["Class: LowRisk", "Precision"] * 100, 2),
    round(svm_cm$byClass["Class: LowRisk", "F1"] * 100, 2)
  )
)

print(metrics_df)

# Save metrics to CSV
write.csv(metrics_df, "model_comparison_metrics.csv", row.names = FALSE)

# Comparison bar plot
metrics_plot <- data.frame(
  Metric = rep(c("Accuracy", "High Risk\nRecall", "High Risk\nF1"), 2),
  Model = rep(c("Random Forest", "SVM"), each = 3),
  Value = c(
    rf_cm$overall["Accuracy"] * 100,
    rf_cm$byClass["Class: HighRisk", "Recall"] * 100,
    rf_cm$byClass["Class: HighRisk", "F1"] * 100,
    svm_cm$overall["Accuracy"] * 100,
    svm_cm$byClass["Class: HighRisk", "Recall"] * 100,
    svm_cm$byClass["Class: HighRisk", "F1"] * 100
  )
)

p_comparison <- ggplot(metrics_plot, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  geom_text(aes(label = round(Value, 1)),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("Random Forest" = "#3498db", "SVM" = "#9b59b6")) +
  labs(title = "Model Performance Comparison", y = "Percentage (%)", x = "") +
  theme_minimal() +
  theme(legend.position = "bottom") +
  ylim(0, 100)

ggsave("plot_model_comparison.png", p_comparison, width = 8, height = 5)

# ============================================================
# 9. INTERPRETABLE ML / XAI
# ============================================================

cat("\n=== INTERPRETABLE ML (XAI) ===\n")

# 9.1 Partial Dependence Plots
cat("\nGenerating Partial Dependence Plots...\n")

# Create predictor object for iml
predictor_rf <- Predictor$new(rf_model, data = train_data[, 1:6],
                               y = train_data$RiskLevel, type = "prob")

# PDP for Blood Sugar
pdp_bs <- FeatureEffect$new(predictor_rf, feature = "BS", method = "pdp")
p_pdp_bs <- plot(pdp_bs) +
  labs(title = "Partial Dependence: Blood Sugar") +
  theme_minimal()
ggsave("plot_pdp_bs.png", p_pdp_bs, width = 8, height = 5)

# PDP for Systolic BP
pdp_sbp <- FeatureEffect$new(predictor_rf, feature = "SystolicBP", method = "pdp")
p_pdp_sbp <- plot(pdp_sbp) +
  labs(title = "Partial Dependence: Systolic BP") +
  theme_minimal()
ggsave("plot_pdp_sbp.png", p_pdp_sbp, width = 8, height = 5)

# PDP for Age
pdp_age <- FeatureEffect$new(predictor_rf, feature = "Age", method = "pdp")
p_pdp_age <- plot(pdp_age) +
  labs(title = "Partial Dependence: Age") +
  theme_minimal()
ggsave("plot_pdp_age.png", p_pdp_age, width = 8, height = 5)

# 9.2 Permutation Feature Importance for SVM
cat("\nCalculating SVM Permutation Importance...\n")

predictor_svm <- Predictor$new(svm_final, data = train_scaled[, 1:6],
                                y = train_scaled$RiskLevel, type = "prob")

set.seed(123)
svm_importance <- FeatureImp$new(predictor_svm, loss = "ce", n.repetitions = 10)

p_svm_imp <- plot(svm_importance) +
  labs(title = "SVM: Permutation Feature Importance") +
  theme_minimal()
ggsave("plot_svm_importance.png", p_svm_imp, width = 8, height = 5)

# 9.3 LIME Explanations
cat("\nGenerating LIME explanations...\n")

# Create LIME explainer
lime_explainer <- lime(train_data[, 1:6], rf_model)

# Find interesting test cases (using corrected factor levels)
high_risk_idx <- which(test_data$RiskLevel == "HighRisk" & rf_pred == "HighRisk")[1:2]
low_risk_idx <- which(test_data$RiskLevel == "LowRisk" & rf_pred == "LowRisk")[1]

selected_cases <- na.omit(c(high_risk_idx, low_risk_idx))
cat("Selected cases for LIME:", selected_cases, "\n")

if (length(selected_cases) > 0) {
  set.seed(123)
  tryCatch({
    lime_explanation <- explain(test_data[selected_cases, 1:6],
                                 lime_explainer,
                                 n_labels = 1,
                                 n_features = 6)

    p_lime <- plot_features(lime_explanation) +
      labs(title = "LIME: Local Explanations") +
      theme_minimal()
    ggsave("plot_lime.png", p_lime, width = 10, height = 8)
    cat("LIME plot saved!\n")
  }, error = function(e) {
    cat("LIME error:", conditionMessage(e), "\n")
  })
}

# 9.4 SHAP Values
cat("\nCalculating SHAP values for a high-risk case...\n")

# Get a valid high-risk index
high_risk_test_idx <- which(test_data$RiskLevel == "HighRisk")[1]
cat("Using test case index:", high_risk_test_idx, "\n")

if (!is.na(high_risk_test_idx) && length(high_risk_test_idx) > 0) {
  set.seed(123)
  tryCatch({
    shapley <- Shapley$new(predictor_rf, x.interest = test_data[high_risk_test_idx, 1:6])

    p_shap <- plot(shapley) +
      labs(title = paste("SHAP Values for Test Case", high_risk_test_idx)) +
      theme_minimal()
    ggsave("plot_shap.png", p_shap, width = 8, height = 5)
    cat("SHAP plot saved!\n")
  }, error = function(e) {
    cat("SHAP error:", conditionMessage(e), "\n")
  })
}

cat("\nXAI plots saved!\n")

# ============================================================
# 10. SUMMARY
# ============================================================

cat("\n")
cat("============================================================\n")
cat("                    ANALYSIS SUMMARY\n")
cat("============================================================\n")
cat("\n")
cat("Dataset: Maternal Health Risk (UCI)\n")
cat("Observations:", nrow(data_clean), "(after removing 2 outliers)\n")
cat("Features: 6 predictors, 1 target (3-class)\n")
cat("\n")
cat("BEST MODEL: Random Forest\n")
cat("  - Overall Accuracy:", round(rf_cm$overall["Accuracy"] * 100, 2), "%\n")
cat("  - High Risk Recall:", round(rf_cm$byClass["Class: HighRisk", "Recall"] * 100, 2), "%\n")
cat("  - High Risk F1:", round(rf_cm$byClass["Class: HighRisk", "F1"] * 100, 2), "%\n")
cat("\n")
cat("TOP PREDICTORS (by importance):\n")
cat("  1. Blood Sugar (BS)\n")
cat("  2. Systolic Blood Pressure\n")
cat("  3. Age\n")
cat("\n")
cat("Files generated:\n")
cat("  - model_comparison_metrics.csv\n")
cat("  - plot_target_distribution.png\n")
cat("  - plot_boxplots.png\n")
cat("  - plot_correlation.png\n")
cat("  - plot_rf_importance.png\n")
cat("  - plot_svm_importance.png\n")
cat("  - plot_model_comparison.png\n")
cat("  - plot_pdp_bs.png\n")
cat("  - plot_pdp_sbp.png\n")
cat("  - plot_pdp_age.png\n")
cat("  - plot_lime.png\n")
cat("  - plot_shap.png\n")
cat("\n")
cat("============================================================\n")
cat("                    ANALYSIS COMPLETE\n")
cat("============================================================\n")

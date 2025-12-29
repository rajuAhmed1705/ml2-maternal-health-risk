# ============================================================
# Generate Model Comparison Figures for MODEL_COMPARISON.md
# ============================================================

# Load required libraries
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(pROC)
library(gridExtra)
library(iml)

# Set seed for reproducibility
set.seed(123)

# Create figures directory
if (!dir.exists("figures")) {
  dir.create("figures")
}

# ============================================================
# 1. Load and Prepare Data
# ============================================================
data <- read.csv("Maternal Health Risk Data Set.csv")

# Remove outliers (HeartRate = 7)
data_clean <- data[data$HeartRate >= 30, ]

# Convert to binary classification
data_clean$RiskLevel <- ifelse(data_clean$RiskLevel == "high risk", "HighRisk", "NotHighRisk")
data_clean$RiskLevel <- factor(data_clean$RiskLevel, levels = c("NotHighRisk", "HighRisk"))

# Split data
set.seed(123)
train_index <- createDataPartition(data_clean$RiskLevel, p = 0.8, list = FALSE)
train_data <- data_clean[train_index, ]
test_data <- data_clean[-train_index, ]

# CV control
cv_control <- trainControl(method = "cv", number = 10, classProbs = TRUE,
                           summaryFunction = twoClassSummary, savePredictions = "final")

# ============================================================
# 2. Train All 3 Models
# ============================================================
cat("Training Random Forest...\n")
set.seed(123)
rf_model <- train(RiskLevel ~ ., data = train_data, method = "rf", trControl = cv_control,
                  tuneGrid = expand.grid(mtry = c(2, 3, 4, 5)), ntree = 500,
                  importance = TRUE, metric = "ROC")

cat("Training Decision Tree...\n")
set.seed(123)
dt_model <- train(RiskLevel ~ ., data = train_data, method = "rpart", trControl = cv_control,
                  tuneGrid = expand.grid(cp = c(0.001, 0.01, 0.05, 0.1)), metric = "ROC")

cat("Training SVM...\n")
preProcValues <- preProcess(train_data[, 1:6], method = c("center", "scale"))
train_scaled <- predict(preProcValues, train_data)
test_scaled <- predict(preProcValues, test_data)

set.seed(123)
svm_model <- train(RiskLevel ~ ., data = train_scaled, method = "svmRadial", trControl = cv_control,
                   tuneGrid = expand.grid(C = c(0.1, 1, 10, 100), sigma = c(0.01, 0.1, 0.5, 1)),
                   metric = "ROC")

# ============================================================
# 3. Get CV Results
# ============================================================
cv_auc_rf <- max(rf_model$results$ROC)
cv_auc_dt <- max(dt_model$results$ROC)
cv_auc_svm <- max(svm_model$results$ROC)

cat("\n=== CV AUC-ROC Results ===\n")
cat("Random Forest:", round(cv_auc_rf, 4), "\n")
cat("Decision Tree:", round(cv_auc_dt, 4), "\n")
cat("SVM:", round(cv_auc_svm, 4), "\n")

# ============================================================
# FIGURE 1: Model Selection - CV AUC Comparison
# ============================================================
cat("\nGenerating Figure 1: CV AUC Comparison...\n")

cv_results_df <- data.frame(
  Model = c("Random Forest", "Decision Tree", "SVM"),
  AUC = c(cv_auc_rf, cv_auc_dt, cv_auc_svm),
  Selected = c("Selected", "Eliminated", "Selected")
)
cv_results_df$Model <- factor(cv_results_df$Model, levels = cv_results_df$Model[order(-cv_results_df$AUC)])

png("figures/01_cv_auc_comparison.png", width = 800, height = 500, res = 120)
ggplot(cv_results_df, aes(x = reorder(Model, AUC), y = AUC, fill = Selected)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.9) +
  geom_text(aes(label = sprintf("%.3f", AUC)), hjust = -0.1, size = 5, fontface = "bold") +
  geom_hline(yintercept = 0.85, linetype = "dashed", color = "red", size = 1) +
  annotate("text", x = 0.7, y = 0.86, label = "Selection Threshold", color = "red", size = 3.5) +
  scale_fill_manual(values = c("Selected" = "#2ecc71", "Eliminated" = "#e74c3c")) +
  coord_flip() +
  labs(title = "Model Selection: Cross-Validation AUC-ROC",
       subtitle = "10-Fold CV | Best 2 models selected for final comparison",
       x = "", y = "AUC-ROC") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold")) +
  ylim(0, 1.05)
dev.off()

# ============================================================
# FIGURE 2: Hyperparameter Tuning Plots
# ============================================================
cat("Generating Figure 2: Hyperparameter Tuning...\n")

png("figures/02_hyperparameter_tuning.png", width = 1000, height = 400, res = 120)
p1 <- ggplot(rf_model$results, aes(x = mtry, y = ROC)) +
  geom_line(color = "#3498db", size = 1.5) +
  geom_point(color = "#3498db", size = 4) +
  geom_point(data = rf_model$results[rf_model$results$mtry == rf_model$bestTune$mtry, ],
             aes(x = mtry, y = ROC), color = "#e74c3c", size = 6, shape = 18) +
  labs(title = "Random Forest: mtry Tuning",
       subtitle = paste("Best mtry =", rf_model$bestTune$mtry),
       x = "mtry (Features per Split)", y = "AUC-ROC") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

p2 <- ggplot(svm_model$results, aes(x = sigma, y = ROC, color = factor(C), group = factor(C))) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_color_brewer(palette = "Set1", name = "Cost (C)") +
  labs(title = "SVM: Cost vs Sigma Tuning",
       subtitle = paste("Best C =", svm_model$bestTune$C, ", sigma =", round(svm_model$bestTune$sigma, 3)),
       x = "Sigma (γ)", y = "AUC-ROC") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "right")

grid.arrange(p1, p2, ncol = 2)
dev.off()

# ============================================================
# FIGURE 3: Test Set Performance Comparison
# ============================================================
cat("Generating Figure 3: Test Performance Comparison...\n")

# Get predictions
rf_pred <- predict(rf_model, newdata = test_data)
dt_pred <- predict(dt_model, newdata = test_data)
svm_pred <- predict(svm_model, newdata = test_scaled)

# Confusion matrices
rf_cm <- confusionMatrix(rf_pred, test_data$RiskLevel, positive = "HighRisk")
dt_cm <- confusionMatrix(dt_pred, test_data$RiskLevel, positive = "HighRisk")
svm_cm <- confusionMatrix(svm_pred, test_scaled$RiskLevel, positive = "HighRisk")

# Create metrics dataframe
metrics_df <- data.frame(
  Metric = rep(c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"), 3),
  Model = rep(c("Random Forest", "Decision Tree", "SVM"), each = 5),
  Value = c(
    rf_cm$overall["Accuracy"] * 100, rf_cm$byClass["Sensitivity"] * 100,
    rf_cm$byClass["Specificity"] * 100, rf_cm$byClass["Pos Pred Value"] * 100,
    rf_cm$byClass["F1"] * 100,
    dt_cm$overall["Accuracy"] * 100, dt_cm$byClass["Sensitivity"] * 100,
    dt_cm$byClass["Specificity"] * 100, dt_cm$byClass["Pos Pred Value"] * 100,
    dt_cm$byClass["F1"] * 100,
    svm_cm$overall["Accuracy"] * 100, svm_cm$byClass["Sensitivity"] * 100,
    svm_cm$byClass["Specificity"] * 100, svm_cm$byClass["Pos Pred Value"] * 100,
    svm_cm$byClass["F1"] * 100
  )
)

metrics_df$Model <- factor(metrics_df$Model, levels = c("Random Forest", "SVM", "Decision Tree"))
metrics_df$Metric <- factor(metrics_df$Metric, levels = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1 Score"))

png("figures/03_test_performance_comparison.png", width = 900, height = 500, res = 120)
ggplot(metrics_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), alpha = 0.9, width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", Value)), position = position_dodge(width = 0.8),
            vjust = -0.3, size = 3) +
  scale_fill_manual(values = c("Random Forest" = "#3498db", "SVM" = "#9b59b6", "Decision Tree" = "#e67e22")) +
  labs(title = "Test Set Performance Comparison (All 3 Models)",
       subtitle = "Random Forest and SVM outperform Decision Tree across all metrics",
       x = "", y = "Percentage (%)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 0)) +
  ylim(0, 105)
dev.off()

# ============================================================
# FIGURE 4: ROC Curves (All 3 Models)
# ============================================================
cat("Generating Figure 4: ROC Curves...\n")

rf_probs <- predict(rf_model, newdata = test_data, type = "prob")
dt_probs <- predict(dt_model, newdata = test_data, type = "prob")
svm_probs <- predict(svm_model, newdata = test_scaled, type = "prob")

roc_rf <- roc(test_data$RiskLevel, rf_probs$HighRisk, levels = c("NotHighRisk", "HighRisk"))
roc_dt <- roc(test_data$RiskLevel, dt_probs$HighRisk, levels = c("NotHighRisk", "HighRisk"))
roc_svm <- roc(test_scaled$RiskLevel, svm_probs$HighRisk, levels = c("NotHighRisk", "HighRisk"))

png("figures/04_roc_curves.png", width = 700, height = 600, res = 120)
plot(roc_rf, col = "#3498db", lwd = 3, main = "ROC Curves: All 3 Models")
plot(roc_svm, col = "#9b59b6", lwd = 3, add = TRUE)
plot(roc_dt, col = "#e67e22", lwd = 3, lty = 2, add = TRUE)
abline(a = 0, b = 1, lty = 3, col = "gray50", lwd = 2)
legend("bottomright",
       legend = c(paste0("Random Forest (AUC = ", round(auc(roc_rf), 3), ") ✓"),
                  paste0("SVM (AUC = ", round(auc(roc_svm), 3), ") ✓"),
                  paste0("Decision Tree (AUC = ", round(auc(roc_dt), 3), ") ✗"),
                  "Random Baseline"),
       col = c("#3498db", "#9b59b6", "#e67e22", "gray50"),
       lwd = c(3, 3, 3, 2), lty = c(1, 1, 2, 3), cex = 0.9)
dev.off()

# ============================================================
# FIGURE 5: Confusion Matrices
# ============================================================
cat("Generating Figure 5: Confusion Matrices...\n")

# Function to create confusion matrix plot
plot_cm <- function(cm, model_name, color) {
  cm_table <- as.data.frame(cm$table)
  colnames(cm_table) <- c("Prediction", "Reference", "Freq")

  ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = Freq), size = 8, fontface = "bold") +
    scale_fill_gradient(low = "white", high = color) +
    labs(title = model_name,
         subtitle = paste0("Acc: ", round(cm$overall["Accuracy"]*100, 1), "% | ",
                          "Sens: ", round(cm$byClass["Sensitivity"]*100, 1), "% | ",
                          "Spec: ", round(cm$byClass["Specificity"]*100, 1), "%"),
         x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 11) +
    theme(legend.position = "none", plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, size = 9))
}

png("figures/05_confusion_matrices.png", width = 1100, height = 400, res = 120)
p1 <- plot_cm(rf_cm, "Random Forest ✓", "#3498db")
p2 <- plot_cm(svm_cm, "SVM ✓", "#9b59b6")
p3 <- plot_cm(dt_cm, "Decision Tree ✗", "#e67e22")
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

# ============================================================
# FIGURE 6: Feature Importance (All 3 Models)
# ============================================================
cat("Generating Figure 6: Feature Importance...\n")

# RF importance
rf_imp <- importance(rf_model$finalModel)
rf_imp_df <- data.frame(Feature = rownames(rf_imp),
                         Importance = rf_imp[, "MeanDecreaseGini"],
                         Model = "Random Forest")

# DT importance
dt_imp <- varImp(dt_model)$importance
dt_imp_df <- data.frame(Feature = rownames(dt_imp),
                         Importance = dt_imp$Overall,
                         Model = "Decision Tree")

# SVM importance (permutation)
predictor_svm <- Predictor$new(model = svm_model, data = train_scaled[, 1:6],
                                y = train_scaled$RiskLevel, type = "prob")
set.seed(123)
svm_imp <- FeatureImp$new(predictor_svm, loss = "ce", n.repetitions = 5)
svm_imp_df <- data.frame(Feature = svm_imp$results$feature,
                          Importance = svm_imp$results$importance,
                          Model = "SVM")

# Normalize to 0-100 scale
rf_imp_df$Importance <- rf_imp_df$Importance / max(rf_imp_df$Importance) * 100
dt_imp_df$Importance <- dt_imp_df$Importance / max(dt_imp_df$Importance) * 100
svm_imp_df$Importance <- svm_imp_df$Importance / max(svm_imp_df$Importance) * 100

all_imp <- rbind(rf_imp_df, dt_imp_df, svm_imp_df)
all_imp$Model <- factor(all_imp$Model, levels = c("Random Forest", "SVM", "Decision Tree"))

png("figures/06_feature_importance.png", width = 1000, height = 500, res = 120)
ggplot(all_imp, aes(x = reorder(Feature, Importance), y = Importance, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), alpha = 0.9, width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("Random Forest" = "#3498db", "SVM" = "#9b59b6", "Decision Tree" = "#e67e22")) +
  labs(title = "Feature Importance Comparison (All 3 Models)",
       subtitle = "Blood Sugar (BS) is the most important predictor across all models",
       x = "", y = "Relative Importance (%)") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))
dev.off()

# ============================================================
# FIGURE 7: Decision Tree Visualization
# ============================================================
cat("Generating Figure 7: Decision Tree Visualization...\n")

png("figures/07_decision_tree.png", width = 900, height = 600, res = 120)
rpart.plot(dt_model$finalModel,
           main = "Decision Tree Visualization\n(Used for model selection, not final comparison)",
           extra = 104, # Show % of obs + predicted class
           box.palette = c("#2ecc71", "#e74c3c"),
           shadow.col = "gray",
           nn = TRUE,
           cex = 0.9)
dev.off()

# ============================================================
# FIGURE 8: Model Selection Summary
# ============================================================
cat("Generating Figure 8: Model Selection Summary...\n")

summary_df <- data.frame(
  Model = c("Random Forest", "SVM", "Decision Tree"),
  CV_AUC = c(cv_auc_rf, cv_auc_svm, cv_auc_dt),
  Test_Acc = c(rf_cm$overall["Accuracy"], svm_cm$overall["Accuracy"], dt_cm$overall["Accuracy"]),
  Sensitivity = c(rf_cm$byClass["Sensitivity"], svm_cm$byClass["Sensitivity"], dt_cm$byClass["Sensitivity"]),
  Selected = c("Yes", "Yes", "No")
)

summary_long <- summary_df %>%
  pivot_longer(cols = c(CV_AUC, Test_Acc, Sensitivity), names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels = c("CV_AUC", "Test_Acc", "Sensitivity"),
                         labels = c("CV AUC-ROC", "Test Accuracy", "Sensitivity")))

png("figures/08_model_selection_summary.png", width = 900, height = 500, res = 120)
ggplot(summary_long, aes(x = Model, y = Value, fill = Selected)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.9) +
  geom_text(aes(label = sprintf("%.2f", Value)), vjust = -0.3, size = 3.5) +
  facet_wrap(~Metric, scales = "free_y") +
  scale_fill_manual(values = c("Yes" = "#2ecc71", "No" = "#e74c3c"),
                    labels = c("Yes" = "Selected ✓", "No" = "Eliminated ✗")) +
  labs(title = "Model Selection Summary: Why RF and SVM Were Chosen",
       subtitle = "Decision Tree consistently underperforms across all key metrics",
       x = "", y = "", fill = "Status") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold", size = 11)) +
  ylim(0, 1.1)
dev.off()

# ============================================================
# Print Summary
# ============================================================
cat("\n========================================\n")
cat("All figures generated successfully!\n")
cat("========================================\n")
cat("\nFigures saved to ./figures/ directory:\n")
cat("  1. 01_cv_auc_comparison.png\n")
cat("  2. 02_hyperparameter_tuning.png\n")
cat("  3. 03_test_performance_comparison.png\n")
cat("  4. 04_roc_curves.png\n")
cat("  5. 05_confusion_matrices.png\n")
cat("  6. 06_feature_importance.png\n")
cat("  7. 07_decision_tree.png\n")
cat("  8. 08_model_selection_summary.png\n")

cat("\n=== Final Model Selection ===\n")
cat("Selected: Random Forest (AUC:", round(cv_auc_rf, 3), ") + SVM (AUC:", round(cv_auc_svm, 3), ")\n")
cat("Eliminated: Decision Tree (AUC:", round(cv_auc_dt, 3), ")\n")

cat("\n=== Test Set Results ===\n")
cat("Random Forest - Acc:", round(rf_cm$overall["Accuracy"]*100, 1), "%, Sens:", round(rf_cm$byClass["Sensitivity"]*100, 1), "%\n")
cat("SVM - Acc:", round(svm_cm$overall["Accuracy"]*100, 1), "%, Sens:", round(svm_cm$byClass["Sensitivity"]*100, 1), "%\n")
cat("Decision Tree - Acc:", round(dt_cm$overall["Accuracy"]*100, 1), "%, Sens:", round(dt_cm$byClass["Sensitivity"]*100, 1), "%\n")

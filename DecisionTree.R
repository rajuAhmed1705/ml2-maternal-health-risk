# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(rpart)
library(rpart.plot)
library(ROCR)
library(randomForest)

#set seed
set.seed(123)

## Read the dataset
df <- read.csv('./Maternal Health Risk Data Set.csv')
head(df)
summary(df)

# Check for missing values
colSums(is.na(df))

# Map risk levels to binary values
risk_mapping <- c('low risk' = 0,'high risk' = 1)
df$RiskLevel <- factor(risk_mapping[df$RiskLevel], levels = c(0, 1))
#df$RiskLevel <- factor(risk_mapping[df$RiskLevel], levels = c(0, 1), labels = c('low risk', 'high risk'))
#df$RiskLevel <- as.numeric(risk_mapping[df$RiskLevel])
df$RiskLevel <- ordered(df$RiskLevel,levels=c("1","0"))# Convert to ordered factor
summary(df)


# Create the bar plot with y-axis limit set to 700
counts <- table(df$RiskLevel)
bp <- barplot(counts, 
              main = "Distribution of Risk Levels", 
              xlab = "Risk Level", 
              ylab = "Count", 
              ylim = c(0, 700)) # Set y-axis limit to 700

# Add the count labels above the bars
text(bp, counts + 10, labels = counts, pos = 3)


# Split data into train and test sets
trainIndex <- createDataPartition(df$RiskLevel, p = 0.8, list = FALSE)
trainData <- df[trainIndex, ]
testData <- df[-trainIndex, ]

# Define k-fold cross-validation (5-fold)
control <- trainControl(
  method = "repeatedcv", 
  number = 5,             # Use 10 partitions
  repeats = 1  # Repeat 2 times
)

tune_grid = expand.grid(cp=c(0.001)) ## Setting cp=0.001

set.seed(123)
model <- train(
  RiskLevel ~ ., 
  data = trainData,
  method="rpart",                     # Model type(decision tree)
  trControl= control,           # Model control options
  tuneGrid = tune_grid,               # Required model parameters
  maxdepth = 5,                       # Additional parameters***
  minbucket=5)   

model

# Make predictions
predictions <- predict(model, testData)


# Evaluate the model
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(testData$RiskLevel))
conf_matrix

#Update row and column names to reflect risk levels
rownames(conf_matrix$table) <- c('Low Risk', 'High Risk')
colnames(conf_matrix$table) <- c('Low Risk', 'High Risk')
conf_matrix$table <- t(conf_matrix$table)  # Transpose to match predicted as columns and actual as rows
colnames(conf_matrix$table) <- c('Low Risk', 'High Risk')  # Predicted
rownames(conf_matrix$table) <- c('Low Risk', 'High Risk')  # Actual
print(conf_matrix$table)

recall <- conf_matrix$byClass["Sensitivity"]        # Recall
precision <- conf_matrix$byClass["Pos Pred Value"]  # Precision
f1_score <- 2 * (precision * recall) / (precision + recall)  # F1-Score

#  metrics
cat(sprintf("Precision: %.2f\n", precision))
cat(sprintf("Recall: %.2f\n", recall))
cat(sprintf("F1-Score: %.2f\n", f1_score))



##################

# Create prediction object
predictions <- predict(model, testData, type = "prob")[, 2] 
predictions_rocr <- prediction(predictions, testData$RiskLevel)

# Calculate performance (True Positive Rate and False Positive Rate)
perf <- performance(predictions_rocr, "tpr", "fpr")

# Plot the ROC curve
plot(perf, col = "darkgreen", lwd = 2, main = "ROC Curve for Decision tree Model")

# Add a diagonal reference line
abline(a = 0, b = 1, col = "red", lty = 2)


##########





# Generate and Plot the ROC Curve
roc_curve <- roc(testData$RiskLevel,predictions)

# Plot the ROC curve
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Decision tree Model")
abline(a = 0, b = 1, lty = 2, col = "red")  # Diagonal line for random guessing

# Calculate and display AUC
auc_value <- auc(roc_curve)
cat("AUC for Decision tree Model:", auc_value, "\n")

## checking balanicng
table(trainData$RiskLevel)


## balancing

# Separate the majority and minority classes
majority_class <- trainData[trainData$RiskLevel == 1, ]
minority_class <- trainData[trainData$RiskLevel == 0, ]

# Oversample the minority class
set.seed(42)
minority_oversampled <- minority_class[sample(1:nrow(minority_class), nrow(majority_class), replace = TRUE), ]

# Combine the oversampled minority class with the majority class
balanced_train_data <- rbind(majority_class, minority_oversampled)

# Shuffle the balanced dataset
balanced_train_data <- balanced_train_data[sample(1:nrow(balanced_train_data)), ]

# Separate the features and target variable again
X_train_balanced <- balanced_train_data[, !names(balanced_train_data) %in% c('RiskLevel')]
Y_train_balanced <- balanced_train_data$RiskLevel

# Output the counts of each class in the balanced dataset
cat("Balanced Training Data Class Distribution:\n")
print(table(Y_train_balanced))
################################
#after balacing testign the data

# Retrain the model on the balanced dataset
set.seed(123)
balanced_model <- train(
  RiskLevel ~ ., 
  data = balanced_train_data,        # Use the balanced training data
  method = "rpart",                  # Model type (decision tree)
  trControl = control,               # Cross-validation settings
  tuneGrid = tune_grid,              # Required model parameters
  maxdepth = 5,                      # Additional parameters
  minbucket = 5
)

# Evaluate the model on the test dataset
balanced_predictions <- predict(balanced_model, testData)

# Generate confusion matrix
balanced_conf_matrix <- confusionMatrix(as.factor(balanced_predictions), as.factor(testData$RiskLevel))

# Update row and column names to reflect risk levels
rownames(balanced_conf_matrix$table) <- c('Low Risk', 'High Risk')
colnames(balanced_conf_matrix$table) <- c('Low Risk', 'High Risk')
balanced_conf_matrix$table <- t(balanced_conf_matrix$table)  # Transpose for consistency
colnames(balanced_conf_matrix$table) <- c('Low Risk', 'High Risk')  # Predicted
rownames(balanced_conf_matrix$table) <- c('Low Risk', 'High Risk')  # Actual

# Print confusion matrix
cat("Confusion Matrix after Balancing the Dataset:\n")
print(balanced_conf_matrix$table)

# Calculate metrics
recall_balanced <- balanced_conf_matrix$byClass["Sensitivity"]
precision_balanced <- balanced_conf_matrix$byClass["Pos Pred Value"]
f1_score_balanced <- 2 * (precision_balanced * recall_balanced) / (precision_balanced + recall_balanced)

# Print metrics
cat(sprintf("Precision after Balancing: %.2f\n", precision_balanced))
cat(sprintf("Recall after Balancing: %.2f\n", recall_balanced))
cat(sprintf("F1-Score after Balancing: %.2f\n", f1_score_balanced))


##################

# Create prediction object
predictions <- predict(balanced_model, testData, type = "prob")[, 2] 
predictions_rocr <- prediction(predictions, testData$RiskLevel)

# Calculate performance (True Positive Rate and False Positive Rate)
perf <- performance(predictions_rocr, "tpr", "fpr")

# Plot the ROC curve
plot(perf, col = "darkgreen", lwd = 2, main = "ROC Curve after Balancing")

# Add a diagonal reference line
abline(a = 0, b = 1, col = "red", lty = 2)

##########

# Generate and plot the ROC curve
roc_curve_balanced <- roc(testData$RiskLevel, as.numeric(balanced_predictions))
plot(roc_curve_balanced, col = "blue", lwd = 2, main = "ROC Curve after Balancing")
abline(a = 0, b = 1, lty = 2, col = "red")

# Calculate and display AUC
auc_value_balanced <- auc(roc_curve_balanced)
cat("AUC after Balancing:", auc_value_balanced, "\n")
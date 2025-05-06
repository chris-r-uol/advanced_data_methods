# Load libraries
library(ggplot2)
library(caret)
library(rpart.plot)

# Use the iris dataset for multi-class classification
data(iris)
set.seed(42)

# Create training and test splits (70/30 split)
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data  <- iris[-train_index, ]

# Train a simple logistic regression (multinomial via nnet) or decision tree model
model_class <- train(Species ~ ., data = train_data, method = "rpart")

# Predict on the test set and evaluate
pred_class <- predict(model_class, test_data)
conf_matrix <- confusionMatrix(pred_class, test_data$Species)
print(conf_matrix)

# Convert the confusion matrix table to a data frame
conf_df <- as.data.frame(conf_matrix$table)

# Plot a heatmap of the confusion matrix
ggplot(conf_df, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(title = "Confusion Matrix Heatmap", x = "Predicted Label", y = "Actual Label") +
  theme_minimal()



# Visualize the decision tree
rpart.plot(model_class$finalModel, 
           main = "Decision Tree for Iris Classification",
           extra = 104)

importance <- varImp(model_class)
print(importance)

# Plot variable importance
plot(importance, main = "Variable Importance")
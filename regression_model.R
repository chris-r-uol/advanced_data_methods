# Load libraries
library(ggplot2)
library(caret)

data(mtcars)
set.seed(42)

# Split data into training and test sets (70/30 split)
train_index <- createDataPartition(mtcars$mpg, p = 0.7, list = FALSE)
train_data <- mtcars[train_index, ]
test_data  <- mtcars[-train_index, ]

# Train a linear regression model to predict mpg
model_reg <- train(mpg ~ ., data = train_data, method = "lm")

# Predict on the test set and calculate performance metrics
pred_reg <- predict(model_reg, test_data)
rmse <- sqrt(mean((pred_reg - test_data$mpg)^2))
cat("Test RMSE:", round(rmse, 2), "\n")

# Create a data frame with actual, predicted values, and residuals
results <- data.frame(
  actual = test_data$mpg,
  predicted = pred_reg,
  residuals = test_data$mpg - pred_reg
)

# Scatter plot: Actual vs. Predicted
ggplot(results, aes(x = actual, y = predicted)) +
  geom_point(color = "blue", size = 3) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted MPG",
       x = "Actual MPG",
       y = "Predicted MPG") +
  theme_bw()
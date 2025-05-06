# Load necessary library
library(ggplot2)

# Simulate some data
set.seed(123)
n <- 100
x <- sort(runif(n, 0, 10))
y <- sin(x) + rnorm(n, sd = 0.5)  # noisy sine wave
data <- data.frame(x = x, y = y)

# Fit an underfit model: linear regression
model_under <- lm(y ~ x, data = data)

# Fit a good model: 3rd degree polynomial
model_good <- lm(y ~ poly(x, 6), data = data)

# Fit an overfit model: 10th degree polynomial
model_over <- lm(y ~ poly(x, 30), data = data)

# Create a grid for predictions
x_grid <- seq(0, 10, length.out = 200)
pred_data <- data.frame(x = x_grid)
pred_data$true <- sin(x_grid)
pred_data$under <- predict(model_under, newdata = pred_data)
pred_data$good  <- predict(model_good, newdata = pred_data)
pred_data$over  <- predict(model_over, newdata = pred_data)
pred_data$y = data$y

# Plot the true function and predictions from the three models
ggplot(data, aes(x = x, y = y)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_line(data = pred_data, aes(x = x, y = true),
            color = "#666666", linewidth = 1, linetype = "dashed") +
  geom_line(data = pred_data, aes(x = x, y = under),
            color = "#ff5733", linewidth = 1, linetype = "solid") +
  geom_line(data = pred_data, aes(x = x, y = good),
            color = "dodgerblue", linewidth = 1, linetype = "solid") +
  geom_line(data = pred_data, aes(x = x, y = over),
            color = "lightseagreen", linewidth = 1, linetype = "solid") +
  labs(title = "Demonstrating Underfitting vs. Overfitting",
       subtitle = "Red: Underfit (Linear), Blue: Good Fit (6th degree), Green: Overfit (30th degree)",
       x = "x",
       y = "y (predicted)") +
  theme_bw() + ylim(-2, 2) + theme(legend.position = "top")

p2 = ggplot(pred_data, aes(x = true, y = over)) + theme_bw() +
  geom_point() + 
  geom_smooth(method = 'lm')

print(p2)
            

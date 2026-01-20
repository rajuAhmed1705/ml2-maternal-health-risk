# ============================================================================
# R Script to Generate Plots for Bayes/LDA/QDA/Naive Bayes Notes
# ============================================================================

# Load required libraries
library(ggplot2)
library(gridExtra)
library(MASS)

# Create output directory for images
if (!dir.exists("images")) {
  dir.create("images")
}

# Set consistent theme
theme_set(theme_minimal(base_size = 12))

# ============================================================================
# Plot 1: Regression vs Classification
# ============================================================================
set.seed(123)

# Regression data
reg_data <- data.frame(
  x = seq(50, 150, length.out = 50),
  y = 50000 + 1500 * seq(50, 150, length.out = 50) + rnorm(50, 0, 15000)
)

p1_reg <- ggplot(reg_data, aes(x = x, y = y)) +
  geom_point(color = "steelblue", alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", se = FALSE, color = "darkblue", linewidth = 1) +
  labs(title = "Regression",
       subtitle = "Predict a continuous value",
       x = "Size (m²)",
       y = "Price (€)") +
  scale_y_continuous(labels = scales::comma) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Classification data
class_data <- data.frame(
  x1 = c(rnorm(30, 2, 0.8), rnorm(30, 5, 0.8)),
  x2 = c(rnorm(30, 2, 0.8), rnorm(30, 5, 0.8)),
  class = factor(rep(c("Class A", "Class B"), each = 30))
)

p1_class <- ggplot(class_data, aes(x = x1, y = x2, color = class, shape = class)) +
  geom_point(size = 3, alpha = 0.8) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", linewidth = 1) +
  scale_color_manual(values = c("steelblue", "coral")) +
  labs(title = "Classification",
       subtitle = "Predict a category/class",
       x = "Feature X₁",
       y = "Feature X₂",
       color = "Class", shape = "Class") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

p1_combined <- grid.arrange(p1_reg, p1_class, ncol = 2)
ggsave("images/01_regression_vs_classification.png", p1_combined, width = 10, height = 5, dpi = 150)

# ============================================================================
# Plot 2: Binary Classification Posterior Probability
# ============================================================================
x_seq <- seq(-3, 3, length.out = 100)
posterior <- 1 / (1 + exp(-3 * x_seq))

p2 <- ggplot(data.frame(x = x_seq, prob = posterior), aes(x = x, y = prob)) +
  geom_line(color = "steelblue", linewidth = 1.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", linewidth = 1) +
  geom_vline(xintercept = 0, linetype = "dotted", color = "gray40") +
  annotate("rect", xmin = -3, xmax = 0, ymin = 0, ymax = 1, fill = "coral", alpha = 0.1) +
  annotate("rect", xmin = 0, xmax = 3, ymin = 0, ymax = 1, fill = "steelblue", alpha = 0.1) +
  annotate("text", x = -1.5, y = 0.9, label = "Predict Class 1", color = "coral", fontface = "bold") +
  annotate("text", x = 1.5, y = 0.9, label = "Predict Class 2", color = "steelblue", fontface = "bold") +
  annotate("text", x = 0.5, y = 0.55, label = "Threshold = 0.5", color = "red") +
  labs(title = "Binary Classification Decision",
       subtitle = "Assign to Class 2 if P(Y = Class 2 | X) > 0.5",
       x = "Feature X",
       y = "P(Y = Class 2 | X)") +
  scale_y_continuous(breaks = seq(0, 1, 0.2)) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("images/02_binary_classification_posterior.png", p2, width = 8, height = 5, dpi = 150)

# ============================================================================
# Plot 3: Bayes Error Rate - Overlapping Classes
# ============================================================================
x_seq <- seq(-4, 8, length.out = 200)
class1 <- dnorm(x_seq, mean = 1, sd = 1.2)
class2 <- dnorm(x_seq, mean = 4, sd = 1.2)

df_bayes_error <- data.frame(
  x = rep(x_seq, 2),
  density = c(class1, class2),
  class = rep(c("Class 1", "Class 2"), each = 200)
)

# Decision boundary at intersection
boundary <- 2.5

p3 <- ggplot(df_bayes_error, aes(x = x, y = density, fill = class)) +
  geom_area(alpha = 0.4, position = "identity") +
  geom_line(aes(color = class), linewidth = 1) +
  geom_vline(xintercept = boundary, linetype = "dashed", color = "black", linewidth = 1) +
  # Shade error regions
  geom_ribbon(data = data.frame(x = x_seq[x_seq > boundary],
                                 y = class1[x_seq > boundary]),
              aes(x = x, ymin = 0, ymax = y),
              fill = "red", alpha = 0.5, inherit.aes = FALSE) +
  geom_ribbon(data = data.frame(x = x_seq[x_seq < boundary],
                                 y = class2[x_seq < boundary]),
              aes(x = x, ymin = 0, ymax = y),
              fill = "red", alpha = 0.5, inherit.aes = FALSE) +
  scale_fill_manual(values = c("steelblue", "coral")) +
  scale_color_manual(values = c("darkblue", "darkred")) +
  annotate("text", x = boundary, y = max(class1) * 1.1, label = "Decision\nBoundary",
           hjust = 0.5, fontface = "bold") +
  annotate("text", x = 3.5, y = 0.05, label = "Bayes Error\n(unavoidable)",
           color = "red", fontface = "bold") +
  labs(title = "Bayes Error Rate",
       subtitle = "The irreducible error where classes overlap",
       x = "Feature X",
       y = "Density") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

ggsave("images/03_bayes_error_rate.png", p3, width = 8, height = 5, dpi = 150)

# ============================================================================
# Plot 4: LDA Assumption - Normal Distributions with Equal Variance
# ============================================================================
x_seq <- seq(1.4, 2.1, length.out = 200)
female <- dnorm(x_seq, mean = 1.65, sd = 0.08)
male <- dnorm(x_seq, mean = 1.80, sd = 0.08)

df_lda <- data.frame(
  x = rep(x_seq, 2),
  density = c(female, male),
  sex = rep(c("Female", "Male"), each = 200)
)

p4 <- ggplot(df_lda, aes(x = x, y = density, fill = sex)) +
  geom_area(alpha = 0.4, position = "identity") +
  geom_line(aes(color = sex), linewidth = 1) +
  geom_vline(xintercept = 1.715, linetype = "dashed", color = "black", linewidth = 1) +
  scale_fill_manual(values = c("coral", "steelblue")) +
  scale_color_manual(values = c("darkred", "darkblue")) +
  annotate("segment", x = 1.65, xend = 1.80, y = max(female) * 0.3, yend = max(female) * 0.3,
           arrow = arrow(ends = "both", length = unit(0.2, "cm"))) +
  annotate("text", x = 1.725, y = max(female) * 0.35, label = "Same σ²", fontface = "bold") +
  annotate("text", x = 1.65, y = -0.3, label = "μ_female", fontface = "italic") +
  annotate("text", x = 1.80, y = -0.3, label = "μ_male", fontface = "italic") +
  annotate("text", x = 1.715, y = max(female) * 1.05, label = "Decision Boundary\nx = 1.715m",
           hjust = 0.5, fontface = "bold", size = 3.5) +
  labs(title = "LDA: Class-Conditional Densities",
       subtitle = "Normal distributions with EQUAL variance (σ²) - Student height example",
       x = "Height (m)",
       y = "Density") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

ggsave("images/04_lda_class_conditional.png", p4, width = 8, height = 5, dpi = 150)

# ============================================================================
# Plot 5: LDA Posterior Probability Curve
# ============================================================================
x_seq <- seq(1.4, 2.1, length.out = 200)
# Using prior = 0.66 for male (from slides)
pi0 <- 0.66
mu_m <- 1.80
mu_f <- 1.65
sigma <- 0.08

posterior_male <- function(x) {
  lik_m <- dnorm(x, mu_m, sigma)
  lik_f <- dnorm(x, mu_f, sigma)
  (pi0 * lik_m) / (pi0 * lik_m + (1 - pi0) * lik_f)
}

df_posterior <- data.frame(
  x = x_seq,
  prob = sapply(x_seq, posterior_male)
)

p5 <- ggplot(df_posterior, aes(x = x, y = prob)) +
  geom_line(color = "steelblue", linewidth = 1.5) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "red", linewidth = 1) +
  geom_vline(xintercept = 1.715, linetype = "dotted", color = "gray40") +
  annotate("rect", xmin = 1.4, xmax = 1.715, ymin = 0, ymax = 1, fill = "coral", alpha = 0.1) +
  annotate("rect", xmin = 1.715, xmax = 2.1, ymin = 0, ymax = 1, fill = "steelblue", alpha = 0.1) +
  annotate("text", x = 1.55, y = 0.9, label = "Predict\nFemale", color = "coral", fontface = "bold") +
  annotate("text", x = 1.9, y = 0.9, label = "Predict\nMale", color = "steelblue", fontface = "bold") +
  annotate("point", x = 1.715, y = 0.5, size = 4, color = "red") +
  annotate("text", x = 1.78, y = 0.55, label = "x = 1.715m\n(P = 0.5)", size = 3.5) +
  labs(title = "LDA: Posterior Probability P(Male | Height)",
       subtitle = "Prior: P(Male) = 0.66 (engineering students)",
       x = "Height (m)",
       y = "P(Male | Height)") +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

ggsave("images/05_lda_posterior_probability.png", p5, width = 8, height = 5, dpi = 150)

# ============================================================================
# Plot 6: LDA with 2 Variables - Contour Plot and Decision Regions
# ============================================================================
set.seed(123)

# Generate bivariate normal data
mu1 <- c(2.2, 13.5)
mu2 <- c(3.9, 11.8)
Sigma <- matrix(c(1, 0.5, 0.5, 1), 2, 2)

n <- 100
class1_data <- mvrnorm(n, mu1, Sigma)
class2_data <- mvrnorm(n, mu2, Sigma)

df_2d <- data.frame(
  x = c(class1_data[,1], class2_data[,1]),
  y = c(class1_data[,2], class2_data[,2]),
  class = factor(rep(c("Class 1", "Class 2"), each = n))
)

# Create grid for decision boundary
x_grid <- seq(0, 6, length.out = 100)
y_grid <- seq(9, 16, length.out = 100)
grid <- expand.grid(x = x_grid, y = y_grid)

# LDA decision boundary (linear)
# For equal priors, boundary is perpendicular bisector of line connecting means
# Midpoint: (3.05, 12.65)
# Direction: perpendicular to (mu2 - mu1) = (1.7, -1.7)
# Boundary: y - 12.65 = 1 * (x - 3.05) => y = x + 9.6

p6 <- ggplot(df_2d, aes(x = x, y = y, color = class)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_ellipse(level = 0.95, linewidth = 1) +
  geom_abline(intercept = 9.6, slope = 1, linetype = "dashed", color = "black", linewidth = 1) +
  scale_color_manual(values = c("steelblue", "coral")) +
  annotate("point", x = mu1[1], y = mu1[2], size = 5, shape = 4, stroke = 2, color = "darkblue") +
  annotate("point", x = mu2[1], y = mu2[2], size = 5, shape = 4, stroke = 2, color = "darkred") +
  annotate("text", x = mu1[1] - 0.3, y = mu1[2] + 0.5, label = "μ₁", fontface = "bold", color = "darkblue") +
  annotate("text", x = mu2[1] + 0.3, y = mu2[2] - 0.5, label = "μ₂", fontface = "bold", color = "darkred") +
  annotate("text", x = 5, y = 15, label = "LINEAR\nBoundary", fontface = "bold") +
  labs(title = "LDA with Two Variables",
       subtitle = "Same covariance Σ for both classes → Linear decision boundary",
       x = "Feature X₁",
       y = "Feature X₂") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

ggsave("images/06_lda_2d_contour.png", p6, width = 8, height = 6, dpi = 150)

# ============================================================================
# Plot 7: LDA vs QDA Decision Boundaries
# ============================================================================
set.seed(123)

# LDA data (same covariance)
Sigma_same <- matrix(c(1, 0.3, 0.3, 1), 2, 2)
lda_class1 <- mvrnorm(80, c(2, 2), Sigma_same)
lda_class2 <- mvrnorm(80, c(4, 4), Sigma_same)

# QDA data (different covariances)
Sigma1 <- matrix(c(0.5, 0, 0, 0.5), 2, 2)  # Small, circular
Sigma2 <- matrix(c(2, 0.8, 0.8, 2), 2, 2)  # Large, elongated
qda_class1 <- mvrnorm(80, c(2, 2), Sigma1)
qda_class2 <- mvrnorm(80, c(4, 4), Sigma2)

df_lda_data <- data.frame(
  x = c(lda_class1[,1], lda_class2[,1]),
  y = c(lda_class1[,2], lda_class2[,2]),
  class = factor(rep(c("Class 1", "Class 2"), each = 80))
)

df_qda_data <- data.frame(
  x = c(qda_class1[,1], qda_class2[,1]),
  y = c(qda_class2[,2], qda_class2[,2]),
  class = factor(rep(c("Class 1", "Class 2"), each = 80))
)

p7_lda <- ggplot(df_lda_data, aes(x = x, y = y, color = class)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_ellipse(level = 0.9, linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", linewidth = 1.2) +
  scale_color_manual(values = c("steelblue", "coral")) +
  labs(title = "LDA: Linear Boundary",
       subtitle = "Equal covariances",
       x = "X₁", y = "X₂") +
  coord_fixed() +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "none")

# For QDA, we need actual different covariances
qda_class1 <- mvrnorm(80, c(2, 4), Sigma1)
qda_class2 <- mvrnorm(80, c(5, 3), Sigma2)

df_qda_data <- data.frame(
  x = c(qda_class1[,1], qda_class2[,1]),
  y = c(qda_class1[,2], qda_class2[,2]),
  class = factor(rep(c("Class 1", "Class 2"), each = 80))
)

p7_qda <- ggplot(df_qda_data, aes(x = x, y = y, color = class)) +
  geom_point(alpha = 0.6, size = 2) +
  stat_ellipse(level = 0.9, linewidth = 1) +
  # Add curved boundary approximation
  stat_function(fun = function(x) 5.5 - 0.3*(x-3.5)^2,
                color = "black", linetype = "dashed", linewidth = 1.2,
                xlim = c(0, 7)) +
  scale_color_manual(values = c("steelblue", "coral")) +
  labs(title = "QDA: Quadratic Boundary",
       subtitle = "Different covariances",
       x = "X₁", y = "X₂") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "none")

p7_combined <- grid.arrange(p7_lda, p7_qda, ncol = 2)
ggsave("images/07_lda_vs_qda.png", p7_combined, width = 10, height = 5, dpi = 150)

# ============================================================================
# Plot 8: Three-Class LDA (Berlin Rent Zones Example)
# ============================================================================
x_seq <- seq(5, 20, length.out = 200)
zone_c <- dnorm(x_seq, mean = 8, sd = 2)
zone_b <- dnorm(x_seq, mean = 11, sd = 2)
zone_a <- dnorm(x_seq, mean = 15, sd = 2)

df_3class <- data.frame(
  x = rep(x_seq, 3),
  density = c(zone_c, zone_b, zone_a),
  zone = rep(c("Zone C (outer)", "Zone B (middle)", "Zone A (center)"), each = 200)
)

p8_density <- ggplot(df_3class, aes(x = x, y = density, fill = zone, color = zone)) +
  geom_area(alpha = 0.3, position = "identity") +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = c(10.02, 12.15), linetype = "dashed", color = "black") +
  scale_fill_manual(values = c("coral", "gold", "steelblue")) +
  scale_color_manual(values = c("darkred", "darkgoldenrod", "darkblue")) +
  annotate("text", x = 8, y = max(zone_c) * 1.1, label = "Zone C", fontface = "bold") +
  annotate("text", x = 11, y = max(zone_b) * 1.1, label = "Zone B", fontface = "bold") +
  annotate("text", x = 15, y = max(zone_a) * 1.1, label = "Zone A", fontface = "bold") +
  labs(title = "Three-Class LDA: Berlin Rent Zones",
       subtitle = "Class-conditional densities based on rent per m²",
       x = "Rent (€/m²)",
       y = "Density") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

ggsave("images/08_three_class_lda.png", p8_density, width = 9, height = 5, dpi = 150)

# ============================================================================
# Plot 8b: Three-Class Posterior Probability (Berlin Rent Zones)
# ============================================================================
# This shows P(Zone | Rent) for all three zones - from Topic 3 slides p.11

x_seq <- seq(5, 20, length.out = 200)

# Parameters (estimated from slide context)
mu_C <- 8    # Zone C (outer) - lower rent
mu_B <- 11   # Zone B (middle)
mu_A <- 15   # Zone A (center) - higher rent
sigma <- 2   # Same variance (LDA assumption)

# Equal priors for simplicity
prior <- 1/3

# Calculate posteriors using Bayes' theorem
calc_posterior <- function(x, mu_target, mu_others, sigma, prior) {
  lik_target <- dnorm(x, mu_target, sigma)
  lik_others <- sapply(mu_others, function(mu) dnorm(x, mu, sigma))
  total_lik <- prior * lik_target + sum(prior * lik_others)
  (prior * lik_target) / total_lik
}

posterior_C <- sapply(x_seq, function(x) calc_posterior(x, mu_C, c(mu_B, mu_A), sigma, prior))
posterior_B <- sapply(x_seq, function(x) calc_posterior(x, mu_B, c(mu_C, mu_A), sigma, prior))
posterior_A <- sapply(x_seq, function(x) calc_posterior(x, mu_A, c(mu_C, mu_B), sigma, prior))

df_posterior_3class <- data.frame(
  rent = rep(x_seq, 3),
  posterior = c(posterior_C, posterior_B, posterior_A),
  zone = factor(rep(c("Zone C (outer)", "Zone B (middle)", "Zone A (center)"), each = 200),
                levels = c("Zone C (outer)", "Zone B (middle)", "Zone A (center)"))
)

p8b <- ggplot(df_posterior_3class, aes(x = rent, y = posterior, color = zone)) +
  geom_line(linewidth = 1.2) +
  geom_hline(yintercept = 1/3, linetype = "dotted", color = "gray50") +
  geom_vline(xintercept = c(9.5, 13), linetype = "dashed", color = "gray40", alpha = 0.7) +
  scale_color_manual(values = c("steelblue", "gold3", "coral")) +
  annotate("text", x = 7, y = 0.9, label = "Zone C\nmost likely", color = "steelblue", fontface = "bold", size = 3.5) +
  annotate("text", x = 11, y = 0.55, label = "Zone B\nmost likely", color = "gold3", fontface = "bold", size = 3.5) +
  annotate("text", x = 16, y = 0.9, label = "Zone A\nmost likely", color = "coral", fontface = "bold", size = 3.5) +
  annotate("text", x = 9.5, y = 0.05, label = "Boundary 1", size = 3) +
  annotate("text", x = 13, y = 0.05, label = "Boundary 2", size = 3) +
  labs(title = "Three-Class LDA: Posterior Probabilities P(Zone | Rent)",
       subtitle = "Decision boundaries where posterior probabilities cross",
       x = "Rent (EUR/m²)",
       y = "P(Zone | Rent)",
       color = "BVG Zone") +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

ggsave("images/08b_three_class_posterior.png", p8b, width = 9, height = 5, dpi = 150)

# ============================================================================
# Plot 9: Conditional Independence Visualization
# ============================================================================
# Create a simple diagram showing conditional independence

library(grid)

png("images/09_conditional_independence.png", width = 800, height = 400, res = 100)

grid.newpage()
pushViewport(viewport(width = 1, height = 1))

# Left side - NOT conditionally independent
pushViewport(viewport(x = 0.25, y = 0.5, width = 0.4, height = 0.9))
grid.text("NOT Conditionally Independent", x = 0.5, y = 0.95, gp = gpar(fontface = "bold", fontsize = 12))

# Draw boxes
grid.rect(x = 0.2, y = 0.7, width = 0.25, height = 0.15, gp = gpar(fill = "lightblue"))
grid.text("X₁", x = 0.2, y = 0.7, gp = gpar(fontsize = 11))

grid.rect(x = 0.8, y = 0.7, width = 0.25, height = 0.15, gp = gpar(fill = "lightblue"))
grid.text("X₂", x = 0.8, y = 0.7, gp = gpar(fontsize = 11))

grid.rect(x = 0.5, y = 0.3, width = 0.25, height = 0.15, gp = gpar(fill = "lightyellow"))
grid.text("Y", x = 0.5, y = 0.3, gp = gpar(fontsize = 11))

# Draw arrows
grid.lines(x = c(0.2, 0.8), y = c(0.7, 0.7), arrow = arrow(ends = "both", length = unit(0.1, "inches")))
grid.lines(x = c(0.2, 0.5), y = c(0.62, 0.38), arrow = arrow(length = unit(0.1, "inches")))
grid.lines(x = c(0.8, 0.5), y = c(0.62, 0.38), arrow = arrow(length = unit(0.1, "inches")))

grid.text("X₁ and X₂ directly related\neven after knowing Y", x = 0.5, y = 0.08, gp = gpar(fontsize = 9))
popViewport()

# Right side - Conditionally independent
pushViewport(viewport(x = 0.75, y = 0.5, width = 0.4, height = 0.9))
grid.text("Conditionally Independent", x = 0.5, y = 0.95, gp = gpar(fontface = "bold", fontsize = 12))

# Draw boxes
grid.rect(x = 0.2, y = 0.7, width = 0.25, height = 0.15, gp = gpar(fill = "lightblue"))
grid.text("X₁", x = 0.2, y = 0.7, gp = gpar(fontsize = 11))

grid.rect(x = 0.8, y = 0.7, width = 0.25, height = 0.15, gp = gpar(fill = "lightblue"))
grid.text("X₂", x = 0.8, y = 0.7, gp = gpar(fontsize = 11))

grid.rect(x = 0.5, y = 0.3, width = 0.25, height = 0.15, gp = gpar(fill = "lightyellow"))
grid.text("Y", x = 0.5, y = 0.3, gp = gpar(fontsize = 11))

# Draw arrows (only from Y)
grid.lines(x = c(0.5, 0.2), y = c(0.38, 0.62), arrow = arrow(length = unit(0.1, "inches")))
grid.lines(x = c(0.5, 0.8), y = c(0.38, 0.62), arrow = arrow(length = unit(0.1, "inches")))

grid.text("Y is the common cause\nOnce Y known, X₁ ⊥ X₂", x = 0.5, y = 0.08, gp = gpar(fontsize = 9))
popViewport()

dev.off()

# ============================================================================
# Plot 10: Naive Bayes - Categorical Variable Likelihoods
# ============================================================================
# Data from the slides example
nb_data <- data.frame(
  category = rep(c("A", "B", "C", "D"), 2),
  count = c(26, 27, 7, 4, 5, 5, 2, 0),
  class = rep(c("Y = Yes (n=64)", "Y = No (n=12)"), each = 4)
)

nb_data$proportion <- c(26/64, 27/64, 7/64, 4/64, 5/12, 5/12, 2/12, 0/12)

p10 <- ggplot(nb_data, aes(x = category, y = proportion, fill = class)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", proportion)),
            position = position_dodge(width = 0.7), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("steelblue", "coral")) +
  labs(title = "Naive Bayes: Likelihood Estimation for Categorical Variable",
       subtitle = "P(X = category | Y = class) = count / total in class",
       x = "Category",
       y = "P(X = category | Y)",
       fill = "Class") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom") +
  ylim(0, 0.5)

ggsave("images/10_naive_bayes_categorical.png", p10, width = 8, height = 5, dpi = 150)

# ============================================================================
# Plot 11: Naive Bayes - Continuous Variables (Feature Independence)
# ============================================================================
set.seed(123)
x_seq <- seq(-2, 8, length.out = 200)

# Feature 1 distributions
f1_c1 <- dnorm(x_seq, mean = 2, sd = 1)
f1_c2 <- dnorm(x_seq, mean = 5, sd = 1)

# Feature 2 distributions
f2_c1 <- dnorm(x_seq, mean = 3, sd = 1.5)
f2_c2 <- dnorm(x_seq, mean = 6, sd = 1.5)

df_f1 <- data.frame(
  x = rep(x_seq, 2),
  density = c(f1_c1, f1_c2),
  class = rep(c("Class 1", "Class 2"), each = 200)
)

df_f2 <- data.frame(
  x = rep(x_seq, 2),
  density = c(f2_c1, f2_c2),
  class = rep(c("Class 1", "Class 2"), each = 200)
)

p11_f1 <- ggplot(df_f1, aes(x = x, y = density, fill = class, color = class)) +
  geom_area(alpha = 0.3, position = "identity") +
  geom_line(linewidth = 1) +
  scale_fill_manual(values = c("steelblue", "coral")) +
  scale_color_manual(values = c("darkblue", "darkred")) +
  labs(title = "Feature X₁",
       subtitle = "P(X₁ | Y)",
       x = "X₁", y = "Density") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

p11_f2 <- ggplot(df_f2, aes(x = x, y = density, fill = class, color = class)) +
  geom_area(alpha = 0.3, position = "identity") +
  geom_line(linewidth = 1) +
  scale_fill_manual(values = c("steelblue", "coral")) +
  scale_color_manual(values = c("darkblue", "darkred")) +
  labs(title = "Feature X₂",
       subtitle = "P(X₂ | Y)",
       x = "X₂", y = "Density") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "bottom")

p11_combined <- grid.arrange(
  p11_f1, p11_f2, ncol = 2,
  top = grid::textGrob("Naive Bayes: Each Feature Estimated Independently",
                       gp = grid::gpar(fontface = "bold", fontsize = 14))
)
ggsave("images/11_naive_bayes_continuous.png", p11_combined, width = 10, height = 5, dpi = 150)

# ============================================================================
# Plot 12: Covariance Structure Comparison (LDA vs Naive Bayes)
# ============================================================================
set.seed(123)

# LDA - full covariance (correlated features)
Sigma_full <- matrix(c(1, 0.7, 0.7, 1), 2, 2)
lda_data <- mvrnorm(200, c(0, 0), Sigma_full)

# Naive Bayes - diagonal covariance (independent features)
Sigma_diag <- matrix(c(1, 0, 0, 1), 2, 2)
nb_data_plot <- mvrnorm(200, c(0, 0), Sigma_diag)

df_cov <- data.frame(
  x = c(lda_data[,1], nb_data_plot[,1]),
  y = c(lda_data[,2], nb_data_plot[,2]),
  type = rep(c("LDA: Full Covariance\n(features correlated)",
               "Naive Bayes: Diagonal\n(features independent)"), each = 200)
)

p12 <- ggplot(df_cov, aes(x = x, y = y)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  stat_ellipse(level = 0.95, color = "darkblue", linewidth = 1.2) +
  facet_wrap(~type) +
  coord_fixed() +
  labs(title = "Covariance Structure: LDA vs Naive Bayes",
       subtitle = "LDA allows correlation; Naive Bayes assumes independence",
       x = "Feature X₁",
       y = "Feature X₂") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        strip.text = element_text(face = "bold"))

ggsave("images/12_covariance_comparison.png", p12, width = 10, height = 5, dpi = 150)

# ============================================================================
# Plot 13: Decision Boundary Comparison (LDA vs QDA vs NB)
# ============================================================================
set.seed(42)

# Generate data for comparison
n <- 150
class1_x <- mvrnorm(n, c(2, 2), matrix(c(1, 0.5, 0.5, 1), 2, 2))
class2_x <- mvrnorm(n, c(5, 5), matrix(c(1.5, 0.3, 0.3, 1.5), 2, 2))

df_compare <- data.frame(
  x = c(class1_x[,1], class2_x[,1]),
  y = c(class1_x[,2], class2_x[,2]),
  class = factor(rep(c("Class 1", "Class 2"), each = n))
)

# Fit models
lda_fit <- lda(class ~ x + y, data = df_compare)
qda_fit <- qda(class ~ x + y, data = df_compare)

# Create prediction grid
x_range <- seq(min(df_compare$x) - 1, max(df_compare$x) + 1, length.out = 100)
y_range <- seq(min(df_compare$y) - 1, max(df_compare$y) + 1, length.out = 100)
grid_df <- expand.grid(x = x_range, y = y_range)

# Predictions
grid_df$lda_pred <- predict(lda_fit, grid_df)$class
grid_df$qda_pred <- predict(qda_fit, grid_df)$class

p13_lda <- ggplot() +
  geom_tile(data = grid_df, aes(x = x, y = y, fill = lda_pred), alpha = 0.3) +
  geom_point(data = df_compare, aes(x = x, y = y, color = class), size = 1.5) +
  scale_fill_manual(values = c("steelblue", "coral"), guide = "none") +
  scale_color_manual(values = c("darkblue", "darkred")) +
  labs(title = "LDA", subtitle = "Linear boundary") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

p13_qda <- ggplot() +
  geom_tile(data = grid_df, aes(x = x, y = y, fill = qda_pred), alpha = 0.3) +
  geom_point(data = df_compare, aes(x = x, y = y, color = class), size = 1.5) +
  scale_fill_manual(values = c("steelblue", "coral"), guide = "none") +
  scale_color_manual(values = c("darkblue", "darkred")) +
  labs(title = "QDA", subtitle = "Quadratic boundary") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

# For Naive Bayes, simulate axis-aligned regions
grid_df$nb_pred <- ifelse(grid_df$x + grid_df$y < 7, "Class 1", "Class 2")

p13_nb <- ggplot() +
  geom_tile(data = grid_df, aes(x = x, y = y, fill = nb_pred), alpha = 0.3) +
  geom_point(data = df_compare, aes(x = x, y = y, color = class), size = 1.5) +
  scale_fill_manual(values = c("steelblue", "coral"), guide = "none") +
  scale_color_manual(values = c("darkblue", "darkred")) +
  labs(title = "Naive Bayes", subtitle = "Linear (w/ independence)") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        legend.position = "none")

p13_combined <- grid.arrange(
  p13_lda, p13_qda, p13_nb, ncol = 3,
  top = grid::textGrob("Decision Boundary Comparison",
                       gp = grid::gpar(fontface = "bold", fontsize = 14))
)
ggsave("images/13_decision_boundary_comparison.png", p13_combined, width = 12, height = 4.5, dpi = 150)

# ============================================================================
# Plot 14: Student Height/Weight Classification (from slides)
# ============================================================================
set.seed(123)

# Simulate student data similar to slides
n_male <- 38
n_female <- 19

males <- data.frame(
  height = rnorm(n_male, mean = 1.80, sd = 0.07),
  weight = rnorm(n_male, mean = 78, sd = 10),
  sex = "Male"
)

females <- data.frame(
  height = rnorm(n_female, mean = 1.65, sd = 0.06),
  weight = rnorm(n_female, mean = 60, sd = 8),
  sex = "Female"
)

students <- rbind(males, females)
students$sex <- factor(students$sex)

# Fit LDA
lda_students <- lda(sex ~ height + weight, data = students)

# Create prediction grid
h_range <- seq(1.50, 1.95, length.out = 100)
w_range <- seq(45, 100, length.out = 100)
grid_students <- expand.grid(height = h_range, weight = w_range)
grid_students$pred <- predict(lda_students, grid_students)$class

p14 <- ggplot() +
  geom_tile(data = grid_students, aes(x = height, y = weight, fill = pred), alpha = 0.2) +
  geom_point(data = students, aes(x = height, y = weight, color = sex, shape = sex), size = 3) +
  scale_fill_manual(values = c("coral", "steelblue"), guide = "none") +
  scale_color_manual(values = c("darkred", "darkblue")) +
  scale_shape_manual(values = c(16, 17)) +
  labs(title = "LDA Classification: Student Sex by Height and Weight",
       subtitle = "Linear decision boundary separates males from females",
       x = "Height (m)",
       y = "Weight (kg)",
       color = "Sex", shape = "Sex") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "bottom")

ggsave("images/14_student_classification.png", p14, width = 8, height = 6, dpi = 150)

# ============================================================================
# Print completion message
# ============================================================================
cat("\n")
cat("============================================================\n")
cat("All plots have been generated and saved to the 'images/' folder!\n")
cat("============================================================\n")
cat("\nGenerated files:\n")
list.files("images", pattern = "\\.png$")

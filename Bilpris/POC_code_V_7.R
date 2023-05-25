library(tidyverse)
library(readxl)
library(caret)
library(boot)
library(glmnet)

# Del 1 -------------------------------------------------------------------


# Data preparation --------------------------------------------------------

# Loading data
data <- read_excel("E:/ecutbildning/malmo/R/kunskapskontroll/BlocketBilData.xlsx", sheet = "raw_data")

# Head of data
head(data)

# Feature names
names(data)

# Dimensions
dim(data)

# Data structure
str(data)

# Summary of data
summary(data)

# Unique values for the categorical variables
unique(data$Biltyp)
unique(data$Växellåda)
unique(data$Bränsle)
unique(data$Miltal)
# Checking the number of categories.
length(unique(data$Miltal))

# Some hybrid cars was collected by mistake. So we remove those cars.
data <- data[data$Bränsle %in% c("Bensin", "Diesel"), ]
dim(data)
sum(is.na(data))

# Making sure we now only have two chategories in "Bränsle"
unique(data$Bränsle)

# Transforming "Miltal" category to numerical by calculating the middle value for each car.
data <- data %>%
  separate(Miltal, into = c("Lägsta_miltal", "Högsta_miltal"), sep = " - ") %>%
  mutate(
    Lägsta_miltal = str_replace_all(Lägsta_miltal, " ", ""),
    Högsta_miltal = str_replace_all(Högsta_miltal, " ", ""),
    Miltal_mitten = round((as.numeric(Lägsta_miltal) + as.numeric(Högsta_miltal)) / 2)
  ) %>%
  select(-Lägsta_miltal, -Högsta_miltal)

# Warning received regarding generated NA values. This due to one of the mileage chategory is: "över - 50000"
# checking for Na values.
sum(is.na(data))

# There where only 2, so we remove those cars.
data <- na.omit(data)

# Making sure we do not have any missing values.
sum(is.na(data))


# Checking the data after data cleaning -----------------------------------

# The structure
str(data)

# Columns
names(data)

# Unique values
unique(data$Biltyp)
unique(data$Växellåda)
unique(data$Bränsle)
unique(data$Miltal_mitten)

# Train/test split (stratified on "Modellår") -----------------------------
# To get a more fair evaluation on how the model will be performing on new data.
# We choose to stratify the data on "Modellår".To ensure that both test and train 
# data has the same proportions of "Modellår".

# Stratified train/test split with "createDataPartition" (caret)

set.seed(42)

X_train_strat_index <- createDataPartition(data$Modellår, p = 0.8, list = FALSE)
X_train_strat <- data[X_train_strat_index, ]
X_test_strat <- data[-X_train_strat_index, ]
y_train_strat <- X_train_strat[, 5]
y_test_strat <- X_test_strat$Pris
X_train_strat
X_test_strat
# "null" model (lm) to check if the data can be used ----------------------

model_lm <- lm(Pris ~ ., data = data, subset = X_train_strat_index)
summary(model_lm)
predict_strat_train <- predict(model_lm, newdata = X_train_strat, type = "response")
rmse_train <- sqrt(sum((predict_strat_train -y_train_strat )^2)/nrow(X_train_strat_index))
rmse_train
# CV on train data --------------------------------------------------------
# To be able to use CV, we need to use a "glm" (not lm) 
# The model is still a "null" model.

# Note: using "glm" will NOT produce exactly the same summary as for "lm"
# glm uses AIC instead of F-statistics, p-value
# The two summary's are used in the "conclusion" part (see below)

set.seed(42) 

model_glm <- glm(Pris ~ ., data = data, subset = X_train_strat_index)
set.seed(42)
cv.err2 <- cv.glm(X_train_strat, model_glm, K = 10)
cv.err2$delta[1]

cv_rmse_strat_train <- sqrt(cv.err2$delta[1])
cv_rmse_strat_train


# Prediction on the full training data ------------------------------------

predict_strat_train <- predict(model_glm, newdata = X_train_strat, type = "response")

residuals_strat_train <- predict_strat_train - y_train_strat

squared_error_strat_train <- residuals_strat_train^2

mse_strat_train <- sum(squared_error_strat_train)/nrow(squared_error_strat_train)

rmse_strat_train <- sqrt(mse_strat_train)
rmse_strat_train


# Results -----------------------------------------------------------------

rmse_data <- tibble(Data_set = c("Train", "CV"),
                    RMSE = c(rmse_strat_train, cv_rmse_strat_train))
rmse_data

# Conclusion --------------------------------------------------------------

# Note: both models are "null" models
summary(model_lm)
summary(model_glm)

# Using the commonly used threshold for P-value: 0.05
# It seems that see that some variables seems to have a effect the Price.
# This also applies to the full model.



# Individual POC part -----------------------------------------------------

# Perform one-hot encoding on categorical variables
encoded_train_strat <- model.matrix(~ Bränsle + Växellåda + Modellår + Biltyp + Miltal_mitten - 1, data = X_train_strat)
encoded_test_strat <- model.matrix(~ Bränsle + Växellåda + Modellår + Biltyp + Miltal_mitten - 1, data = X_test_strat)

# Extract response variables for training and testing
y_train_strat <- X_train_strat$Pris
y_test_strat <- X_test_strat$Pris

# Standardize the predictors
scaled_train_strat <- scale(encoded_train_strat)
scaled_test_strat <- scale(encoded_test_strat)

# Perform Lasso regression
lasso_model <- cv.glmnet(scaled_train_strat, y_train_strat, alpha = 1)

# Perform Ridge regression
ridge_model <- cv.glmnet(scaled_train_strat, y_train_strat, alpha = 0)

# Predict using the Lasso and Ridge models
lasso_predictions <- predict(lasso_model, newx = scaled_test_strat)
ridge_predictions <- predict(ridge_model, newx = scaled_test_strat)

# Calculate the mean squared error (MSE) for Lasso and Ridge predictions
lasso_mse <- sqrt(mean((y_test_strat - lasso_predictions)^2))
ridge_mse <- sqrt(mean((y_test_strat - ridge_predictions)^2))

# Print the MSE values
cat("lm RMSE:", rmse_train , "\n")
cat("glm RMSE:", rmse_strat_train , "\n")
cat("Lasso RMSE:", lasso_mse, "\n")
cat("Ridge RMSE:", ridge_mse, "\n")

# #diagnostic plots -------------------------------------------------------
#diagnostic plots
par(mfrow = c(2,2))
plot(lasso_model)
plot(ridge_model)

# Del 2 --------------------------------------------------------------------

# EDA ---------------------------------------------------------------------

library(ggplot2)
library(readxl)
library(caret)
library(tidyverse)
library(tidyr)
library(dplyr)
library(stringr)
# loading data
data <- read_excel("E:/ecutbildning/malmo/R/kunskapskontroll/BlocketBilData.xlsx",sheet = "raw_data")
# away with the hybrid!
data <- data[data$Bränsle!= "Miljöbränsle/Hybrid", ]
data <-na.omit(data)
sum(is.na(data))

# Create the multi-boxplot
ggplot(data, aes(x = Bränsle, y = Pris, fill = Växellåda)) +
  geom_boxplot() +
  facet_wrap(~ Biltyp) +
  xlab("Bränsle") +
  ylab("Pris") +
  ggtitle("Multi-Boxplot: Pris vs Bränsle, Växellåda, and Biltyp") +
  scale_fill_discrete(name = "Växellåda")+
  scale_fill_manual(values = c("green", "blue", "red"))
head(data)
# Perform one-hot encoding
data$Bränsle<- model.matrix(~ Bränsle- 1, data = data)
data$Växellåda <-model.matrix(~ Växellåda - 1, data = data)
data$Biltyp <- model.matrix(~ Biltyp - 1, data = data)
data
# Transforming "Miltal" category to numerical by calculating the middle value for each car.
data <- data %>%
  separate(Miltal, into = c("Lägsta_miltal", "Högsta_miltal"), sep = " - ") %>%
  mutate(
    Lägsta_miltal = str_replace_all(Lägsta_miltal, " ", ""),
    Högsta_miltal = str_replace_all(Högsta_miltal, " ", ""),
    Miltal_mitten = round((as.numeric(Lägsta_miltal) + as.numeric(Högsta_miltal)) / 2)
  ) %>%
  select(-Lägsta_miltal, -Högsta_miltal)
sum(is.na(data))
# There where only 2, so we remove those cars.
data <- na.omit(data)
sum(is.na(data))
dim(data)
summary(data)
# only to numeric!!!

correlation_matrix0 <- cor(data)
correlation_matrix0
heatmap(correlation_matrix0)

#plot
ggplot(data, aes(x = sort(Miltal_mitten), y = Pris, color = Modellår)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 90, hjust = 10))

hist1 <-ggplot(data, aes(x = Pris)) +
  geom_histogram(binwidth = 10000, fill = "blue", color = "black") +
  labs(x = "Price", y = "Frequency") +
  theme_minimal()
hist2 <-ggplot(data, aes(x = Miltal_mitten)) +
  geom_histogram(binwidth = 5000, fill = "blue", color = "black") +
  labs(x = "Miltal_mitten", y = "Frequency") +
  theme_minimal()
hist3 <-ggplot(data, aes(x = Modellår)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  labs(x = "Modellår", y = "Frequency") +
  theme_minimal()

plot(hist1)
plot(hist2)
plot(hist3)

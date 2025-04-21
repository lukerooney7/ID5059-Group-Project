train <- read.csv("train.csv")

# 1. Group ptype values
train$ptype_grouped <- dplyr::case_when(
  train$ptype == 0 ~ "none",
  train$ptype == 1 ~ "rain",
  TRUE ~ "other"
)

# 2. One-hot encode using fastDummies
library(fastDummies)

train <- fastDummies::dummy_cols(
  train,
  select_columns = "ptype_grouped",
  remove_first_dummy = FALSE,
  remove_selected_columns = FALSE
)

train$id <- NULL

install.packages("lubridate")

library(lubridate)

# Convert 'valid_time' to datetime format
train$valid_time <- ymd_hms(train$valid_time)

# Extract hour, month, and day of week
train$hour <- hour(train$valid_time)
train$month <- month(train$valid_time)
train$dayofweek <- wday(train$valid_time, label = FALSE) - 1  # 0 = Monday like Python

# Fix Sunday wraparound
train$dayofweek[train$dayofweek == -1] <- 6


# Calculate wind speed at 10m and 100m
train$wind10_speed  <- sqrt(train$u10^2 + train$v10^2)
train$wind100_speed <- sqrt(train$u100^2 + train$v100^2)



# Wind direction in degrees (0 = north, 90 = east, etc.)
train$wind10_dir  <- (atan2(train$u10,  train$v10)  * 180 / pi) %% 360
train$wind100_dir <- (atan2(train$u100, train$v100) * 180 / pi) %% 360

# Convert degrees to radians
train$wind10_dir_rad  <- train$wind10_dir  * pi / 180
train$wind100_dir_rad <- train$wind100_dir * pi / 180

# Sine and cosine encoding
train$wind10_dir_sin  <- sin(train$wind10_dir_rad)
train$wind10_dir_cos  <- cos(train$wind10_dir_rad)

train$wind100_dir_sin <- sin(train$wind100_dir_rad)
train$wind100_dir_cos <- cos(train$wind100_dir_rad)



# List of features to scale
features_to_scale <- c(
  "tp", "sp", "wind10_speed", "wind100_speed",
  "hour", "month", "dayofweek"
)

# Create a copy to preserve the original
train_scaled <- train

# Apply standard scaling (mean = 0, sd = 1)
train_scaled[features_to_scale] <- scale(train_scaled[features_to_scale])


# Encode hour as cyclical
train_scaled$hour_sin  <- sin(2 * pi * train_scaled$hour / 24)
train_scaled$hour_cos  <- cos(2 * pi * train_scaled$hour / 24)

# Encode month as cyclical
train_scaled$month_sin <- sin(2 * pi * train_scaled$month / 12)
train_scaled$month_cos <- cos(2 * pi * train_scaled$month / 12)

# Encode day of week as cyclical (0 = Monday, 6 = Sunday)
train_scaled$dow_sin   <- sin(2 * pi * train_scaled$dayofweek / 7)
train_scaled$dow_cos   <- cos(2 * pi * train_scaled$dayofweek / 7)



# Define the target variable
y <- train_scaled$t2m_C

# Define input features
feature_cols <- c(
  "tp", "sp", "wind10_speed", "wind100_speed",
  "hour_sin", "hour_cos",
  "month_sin", "month_cos",
  "dow_sin", "dow_cos",
  "wind10_dir_sin", "wind10_dir_cos",
  "ptype_grouped_none", "ptype_grouped_rain", "ptype_grouped_other"
)

# Create the feature matrix
X <- train_scaled[, feature_cols]

# Print the shape
cat("Features and target selected.\n")
cat("Feature matrix dimensions:", nrow(X), "rows ×", ncol(X), "columns\n")


# Define features to drop
features_to_drop <- c("dow_sin", "dow_cos")

# Drop the features
cleaned_df <- train_scaled[, !(names(train_scaled) %in% features_to_drop)]

# Display first 5 rows
cat("Preview of cleaned dataset:\n")
print(head(cleaned_df, 5))

# Save first 5 rows to CSV
head_df <- head(cleaned_df, 5)
write.csv(head_df, "cleaned_dataset_head.csv", row.names = FALSE)

cat("Saved 'cleaned_dataset_head.csv'\n")

#  Load packages
install.packages("ranger")
install.packages("dplyr")
library(ranger)
library(dplyr)


set.seed(99)
train_sample <- train[sample(nrow(train), size = 0.04 * nrow(train)), ]


set.seed(99)
train_index <- sample(seq_len(nrow(train_sample)), size = 0.8 * nrow(train_sample))

train_subset <- train_sample[train_index, ]
val_subset   <- train_sample[-train_index, ]



rf_model <- ranger(
  formula = t2m ~ .,
  data = train_subset,
  num.trees = 300,
  importance = "impurity"
)


y_val <- val_subset$t2m
X_val <- val_subset %>% select(-t2m)

y_pred <- predict(rf_model, data = X_val)$predictions

rmse <- sqrt(mean((y_val - y_pred)^2))
mae  <- mean(abs(y_val - y_pred))

cat(" Random Forest (5% Sample):\n")
cat("RMSE:", round(rmse, 3), "\n")
cat("MAE :", round(mae, 3), "\n")

rf_model_1 <- ranger(
  formula = t2m ~ .,
  data = train_subset,
  num.trees = 500,
  mtry = 5,
  min.node.size = 5,
  importance = "impurity"
)

y_pred_1 <- predict(rf_model_1, data = X_val)$predictions
rmse_1 <- sqrt(mean((y_val - y_pred_1)^2))
mae_1  <- mean(abs(y_val - y_pred_1))

cat("Combo 1 - mtry=5, min.node.size=5:\n")
cat("RMSE:", round(rmse_1, 3), "\n")
cat("MAE :", round(mae_1, 3), "\n")


rf_model_2 <- ranger(
  formula = t2m ~ .,
  data = train_subset,
  num.trees = 500,
  mtry = 7,
  min.node.size = 10,
  importance = "impurity"
)

y_pred_2 <- predict(rf_model_2, data = X_val)$predictions
rmse_2 <- sqrt(mean((y_val - y_pred_2)^2))
mae_2  <- mean(abs(y_val - y_pred_2))

cat("Combo 2 - mtry=7, min.node.size=10:\n")
cat("RMSE:", round(rmse_2, 3), "\n")
cat("MAE :", round(mae_2, 3), "\n")


rf_model_3 <- ranger(
  formula = t2m ~ .,
  data = train_subset,
  num.trees = 500,
  mtry = 9,
  min.node.size = 3,
  importance = "impurity"
)

y_pred_3 <- predict(rf_model_3, data = X_val)$predictions
rmse_3 <- sqrt(mean((y_val - y_pred_3)^2))
mae_3  <- mean(abs(y_val - y_pred_3))

cat("Combo 3 - mtry=9, min.node.size=3:\n")
cat("RMSE:", round(rmse_3, 3), "\n")
cat("MAE :", round(mae_3, 3), "\n")






set.seed(99)
train_sample <- train_scaled[sample(nrow(train_scaled), size = 0.04 * nrow(train_scaled)), ]

train_index <- sample(seq_len(nrow(train_sample)), size = 0.8 * nrow(train_sample))
train_subset <- train_sample[train_index, ]
val_subset   <- train_sample[-train_index, ]


feature_cols <- c(
  "tp", "sp", "wind10_speed", "wind100_speed",
  "hour_sin", "hour_cos",
  "month_sin", "month_cos",
  "dow_sin", "dow_cos",
  "wind10_dir_sin", "wind10_dir_cos",
  "ptype_grouped_none", "ptype_grouped_rain", "ptype_grouped_other"
)

X_train <- train_subset[, feature_cols]
X_val   <- val_subset[, feature_cols]

y_train <- train_subset$t2m
y_val   <- val_subset$t2m


X_train_scaled <- scale(as.matrix(X_train))
X_val_scaled <- scale(
  as.matrix(X_val),
  center = attr(X_train_scaled, "scaled:center"),
  scale  = attr(X_train_scaled, "scaled:scale")
)


library(glmnet)

sgd_model <- glmnet(
  x = X_train_scaled,
  y = y_train,
  alpha = 0,       # Ridge-style (same as default SGDRegressor)
  lambda = 0,      # No regularization
  standardize = FALSE
)


y_pred <- predict(sgd_model, newx = X_val_scaled)

rmse <- sqrt(mean((y_val - y_pred)^2))
mae  <- mean(abs(y_val - y_pred))

cat("SGD Linear Model Evaluation:\n")
cat("RMSE:", round(rmse, 3), "\n")
cat("MAE :", round(mae, 3), "\n")


install.packages("xgboost")
library(xgboost)


dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dval   <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)


library(xgboost)

set.seed(99)

xgb_model <- xgboost(
  data = dtrain,
  objective = "reg:squarederror",
  eval_metric = "rmse",
  nrounds = 100,
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8,
  verbose = 0
)


y_pred <- predict(xgb_model, newdata = dval)

rmse <- sqrt(mean((y_val - y_pred)^2))
mae  <- mean(abs(y_val - y_pred))

cat("XGBoost Model Evaluation:\n")
cat("RMSE:", round(rmse, 3), "\n")
cat("MAE :", round(mae, 3), "\n")



# Define tuning grid
params_grid <- expand.grid(
  max_depth = c(4, 6),
  eta = c(0.1, 0.05),
  subsample = c(0.8, 1.0)
)

# Store results
results <- data.frame()

# Loop over combinations
for (i in 1:nrow(params_grid)) {
  cat("Fitting combo", i, "of", nrow(params_grid), "...\n")
  
  param <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    max_depth = params_grid$max_depth[i],
    eta = params_grid$eta[i],
    subsample = params_grid$subsample[i],
    colsample_bytree = 0.8
  )
  
  set.seed(99)
  xgb_tuned <- xgb.train(
    params = param,
    data = dtrain,
    nrounds = 100,
    verbose = 0
  )
  
  y_pred <- predict(xgb_tuned, newdata = dval)
  rmse <- sqrt(mean((y_val - y_pred)^2))
  mae  <- mean(abs(y_val - y_pred))
  
  results <- rbind(results, data.frame(
    max_depth = param$max_depth,
    eta = param$eta,
    subsample = param$subsample,
    rmse = rmse,
    mae = mae
  ))
}

# Print sorted results
cat("\n Tuning Results (Sorted by RMSE):\n")
print(results[order(results$rmse), ])



best_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 6,
  eta = 0.1,
  subsample = 1.0,
  colsample_bytree = 0.8
)

set.seed(99)
xgb_best <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = 300,
  verbose = 0
)

y_pred_best <- predict(xgb_best, newdata = dval)

rmse_best <- sqrt(mean((y_val - y_pred_best)^2))
mae_best  <- mean(abs(y_val - y_pred_best))

cat("Final XGBoost Model (300 rounds):\n")
cat("RMSE:", round(rmse_best, 3), "\n")
cat("MAE :", round(mae_best, 3), "\n")



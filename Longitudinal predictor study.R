################################################################################
#### Name           Longitudinal predictor study
#### Author         LM de Groot
#### Date           June 10, 2025
#### Description    Data imputation, longituindal parameter computation,
####                analyses (Random Forest, Lasso regression, and logistic 
####                regression), and sensitivity analyses
################################################################################
################################################################################
rm(list = ls())

library(dplyr)
library(mice)
library(haven)
library(caret)
library(labelled)

random <- 2011
set.seed(random)

################################################################################
#                          CREATE LIST AND DF FOR RESULTS                      #
################################################################################

# --------------------------
# 1. Results: AUC, sensi, speci, ppv, npv, prevalence
# --------------------------

results <- data.frame(
  run = integer(),
  
  # random forest
  auc_train_rf = numeric(),
  auc_test_rf = numeric(),
  sensi_train_rf = numeric(),
  sensi_test_rf = numeric(),
  speci_train_rf = numeric(),
  speci_test_rf = numeric(),
  ppv_train_rf = numeric(),
  ppv_test_rf = numeric(),
  npv_train_rf = numeric(),
  npv_test_rf = numeric(),
  prevalence_train_rf = numeric(),
  prevalence_test_rf = numeric(),
  
  
  # lasso
  auc_train_lasso = numeric(),
  auc_test_lasso = numeric(),
  sensi_train_lasso = numeric(),
  sensi_test_lasso = numeric(),
  speci_train_lasso = numeric(),
  speci_test_lasso = numeric(),
  ppv_train_lasso = numeric(),
  ppv_test_lasso = numeric(),
  npv_train_lasso = numeric(),
  npv_test_lasso = numeric(),
  prevalence_train_lasso = numeric(),
  prevalence_test_lasso = numeric(),
  
  # logistic regression
  auc_train_lr = numeric(),
  auc_test_lr = numeric(),
  sensi_train_lr = numeric(),
  sensi_test_lr = numeric(),
  speci_train_lr = numeric(),
  speci_test_lr = numeric(),
  ppv_train_lr = numeric(),
  ppv_test_lr = numeric(),
  npv_train_lr = numeric(),
  npv_test_lr = numeric(),
  prevalence_train_lr = numeric(),
  prevalence_test_lr = numeric()
)

# --------------------------
# 2. Coefficients and RF-importance
# --------------------------
importance_rf_list <- vector("list", length = 500)  
coef_lasso_list <- vector("list", length = 500)
coef_lr_list <- vector("list", length = 500)

# Helper function
list_to_long_df <- function(lst, all_vars) {
  mat <- matrix(NA, nrow = length(lst), ncol = length(all_vars))
  colnames(mat) <- all_vars
  
  for (i in seq_along(lst)) {
    current <- lst[[i]]
    if (length(current) > 0) {
      matched_vars <- names(current)
      mat[i, matched_vars] <- as.numeric(current)
    }
  }
  
  df <- as.data.frame(mat)
  df$run <- seq_along(lst)
  df <- df[, c("run", all_vars)]
  
  return(df)
}

# --------------------------
# 3. Lists for predicted probabilities
# --------------------------
calibration_rf_train_list <- vector("list", 500)
calibration_rf_test_list <- vector("list", 500)
calibration_lasso_train_list <- vector("list", 500)
calibration_lasso_test_list <- vector("list", 500)
calibration_lr_train_list <- vector("list", 500)
calibration_lr_test_list <- vector("list", 500)


################################################################################
#                                  IMPORT DATA                                 #
################################################################################
setwd("")
dat <- data.frame(read_sav("LASA_preds16.sav")) %>% 
  unlabelled()  
dat$depression <- as.factor(dat$depression)


for (i in 1:500) {
  
  set.seed(random + i)  
  
  ################################################################################
  #### IMPUTATION
  ################################################################################
  
  # --------------------------
  # 1. Data preparation
  # --------------------------
  vars_to_impute <- c(
    "alcohol1", "alcohol2", "alcohol3", "alcohol4", "alcohol5",
    "anxiety1", "anxiety2", "anxiety3", "anxiety4", "anxiety5",
    "bmi1", "bmi2", "bmi3", "bmi4", "bmi5",
    "emosupport_received1", "emosupport_received2", "emosupport_received3", "emosupport_received4", "emosupport_received5",
    "functlimit1", "functlimit2", "functlimit3", "functlimit4", "functlimit5",
    "hobbies1", "hobbies2", "hobbies3", "hobbies4", "hobbies5",
    "ips1", "ips2", "ips3", "ips4", "ips5",
    "loneliness1", "loneliness2", "loneliness3", "loneliness4", "loneliness5",
    "mastery1", "mastery2", "mastery3", "mastery4", "mastery5",
    "medication1", "medication2", "medication3", "medication4", "medication5",
    "memorymax1", "memorymax2", "memorymax3", "memorymax4", "memorymax5",
    "mmse1", "mmse2", "mmse3", "mmse4", "mmse5",
    "networksize1", "networksize2", "networksize3", "networksize4", "networksize5",
    "nochrondis1", "nochrondis2", "nochrondis3", "nochrondis4", "nochrondis5",
    "pain1", "pain2", "pain3", "pain4", "pain5",
    "physact1", "physact2", "physact3", "physact4", "physact5",
    "physperform1", "physperform2", "physperform3", "physperform4", "physperform5",
    "pulse1", "pulse2", "pulse3", "pulse4", "pulse5",
    "satisfaction1", "satisfaction2", "satisfaction3", "satisfaction4", "satisfaction5",
    "selfefficacy1", "selfefficacy2", "selfefficacy3", "selfefficacy4", "selfefficacy5",
    "selfesteem1", "selfesteem2", "selfesteem3", "selfesteem4", "selfesteem5",
    "sleepproblems1", "sleepproblems2", "sleepproblems3", "sleepproblems4", "sleepproblems5",
    "srhfuture1", "srhfuture2", "srhfuture3", "srhfuture4", "srhfuture5",
    "srhpresent1", "srhpresent2", "srhpresent3", "srhpresent4", "srhpresent5",
    "volwork1", "volwork2", "volwork3", "volwork4", "volwork5",
    "waist1", "waist2", "waist3", "waist4", "waist5",
    "workhours1", "workhours2", "workhours3", "workhours4", "workhours5"
  )
  
  predictors_only <- c("firstdepression4", "firstdepression5", "firstdepression6")
  
  # --------------------------
  # 2. Stratified split
  # --------------------------
  
  train_index <- createDataPartition(dat$depression, p = 0.7, list = FALSE)
  train_data <- dat[train_index, ]
  test_data  <- dat[-train_index, ]
  
  # --------------------------
  # 3. Imputation Trainingset
  # --------------------------
  train_sub <- train_data[, c(vars_to_impute, predictors_only)]
  ini <- mice(train_sub, maxit = 0)
  methods <- ini$method
  pred_matrix <- ini$predictorMatrix
  
  pred_matrix[predictors_only, ] <- 0      
  pred_matrix[, predictors_only] <- 1      
  
  imp_train <- mice(train_sub, m = 1, maxit = 1, method = methods, predictorMatrix = pred_matrix, seed = random + i)
  completed_train <- complete(imp_train, 1)
  
  # --------------------------
  # 4. Imputation Testset
  # --------------------------
  test_sub <- test_data[, c(vars_to_impute, predictors_only)]
  imp_test <- mice(test_sub, m = 1, maxit = 1, method = methods, predictorMatrix = pred_matrix, seed = random + i)
  completed_test <- complete(imp_test, 1)
  
  # --------------------------
  # 5. Restore metadata 
  # --------------------------
  completed_train <- bind_cols(
    train_data %>% select(respnr, cohort, depression, depression_time),
    completed_train)
  
  completed_test <- bind_cols(
    test_data %>% select(respnr, cohort, depression, depression_time),
    completed_test)
  
  table(completed_test$depression)
  table(completed_train$depression)
  table(dat$depression)
  
  
  ################################################################################
  #                         LONGITUDINAL PARAMETER COMPUTATION                   #
  ################################################################################
  
  # Load required packages
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(broom)
  
  # --------------------------
  # 1. Define helper function to compute parameters per respondent
  # --------------------------
  calc_longitudinal_params <- function(df) {
    mean_val <- mean(df$value, na.rm = TRUE)
    
    if (sum(!is.na(df$value)) > 1) {
      fit <- lm(value ~ time, data = df)
      slope <- coef(fit)["time"]
      preds <- predict(fit, newdata = df)
      variability <- mean(abs(df$value - preds), na.rm = TRUE)
    } else {
      slope <- NA_real_
      variability <- NA_real_
    }
    
    tibble(mean = mean_val, change = slope, variability = variability)
  }
  
  # --------------------------
  # 2. Main function to calculate longitudinal parameters per dataset
  # --------------------------
  make_longitudinal_params <- function(imputed_data, original_data, predictors, outcome_var = NULL) {
    
    predictor_roots <- unique(gsub("[0-9]+$", "", predictors))
    all_results <- list()
    
    for (pred in predictor_roots) {
      
      # Construct timepoint variable names and check presence
      time_vars <- paste0(pred, 1:5)
      time_vars <- time_vars[time_vars %in% colnames(imputed_data)]
      
      # Reshape to long format and filter based on depression_time
      long_data <- imputed_data %>%
        select(respnr, depression_time, all_of(time_vars)) %>%
        pivot_longer(cols = all_of(time_vars),
                     names_to = "timepoint",
                     values_to = "value") %>%
        mutate(time = as.integer(gsub(pred, "", timepoint))) %>%
        filter(time < depression_time)
      
      # Compute parameters per respondent
      res <- long_data %>%
        group_by(respnr) %>%
        nest() %>%
        mutate(params = map(data, calc_longitudinal_params)) %>%
        unnest(params) %>%
        select(respnr, mean, change, variability)
      
      # Rename columns to include predictor root
      colnames(res)[2:4] <- paste0(pred, c("_mean", "_change", "_variability"))
      
      all_results[[pred]] <- res
    }
    
    # Merge all predictors and attach outcome
    final <- reduce(all_results, full_join, by = "respnr")
    
    if (!is.null(outcome_var)) {
      final <- left_join(final, original_data %>% select(respnr, all_of(outcome_var)), by = "respnr")
    }
    
    return(final)
  }
  
  # --------------------------
  # 3. Apply to training and test datasets
  # --------------------------
  train_final <- make_longitudinal_params(
    imputed_data = completed_train,
    original_data = train_data,
    predictors = vars_to_impute,
    outcome_var = "depression"
  )
  
  test_final <- make_longitudinal_params(
    imputed_data = completed_test,
    original_data = test_data,
    predictors = vars_to_impute,
    outcome_var = "depression"
  )
  
  
  ################################################################################
  #                                ANALYSIS: RANDOM FOREST                       #
  ################################################################################
  library(randomForest)
  library(pROC)
  
  set.seed(random + i)
  
  
  # ----------------------------------
  # 1. Define predictors and formula
  # ----------------------------------
  
  predictor_vars <- c(
    "alcohol_mean", "alcohol_change", "alcohol_variability",
    "anxiety_mean", "anxiety_change", "anxiety_variability", 
    "bmi_mean", "bmi_change", "bmi_variability",
    "emosupport_received_mean", "emosupport_received_change", "emosupport_received_variability",
    "functlimit_mean", "functlimit_change", "functlimit_variability",
    "hobbies_mean", "hobbies_change", "hobbies_variability",
    "ips_mean", "ips_change", "ips_variability",
    "loneliness_mean", "loneliness_change", "loneliness_variability",
    "mastery_mean", "mastery_change", "mastery_variability",
    "medication_mean", "medication_change", "medication_variability",
    "memorymax_mean", "memorymax_change", "memorymax_variability", 
    "mmse_mean", "mmse_change", "mmse_variability",
    "networksize_mean", "networksize_change", "networksize_variability",
    "nochrondis_mean", "nochrondis_change", "nochrondis_variability",
    "pain_mean", "pain_change", "pain_variability",
    "physact_mean", "physact_change", "physact_variability",
    "physperform_mean", "physperform_change", "physperform_variability",
    "pulse_mean", "pulse_change", "pulse_variability", 
    "satisfaction_mean", "satisfaction_change", "satisfaction_variability",
    "selfefficacy_mean", "selfefficacy_change", "selfefficacy_variability",
    "selfesteem_mean", "selfesteem_change", "selfesteem_variability",
    "sleepproblems_mean", "sleepproblems_change", "sleepproblems_variability",
    "srhfuture_mean", "srhfuture_change", "srhfuture_variability",
    "srhpresent_mean", "srhpresent_change", "srhpresent_variability",
    "volwork_mean", "volwork_change", "volwork_variability",
    "waist_mean", "waist_change", "waist_variability",
    "workhours_mean", "workhours_change", "workhours_variability"
  )
  
  # Create formula
  formula <- as.formula(paste("depression ~", paste(predictor_vars, collapse = " + ")))  
  
  # ----------------------------------
  # 2. Train Random Forest model (default settings)
  # ----------------------------------
  
  set.seed(random + i)
  rf_train_default <- randomForest(formula, data = train_final, importance = TRUE)
  
  # ----------------------------------
  # 3. Performance on training set
  # ----------------------------------
  
  # Classification metrics
  pred_train <- predict(rf_train_default, data = train_final)
  cm_train <- confusionMatrix(pred_train, train_final$depression, positive = "1")
  
  sensi_train_rf <- cm_train$byClass["Sensitivity"]
  speci_train_rf <- cm_train$byClass["Specificity"]
  ppv_train_rf   <- cm_train$byClass["Pos Pred Value"]
  npv_train_rf   <- cm_train$byClass["Neg Pred Value"]
  prevalence_train_rf <- cm_train$byClass["Detection Prevalence"]
  
  # Probabilities
  probs_rf_train <- predict(rf_train_default, data = train_final, type = "prob")[, 2]
  if (length(probs_rf_train) != nrow(train_final)) {
    warning(sprintf("Run %d: RF train mismatch — %d vs %d", i, length(probs_rf_train), nrow(train_final)))
    next
  }
  
  # ROC/AUC
  roc_obj <- roc(response = train_final$depression, predictor = probs_rf_train)
  auc_train_rf <- as.numeric(roc_obj$auc)  
  
  # Calibration 
  train_final$depression_num <- as.numeric(as.character(train_final$depression))
  group_dec_rf_train <- cut(probs_rf_train, breaks = quantile(probs_rf_train, probs = seq(0, 1, 0.1), na.rm = TRUE), include.lowest = TRUE)
  pred_group_rf_train <- tapply(probs_rf_train, group_dec_rf_train, mean)
  obs_group_rf_train  <- tapply(train_final$depression_num, group_dec_rf_train, mean)
  calibration_rf_train_list[[i]] <- data.frame(run = i, 
                                               Decile = 1:10, 
                                               Observed = as.numeric(obs_group_rf_train), 
                                               Predicted = as.numeric(pred_group_rf_train))
  
  # ----------------------------------
  # 4. Performance on test set
  # ----------------------------------
  
  # Classification metrics
  pred_test <- predict(rf_train_default, newdata = test_final)
  cm_test <- confusionMatrix(pred_test, test_final$depression, positive = "1")
  
  sensi_test_rf <- cm_test$byClass["Sensitivity"]
  speci_test_rf <- cm_test$byClass["Specificity"]
  ppv_test_rf   <- cm_test$byClass["Pos Pred Value"]
  npv_test_rf   <- cm_test$byClass["Neg Pred Value"]
  prevalence_test_rf <- cm_test$byClass["Detection Prevalence"]
  
  # Probabilities
  probs_rf_test <- predict(rf_train_default, newdata = test_final, type = "prob")[, 2]
  if (length(probs_rf_test) != nrow(test_final)) {
    warning(sprintf("Run %d: RF test mismatch — %d vs %d", i, length(probs_rf_test), nrow(test_final)))
    next
  }
  
  # ROC/AUC
  roc_obj <- roc(response = test_final$depression, predictor = probs_rf_test)
  auc_test_rf <- as.numeric(roc_obj$auc)
  
  # Calibration 
  test_final$depression_num <- as.numeric(as.character(test_final$depression))
  group_dec_rf_test <- cut(probs_rf_test, breaks = quantile(probs_rf_test, probs = seq(0, 1, 0.1), na.rm = TRUE), include.lowest = TRUE)
  pred_group_rf_test <- tapply(probs_rf_test, group_dec_rf_test, mean)
  obs_group_rf_test  <- tapply(test_final$depression_num, group_dec_rf_test, mean)
  calibration_rf_test_list[[i]] <- data.frame(run = i, 
                                              Decile = 1:10, 
                                              Observed = as.numeric(obs_group_rf_test), 
                                              Predicted = as.numeric(pred_group_rf_test))
  
  
  # ----------------------------------
  # 5. Variable Importance
  # ----------------------------------
  
  var_importance <- data.frame(importance(rf_train_default))
  var_importance$variable <- rownames(var_importance)
  colnames(var_importance)[colnames(var_importance) == "MeanDecreaseAccuracy"] <- "importance"
  importance_rf_list[[i]] <- setNames(var_importance$importance, var_importance$variable)
  
  
  
  ################################################################################
  #                             ANALYSIS: LASSO REGRESSION                       #
  ################################################################################
  library(glmnet)
  library(caret)
  set.seed(random + i)
  
  # ----------------------------------
  # 1. Prepare matrix
  # ----------------------------------
  X_train <- model.matrix(depression ~ ., data = train_final[, c(predictor_vars, "depression")])[, -1]
  y <- as.numeric(as.character(train_final$depression))
  
  # ----------------------------------
  # 2. Fit Lasso, with cross-validation for optimal Lambda
  # ----------------------------------
  set.seed(random + i)
  fit_lasso <- glmnet(x = X_train, y = y, alpha = 1, family = "binomial", nlambda = 100)
  cv.lasso <- cv.glmnet(x = X_train, y = y, alpha = 1, family = "binomial", nfolds = 10)
  optimal_lambda <- cv.lasso$lambda.min
  
  # Coefficients at optimal lambda
  lasso_coeffs <- coef(fit_lasso, s = optimal_lambda)
  selected_indices <- which(lasso_coeffs != 0)
  selected_variables <- rownames(lasso_coeffs)[selected_indices]
  selected_coeffs <- as.numeric(lasso_coeffs[selected_indices])
  
  # Remove intercept
  if ("(Intercept)" %in% selected_variables) {
    intercept_index <- which(selected_variables == "(Intercept)")
    selected_variables <- selected_variables[-intercept_index]
    selected_coeffs <- selected_coeffs[-intercept_index]
  }
  
  # Output selected variables
  lasso_results <- data.frame(
    Variable = selected_variables,
    Coefficient = selected_coeffs
  )
  print(lasso_results)
  
  if (length(selected_variables) > 0) {
    coef_lasso_list[[i]] <- setNames(selected_coeffs, selected_variables)
  } else {
    coef_lasso_list[[i]] <- numeric(0)  
  }
  
  # Fit final model with selected variables
  formula_lasso <- as.formula(paste("depression ~", paste(selected_variables, collapse = " + ")))
  fit_lasso_train <- glm(formula_lasso, data = train_final, family = binomial)
  
  
  # ----------------------------------
  # 3. Performance on Training set
  # ----------------------------------
  
  # Classification metrics
  predicted_train <- ifelse(probs_lasso_train > 0.5, 1, 0)
  predicted_train <- factor(predicted_train, levels = c(0, 1))
  actual_train <- factor(train_final$depression, levels = c(0, 1))
  cm_train <- confusionMatrix(predicted_train, actual_train, positive = "1")
  sensi_train_lasso <- cm_train$byClass["Sensitivity"]
  speci_train_lasso <- cm_train$byClass["Specificity"]
  ppv_train_lasso   <- cm_train$byClass["Pos Pred Value"]
  npv_train_lasso   <- cm_train$byClass["Neg Pred Value"]
  prevalence_train_lasso <- cm_train$byClass["Detection Prevalence"]
  
  # Probabilities
  probs_lasso_train <- predict(fit_lasso_train, type = "response")
  if (length(probs_lasso_train) != nrow(train_final)) {
    warning(sprintf("Run %d: Lasso train mismatch — %d preds vs %d rows", i, length(probs_lasso_train), nrow(train_final)))
    next
  }
  
  # AUC/ROC
  roc_obj <- roc(response = train_final$depression, predictor = probs_lasso_train)
  auc_train_lasso <- as.numeric(roc_obj$auc)
  
  # Calibration
  train_final$depression_num <- as.numeric(as.character(train_final$depression))
  group_dec_lasso_train <- cut(probs_lasso_train, breaks = quantile(probs_lasso_train, probs = seq(0, 1, 0.1), na.rm = TRUE), include.lowest = TRUE)
  pred_group_lasso_train <- tapply(probs_lasso_train, group_dec_lasso_train, mean)
  obs_group_lasso_train <- tapply(train_final$depression_num, group_dec_lasso_train, mean)
    calibration_lasso_train_list[[i]] <- data.frame(run = i,
                                                  Decile = 1:10,
                                                  Observed = as.numeric(obs_group_lasso_train),
                                                  Predicted = as.numeric(pred_group_lasso_train))
  
  # ----------------------------------
  # 3. Performance on Test set
  # ----------------------------------
  
  # Apply model to test set
  X_test <- model.matrix(depression ~ ., data = test_final[, c(predictor_vars, "depression")])[, -1]
  
  # Classification metrics
  predicted_test <- ifelse(probs_lasso_test > 0.5, 1, 0)
  predicted_test <- factor(predicted_test, levels = c(0, 1))
  actual_test <- factor(test_final$depression, levels = c(0, 1))
  cm_test <- confusionMatrix(predicted_test, actual_test, positive = "1")
  sensi_test_lasso <- cm_test$byClass["Sensitivity"]
  speci_test_lasso <- cm_test$byClass["Specificity"]
  ppv_test_lasso   <- cm_test$byClass["Pos Pred Value"]
  npv_test_lasso   <- cm_test$byClass["Neg Pred Value"]
  prevalence_test_lasso <- cm_test$byClass["Detection Prevalence"]
  
  # Probabilities
  probs_lasso_test <- predict(fit_lasso, newx = X_test, s = optimal_lambda, type = "response")[,1]
  if (length(probs_lasso_test) != nrow(test_final)) {
    warning(sprintf("Run %d: Lasso test mismatch — %d preds vs %d rows", i, length(probs_lasso_test), nrow(test_final)))
    next
  }
  
  # AUC
  roc_obj <- roc(response = test_final$depression, predictor = probs_lasso_test)
  auc_test_lasso <- as.numeric(roc_obj$auc)
  
  # Calibration
  test_final$depression_num <- as.numeric(as.character(test_final$depression))
  group_dec_lasso_test <- cut(probs_lasso_test, breaks = quantile(probs_lasso_test, probs = seq(0, 1, 0.1), na.rm = TRUE), include.lowest = TRUE)
  pred_group_lasso_test <- tapply(probs_lasso_test, group_dec_lasso_test, mean)
  obs_group_lasso_test <- tapply(test_final$depression_num, group_dec_lasso_test, mean)
  calibration_lasso_test_list[[i]] <- data.frame(run = i,
                                                 Decile = 1:10,
                                                 Observed = as.numeric(obs_group_lasso_test),
                                                 Predicted = as.numeric(pred_group_lasso_test))
  

  ################################################################################  
  #                            ANALYSIS: LOGISTIC REGRESSION                     #
  ################################################################################
  library(psfmi)
  
  # ----------------------------------
  # 1. Backward selection on Train set
  # ----------------------------------
  bw_train <- glm_bw(formula, data=train_final, p.crit = 0.05000, model_type = "binomial")
  # Results
  step_names <- names(bw_train$RR_model_final)
  step_numbers <- as.integer(gsub("Step ", "", step_names))
  final_step_num <- max(step_numbers, na.rm = TRUE)
  final_step_name <- paste0("Step ", final_step_num)
  round(bw_train$RR_model_final[[final_step_name]], 5)
  formula_final_bw_train <- bw_train$formula_final[[final_step_name]]
  
  # ----------------------------------
  # 2. Performance on Train set
  # ----------------------------------
  
  # Fit
   fit_bw_training <- glm(formula_final_bw_train, x = TRUE, y = TRUE, data = train_final, family = binomial)
  
  # Probabilities
  probs_lr_train <- predict(fit_bw_training, type = "response")
    if (length(probs_lr_train) != nrow(train_final)) {
    warning(sprintf("Run %d: LR train mismatch — %d preds vs %d rows", i, length(probs_lr_train), nrow(train_final)))
    
    coef_lr_list[[i]] <- NA
    auc_train_lr <- NA_real_
    sensi_train_lr <- NA_real_
    speci_train_lr <- NA_real_
    ppv_train_lr <- NA_real_
    npv_train_lr <- NA_real_
    prevalence_train_lr <- NA_real_
    calibration_lr_train_list[[i]] <- data.frame(run = i, Decile = 1:10, Observed = NA_real_, Predicted = NA_real_)
    
  } else {
    
    # Select predictors and save coefficients
    selected_vars <- bw_train$predictors_final
    selected_coefs <- coef(fit_bw_training)[selected_vars]
    coef_lr_list[[i]] <- setNames(as.numeric(selected_coefs), names(selected_coefs))
    
    # AUC 
    roc_obj <- roc(response = train_final$depression, predictor = probs_lr_train)
    auc_train_lr <- as.numeric(roc_obj$auc)
    
    # Calibration
    train_final$depression_num <- as.numeric(as.character(train_final$depression))
    group_dec_lr_train <- cut(probs_lr_train, breaks = quantile(probs_lr_train, probs = seq(0, 1, 0.1), na.rm = TRUE), include.lowest = TRUE)
    obs_group_lr_train <- tapply(train_final$depression_num, group_dec_lr_train, mean)
    pred_group_lr_train <- tapply(probs_lr_train, group_dec_lr_train, mean)
    calibration_lr_train_list[[i]] <- data.frame(run = i, 
                                                 Decile = 1:10, 
                                                 Observed = as.numeric(obs_group_lr_train), 
                                                 Predicted = as.numeric(pred_group_lr_train))
    
    # Classification metrics
    predicted_train <- ifelse(probs_lr_train > 0.5, 1, 0)
    predicted_train <- factor(predicted_train, levels = c(0, 1))
    actual_train <- factor(train_final$depression, levels = c(0, 1))
    cm_train <- confusionMatrix(predicted_train, actual_train, positive = "1")
    sensi_train_lr <- cm_train$byClass["Sensitivity"]
    speci_train_lr <- cm_train$byClass["Specificity"]
    ppv_train_lr   <- cm_train$byClass["Pos Pred Value"]
    npv_train_lr   <- cm_train$byClass["Neg Pred Value"]
    prevalence_train_lr <- cm_train$byClass["Detection Prevalence"]
    
  }
  
  # ----------------------------------
  # 3. Performance on Test set
  # ----------------------------------
  
  # Fit
  fit_bw_test <- glm(formula_final_bw_train, x = TRUE, y = TRUE, data = test_final, family = binomial)
  
  # Probabilities
  probs_lr_test <- predict(fit_bw_test, type = "response")
  
  if (length(probs_lr_test) != nrow(test_final)) {
    warning(sprintf("Run %d: LR test mismatch — %d preds vs %d rows", i, length(probs_lr_test), nrow(test_final)))
    
    auc_test_lr <- NA_real_
    sensi_test_lr <- NA_real_
    speci_test_lr <- NA_real_
    ppv_test_lr <- NA_real_
    npv_test_lr <- NA_real_
    prevalence_test_lr <- NA_real_
    calibration_lr_test_list[[i]] <- data.frame(run = i, Decile = 1:10, Observed = NA_real_, Predicted = NA_real_)
    
  } else {
    
    # AUC 
    roc_obj <- roc(response = test_final$depression, predictor = probs_lr_test)
    auc_test_lr <- as.numeric(roc_obj$auc)
    
    # Calibration
    test_final$depression_num <- as.numeric(as.character(test_final$depression))
    group_dec_lr_test <- cut(probs_lr_test, breaks = quantile(probs_lr_test, probs = seq(0, 1, 0.1), na.rm = TRUE), include.lowest = TRUE)
    obs_group_lr_test <- tapply(test_final$depression_num, group_dec_lr_test, mean)
    pred_group_lr_test <- tapply(probs_lr_test, group_dec_lr_test, mean)
    calibration_lr_test_list[[i]] <- data.frame(run = i,
                                                Decile = 1:10,
                                                Observed = as.numeric(obs_group_lr_test),
                                                Predicted = as.numeric(pred_group_lr_test))
    
    # Classification metrics
    predicted_test <- ifelse(probs_lr_test > 0.5, 1, 0)
    predicted_test <- factor(predicted_test, levels = c(0, 1))
    actual_test <- factor(test_final$depression, levels = c(0, 1))
    cm_test <- confusionMatrix(predicted_test, actual_test, positive = "1")
    sensi_test_lr <- cm_test$byClass["Sensitivity"]
    speci_test_lr <- cm_test$byClass["Specificity"]
    ppv_test_lr   <- cm_test$byClass["Pos Pred Value"]
    npv_test_lr   <- cm_test$byClass["Neg Pred Value"]
    prevalence_test_lr <- cm_test$byClass["Detection Prevalence"] 
    
  }
  
  ################################################################################  
  #                             ANALYSIS: RETRIEVE RESULTS                       #
  ################################################################################
  
  # ----------------------------------
  # 1. Classification metrics 
  # ----------------------------------
  results_temp <- data.frame(
    run = i,
    
    # random forest
    auc_train_rf = auc_train_rf,
    auc_test_rf = auc_test_rf,
    sensi_train_rf = sensi_train_rf,
    sensi_test_rf = sensi_test_rf,
    speci_train_rf = speci_train_rf,
    speci_test_rf = speci_test_rf,
    ppv_train_rf = ppv_train_rf,
    ppv_test_rf = ppv_test_rf,
    npv_train_rf = npv_train_rf,
    npv_test_rf = npv_test_rf,
    prevalence_train_rf = prevalence_train_rf,
    prevalence_test_rf = prevalence_test_rf,
    
    # lasso
    auc_train_lasso = auc_train_lasso,
    auc_test_lasso = auc_test_lasso,
    sensi_train_lasso = sensi_train_lasso,
    sensi_test_lasso = sensi_test_lasso,
    speci_train_lasso = speci_train_lasso,
    speci_test_lasso = speci_test_lasso,
    ppv_train_lasso = ppv_train_lasso,
    ppv_test_lasso = ppv_test_lasso,
    npv_train_lasso = npv_train_lasso,
    npv_test_lasso = npv_test_lasso,
    prevalence_train_lasso = prevalence_train_lasso,
    prevalence_test_lasso = prevalence_test_lasso,
    
    # logistic regression
    auc_train_lr = auc_train_lr,
    auc_test_lr = auc_test_lr,
    sensi_train_lr = sensi_train_lr,
    sensi_test_lr = sensi_test_lr,
    speci_train_lr = speci_train_lr,
    speci_test_lr = speci_test_lr,
    ppv_train_lr = ppv_train_lr,
    ppv_test_lr = ppv_test_lr,
    npv_train_lr = npv_train_lr,
    npv_test_lr = npv_test_lr,
    prevalence_train_lr = prevalence_train_lr,
    prevalence_test_lr = prevalence_test_lr
  )
  
  rownames(results_temp) <- NULL
  results <- rbind(results, results_temp)
  
  
}    

mean_col <- colMeans(results, na.rm = T)
n <- nrow(results)
sd_col <- apply(results, 2, sd, na.rm = TRUE)
se_col <- sd_col / sqrt(n)
z <- qnorm(0.975)
lower_ci <- mean_col - z * se_col
upper_ci <- mean_col + z * se_col
results_95CI <- data.frame(
  Mean = mean_col,
  Lower_95CI = lower_ci,
  Upper_95CI = upper_ci
)
print(results_95CI)

# ----------------------------------
# 2. Coefficients and importance
# ----------------------------------

all_predictors <- sort(unique(c(
  unlist(lapply(coef_lasso_list, names)),
  unlist(lapply(coef_lr_list, names)),
  unlist(lapply(importance_rf_list, names))
)))

coef_lasso_df    <- list_to_long_df(coef_lasso_list, all_predictors)
coef_lr_df       <- list_to_long_df(coef_lr_list, all_predictors)
importance_rf_df <- list_to_long_df(importance_rf_list, all_predictors)

## Top 20
# Random Forest importance
mda_values <- importance_rf_df[ , !names(importance_rf_df) %in% "run"]
mda_ranks <- t(apply(mda_values, 1, function(x) rank(-x, ties.method = "min")))
mda_rank_df <- cbind(run = importance_rf_df$run, as.data.frame(mda_ranks))
colnames(mda_rank_df)[-1] <- colnames(mda_values)
top20_rf <- mda_rank_df
top20_rf[ , -1] <- as.data.frame(lapply(mda_rank_df[ , -1], function(x) ifelse(x > 20, NA, x)))
sort(colSums(!is.na(top20_rf[ , -1])), decreasing = T)
colMeans(mda_rank_df, na.rm = T)

# Lasso regression coefficients
sort(colSums(!is.na(coef_lasso_df[,-1])), decreasing = T)
colMeans(coef_lasso_df, na.rm = T)
coef_lasso_df$n_selected <- rowSums(!is.na(coef_lasso_df[,2:82]))
coef_lasso_df$n_selected[coef_lasso_df$n_selected == 0] <- NA
summary(coef_lasso_df$n_selected)

# Logistic regression coefficients
sort(colSums(!is.na(coef_lr_df[,-1])), decreasing = T)
colMeans(coef_lr_df, na.rm = T)
coef_lr_df$n_selected <- rowSums(!is.na(coef_lr_df[,2:82]))
coef_lr_df$n_selected[coef_lr_df$n_selected == 0] <- NA
summary(coef_lr_df$n_selected)


# ----------------------------------
# 3. Calibration plots
# ----------------------------------
library(dplyr)
library(tidyr)
library(purrr)

# list w/ named vector to dataframe (column = run, row = respnr) 
list_preds_to_df <- function(preds_list) {
  all_resp <- sort(unique(unlist(lapply(preds_list, names))))
  mat <- matrix(NA, nrow = length(all_resp), ncol = length(preds_list),
                dimnames = list(all_resp, paste0("run_", seq_along(preds_list))))
  
  for (i in seq_along(preds_list)) {
    current <- preds_list[[i]]
    if (length(current) > 0) {
      matched_resp <- names(current)
      mat[matched_resp, i] <- as.numeric(current)
    }
  }
  
  df <- as.data.frame(mat)
  df <- tibble::rownames_to_column(df, var = "respnr")
  return(df)
}

# bind dataframes
calibration_rf_train_df <- do.call(rbind, calibration_rf_train_list)
calibration_rf_test_df <- do.call(rbind, calibration_rf_test_list)
calibration_lasso_train_df <- do.call(rbind, calibration_lasso_train_list)
calibration_lasso_test_df <- do.call(rbind, calibration_lasso_test_list)
calibration_lr_train_df <- do.call(rbind, calibration_lr_train_list)
calibration_lr_test_df <- do.call(rbind, calibration_lr_test_list)

## Calibration plots
# rf train
dat <- calibration_rf_train_df
mean_calibration <- dat %>%
  group_by(Decile) %>%
  summarise(
    mean_observed = mean(Observed),
    mean_predicted = mean(Predicted)
  )
calplot_rf_train <- ggplot(mean_calibration, aes(x = mean_predicted, y = mean_observed)) +
  geom_point() +
  stat_smooth(method = "lm", se = FALSE, formula = y ~ splines::bs(x, 3)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  annotate("text", x = -0.1, y = 1.05, label = "A", fontface = "bold", size = 6) +
  scale_x_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Predicted Probabilities") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Observed Probabilities") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

#rf test
dat <- calibration_rf_test_df
mean_calibration <- dat %>%
  group_by(Decile) %>%
  summarise(
    mean_observed = mean(Observed),
    mean_predicted = mean(Predicted)
  )
calplot_rf_test <- ggplot(mean_calibration, aes(x = mean_predicted, y = mean_observed)) +
  geom_point() +
  stat_smooth(method = "lm", se = FALSE, formula = y ~ splines::bs(x, 3)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  annotate("text", x = -0.1, y = 1.05, label = "B", fontface = "bold", size = 6) +
  scale_x_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Predicted Probabilities") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Observed Probabilities") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# lasso train
dat <- calibration_lasso_train_df
mean_calibration <- dat %>%
  group_by(Decile) %>%
  summarise(
    mean_observed = mean(Observed),
    mean_predicted = mean(Predicted)
  )
calplot_lasso_train <- ggplot(mean_calibration, aes(x = mean_predicted, y = mean_observed)) +
  geom_point() +
  stat_smooth(method = "lm", se = FALSE, formula = y ~ splines::bs(x, 3)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  annotate("text", x = -0.1, y = 1.05, label = "C", fontface = "bold", size = 6) +
  scale_x_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Predicted Probabilities") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Observed Probabilities") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# lasso test
dat <- calibration_lasso_test_df
mean_calibration <- dat %>%
  group_by(Decile) %>%
  summarise(
    mean_observed = mean(Observed),
    mean_predicted = mean(Predicted)
  )
calplot_lasso_test <- ggplot(mean_calibration, aes(x = mean_predicted, y = mean_observed)) +
  geom_point() +
  stat_smooth(method = "lm", se = FALSE, formula = y ~ splines::bs(x, 3)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  annotate("text", x = -0.1, y = 1.05, label = "D", fontface = "bold", size = 6) +
  scale_x_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Predicted Probabilities") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Observed Probabilities") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# lr train
dat <- calibration_lr_train_df
mean_calibration <- dat %>%
  group_by(Decile) %>%
  summarise(
    mean_observed = mean(Observed),
    mean_predicted = mean(Predicted)
  )
calplot_lr_train <- ggplot(mean_calibration, aes(x = mean_predicted, y = mean_observed)) +
  geom_point() +
  stat_smooth(method = "lm", se = FALSE, formula = y ~ splines::bs(x, 3)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  annotate("text", x = -0.1, y = 1.05, label = "E", fontface = "bold", size = 6) +
  scale_x_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Predicted Probabilities") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Observed Probabilities") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

# lr test
dat <- calibration_lr_test_df
mean_calibration <- dat %>%
  group_by(Decile) %>%
  summarise(
    mean_observed = mean(Observed),
    mean_predicted = mean(Predicted)
  )
calplot_lr_test <- ggplot(mean_calibration, aes(x = mean_predicted, y = mean_observed)) +
  geom_point() +
  stat_smooth(method = "lm", se = FALSE, formula = y ~ splines::bs(x, 3)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  annotate("text", x = -0.1, y = 1.05, label = "F", fontface = "bold", size = 6) +
  scale_x_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Predicted Probabilities") +
  scale_y_continuous(limits = c(-0.1, 1.1), breaks = seq(0, 1, 0.1), name = "Observed Probabilities") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())

grid.arrange(calplot_rf_train, calplot_rf_test, calplot_lasso_train, calplot_lasso_test, calplot_lr_train, calplot_lr_test)

# ----------------------------------
# 4. PDPs
# ----------------------------------

## PDPs
par.anxiety_mean <- pdp::partial(object = rf_train_default, pred.var = c('anxiety_mean'), type = 'classification', prob = T, which.class=2)
plot_anxiety_mean <- plotPartial(par.anxiety_mean, ylim=c(0,1.0), ylab = 'Probability', xlab = "Anxiety mean")

par.anxiety_change <- partial(rf_train_default, pred.var = c('anxiety_change'), type = 'classification', prob = T, which.class=2)
plot_anxiety_change <- plotPartial(par.anxiety_change, ylim=c(0,1.0), ylab = 'Probability', xlab = "Anxiety change")

par.pain_mean = partial(rf_train_default, pred.var = c('pain_mean'), type = 'classification', prob = T, which.class=2)
plot_pain_mean <- plotPartial(par.pain_mean, ylim=c(0,1.0), ylab = 'Probability', xlab = 'Pain mean')

par.satisfaction_mean = partial(rf_train_default, pred.var = c('satisfaction_mean'), type = 'classification', prob = T, which.class=2)
plot_satisfaction_mean <- plotPartial(par.satisfaction_mean, ylim=c(0,1.0), ylab = 'Probability', xlab = 'Satisfaction mean')

par.selfesteem_mean = partial(rf_train_default, pred.var = c('selfesteem_mean'), type = 'classification', prob = T, which.class=2)
plot_selfesteem_mean <- plotPartial(par.selfesteem_mean, ylim=c(0,1.0), ylab = 'Probability', xlab = 'Self-esteem mean')

par.srhpresent_mean = partial(rf_train_default, pred.var = c('srhpresent_mean'), type = 'classification', prob = T, which.class=2)
plot_srhpresent_mean <- plotPartial(par.srhpresent_mean, ylim=c(0,1.0), ylab = 'Probability', xlab = 'self-rated health present mean')

par.memory_variability = partial(rf_train_default, pred.var = c('memorymax_variability'), type = 'classification', prob = T, which.class=2)
plot_memory_variability <- plotPartial(par.memory_variability, ylim=c(0,1.0), ylab = 'Probability', xlab = "Episodic memory variability")

par.physact_variability = partial(rf_train_default, pred.var = c('physact_variability'), type = 'classification', prob = T, which.class=2)
plot_physact_variability <- plotPartial(par.physact_variability, ylim=c(0,1.0), ylab = 'Probability', xlab = 'Physical activity variability')

grid.arrange(plot_anxiety_mean, plot_pain_mean, plot_satisfaction_mean, plot_selfesteem_mean, plot_srhpresent_mean,
             plot_anxiety_change,
             plot_memory_variability, plot_physact_variability, nrow =4, ncol =2)


# ----------------------------------
# 5. Sensitivity analyses
# ----------------------------------

## Random Under Sampling - adjustments to code:

  # --------------------------
  # 2. Stratified split
  # --------------------------

  train_index <- createDataPartition(dat$depression, p = 0.7, list = FALSE)
  train_data <- dat[train_index, ]
  test_data  <- dat[-train_index, ]
  train_depressed <- train_data[train_data$depression == 1, ]
  train_not_depressed <- train_data[train_data$depression == 0, ]
  n_depressed <- nrow(train_depressed)
  set.seed(random)
  train_not_depressed_balanced <- train_not_depressed[sample(nrow(train_not_depressed), n_depressed), ]
  train_data_balanced <- rbind(train_depressed, train_not_depressed_balanced)
  train_data_balanced <- train_data_balanced[sample(nrow(train_data_balanced)), ]
  table(train_data_balanced$depression)

  # --------------------------
  # 5. Restore metadata 
  # --------------------------
  completed_train <- complete(imp_train, 1)
  completed_train <- bind_cols(train_data_balanced %>% select(respnr, cohort, depression, depression_time), completed_train)

  completed_test <- bind_cols(
  test_data %>% select(respnr, cohort, depression, depression_time),
  completed_test)


  ## Tuned Random Forest - adjustments to code:
  
    # ----------------------------------
    # 2. Train Random Forest model (default settings)
    # ----------------------------------
    rf_train <- randomForest(formula, data = train_final, importance = TRUE)
  
    # ----------------------------------
    # 3. Tune parameters
    # ----------------------------------
    # update mtry
    num_predictors <- length(predictor_vars)
    mtry_values <- unique(c(
      max(1, floor(sqrt(num_predictors))),
      max(1, floor(num_predictors / 4)),
      max(1, floor(num_predictors / 3)),
      max(1, floor(num_predictors / 2))
      ))
  
    oob_error_mtry <- vector()
  
    for (mtry in mtry_values) {
    fit <- randomForest(formula, data = train_final, mtry = mtry, importance = TRUE)
    oob_error_mtry <- c(oob_error_mtry, fit$err.rate[500, 1])
      }
  
    mtry_final <- mtry_values[which.min(oob_error_mtry)]
  
    # update nodesize
    nodesize_values <- c(1, 5, 10, 20, 50)
    oob_error_nodesize <- vector()
    for (nodesize in nodesize_values){
    fit <- randomForest(formula, data = train_final, mtry = mtry_final, nodesize = nodesize, importance = TRUE)
    oob_error_nodesize <- c(oob_error_nodesize, fit$err.rate[500,1])
      }
  
    nodesize_final <- nodesize_values[which.min(oob_error_nodesize)]
  
    # final RF
    rf_final_training <- randomForest(formula, data = train_final, mtry = mtry_final, nodesize = nodesize_final, importance = T)
  
  



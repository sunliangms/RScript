# In this script, we apply caret and C50 packages to perform cross-validation
# for multi-class classification
rm(list=ls())
library('C50')
library('caret')
setwd('C:/Users/liasun/Documents/comparison/R/')
source('classification_measures.R')

# Please specify the data source
f_data <- 'C:/Users/liasun/Documents/Multi-Class/Data/Glass/csv/glass.noid.csv'
label_col <- 'last'

# Please choose the list of paremeters and the corresponding candidate values
para_list <- c('trials', 'minCases', 'winnow')
para_type <- rep('numeric', length(para_list))
para1_value_list <- c(1, seq(5, by=5, length=5))
para2_value_list <- seq(5, by=5, length=5)
para3_value_list <- c(T, F)
cv_fold = 5
training_ratio = 0.75

# Generate the grid for CV
myGrid <- expand.grid(trials = para1_value_list,
                      minCases = para2_value_list,
                      winnow = para3_value_list)

# Please select the metric to be optimized
# candidates: 'microF1', 'macroF1', 'accuracy'
measure_standard = 'microF1'


#=============== My Own Customized Code for caret Package ==============
lpC <- list(type='Classification', library='C50', loop=NULL)
prm <- data.frame(parameter = para_list, 
                  class = para_type,
                  label = para_list)
lpC$parameters <- prm

# Since we explicitly specify the parameter values in CV, this function is useless
C5Grid <- function(x, y, len = NULL) {
  expand.grid(trials = c(1, seq(5, by=5, length=len-1)), 
              minCases = seq(5, by=5, length=len),
              winnow = c(T, F))
}
lpC$grid <- C5Grid

C5Fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {  
  C5_control <- C5.0Control(noGlobalPruning = T, 
                            minCases = param$minCases, 
                            winnow = param$winnow)
  
  C5.0(x, y, trials = param$trials, rules = F, weights = NULL, 
       control = C5_control, costs = NULL, ...)  
}
lpC$fit = C5Fit

C5Pred <- function(modelFit, newdata, preProc = NULL, submodels = NULL) {
  predict(modelFit, newdata)
}
lpC$predict <- C5Pred

C5Prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type="prob")
lpC$prob <- C5Prob

C5Sort <- function(x) x[order(x$trials), ]
lpC$sort <- C5Sort

# This custom function is used to compute the performance metric 
# for multi-class classification to work with the caret package.
# We compute microF1, macroF1, and accuracy in this function.
multiClassMetric <- function(data, lev = NULL, model = NULL) {
  predV <- data$pred
  obsV <- data$obs
  ConfM <- confusion_multi(obsV, predV, levels(obsV)) 
  M <- ConfM$confusionMatrix
  maF1 <- macroF1(M)
  miF1 <- microF1(M)
  accu <- accuracyConf(M)
  out <- c(maF1, miF1, accu)
  names(out) <- c('macroF1', 'microF1', 'accuracy')
  out
}
#=============== My Own Customized Code for caret Package ==============



#================ Use Custom caret Package to Perform CV ==============
# Read the data
D <- read.csv(f_data, header=F)
n_sample <- dim(D)[1]
n_feature <- dim(D)[2] - 1
if (label_col =='last') {
  D.x <- D[,1:n_feature]
  D.y <- D[[n_feature+1]]
  if (!is.factor(D.y)) {
    D.y <- as.factor(D.y)
  }
}

set.seed(998)
inTraining <- createDataPartition(D.y, p = training_ratio, list = FALSE)
D.x.train <- D.x[inTraining, ]
D.y.train <- D.y[inTraining]
D.x.test <- D.x[-inTraining, ]
D.y.test <- D.y[-inTraining]


# train the model
fitControl <- trainControl(method = "repeatedcv",
                             number = cv_fold,
                             repeats = 5,
                             summaryFunction = multiClassMetric)
set.seed(825)
myC50 <- train(D.x.train, D.y.train, method = lpC, 
                 trControl = fitControl, tuneGrid = myGrid, metric = measure_standard)
# make prediction
D.y.test.hat <- predict(myC50, newdata = D.x.test)
# compute the performance metrics
ConfM <- confusion_multi(D.y.test, D.y.test.hat, levels(D.y.test)) 
M <- ConfM$confusionMatrix
maF1 <- macroF1(M)
miF1 <- microF1(M)
accu <- accuracyConf(M)


#=============== My Own Customized Code for caret Package ==============


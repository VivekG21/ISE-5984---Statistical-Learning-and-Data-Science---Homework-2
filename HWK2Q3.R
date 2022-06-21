
# Set working directory
setwd("C:/ISE 5984")



# Libraries are read in
library(data.table)
library(caret)
library(pROC)
library(mlbench)
library(tm)
library(class)
library(ISLR)
library(MASS)
library(rpart)
library(rpart.plot)
library(Rcpp)
library(InformationValue)
library(ISLR)
library("partykit")
library(dplyr)
library(ROSE)
library(car)



# The data is read into R Studio 
data = read.csv("hmeq.csv",stringsAsFactors = FALSE)

# The data is pre-processed for further analysis

# The variable 'BAD' is converted from integer to factor
data$BAD <- as.factor(data$BAD)
# Integer variables are converted to numeric variables
data$LOAN <- as.numeric(data$LOAN)
data$NINQ <- as.numeric(data$NINQ)
data$CLNO <- as.numeric(data$CLNO)
data$DEROG <- as.numeric(data$DEROG)
data$DELINQ <- as.numeric(data$DELINQ)
str(data)
summary(data)
data<-na.omit(data)
dim(data)

# Boxplots of Numeric Variables are generated 
par(mfrow = c(2,3))
boxplot(data$LOAN, main = "LOAN")
boxplot(data$MORTDUE, main = "MORTDUE")
boxplot(data$VALUE, main = "VALUE")
boxplot(data$YOJ, main = "YOJ")
boxplot(data$DEROG, main = "DEROG")
boxplot(data$DELINQ, main = "DELINQ")
boxplot(data$CLAGE, main = "CLAGE")
boxplot(data$NINQ, main = "NINQ")
boxplot(data$CLNO, main = "CLNO")
boxplot(data$DEBTINC, main = "DEBTINC")

# The outliers are removed
outliers_remover <- function(a){
  df <- a
  aa<-c()
  count<-1
  for(i in 1:ncol(df)){
    if(is.numeric(df[,i])){
      Q3 <- quantile(df[,i], 0.75, na.rm = TRUE)
      Q1 <- quantile(df[,i], 0.25, na.rm = TRUE) 
      IQR <- Q3 - Q1  
      upper <- Q3 + 1.5 * IQR
      lower <- Q1 - 1.5 * IQR
      for(j in 1:nrow(df)){
        if(is.na(df[j,i]) == TRUE){
          next
        }
        else if(df[j,i] > upper | df[j,i] < lower){
          aa[count]<-j
          count<-count+1                  
        }
      }
    }
  }
  df<-df[-aa,]
}

data <- outliers_remover(data)
dim(data)


par(mfrow = c(2,2))
barplot(table(data$REASON), main = "REASON")
barplot(table(data$JOB), main = "JOB")
barplot(table(data$BAD), main = "BAD")



#The errors in the barplots are corrected

data <- data[!(data$REASON ==""),]
data <- data[!(data$JOB ==""),]
par(mfrow = c(1,2))
barplot(table(data$REASON), main = "REASON")
barplot(table(data$JOB), main = "JOB")
dim(data)

# Correlation Matrix is created
cor(data[c("LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC")])


data$DEROG <- NULL
data$DELINQ <- NULL
# There is multicollinearity between MORTDUE and VALUE
# MORTDUE is deleted
data$MORTDUE <- NULL 
dim(data)

# The training set and validation set data is created
input_ones <- data[which(data$BAD == 1), ] #all 1's
input_zeros <- data[which(data$BAD == 0), ] # all 0's
set.seed(100) #This is for the repeatability of sample
input_ones_training_rows <- sample(1:nrow(input_ones), 0.5 * nrow(input_ones)) #1's for training
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.5 * nrow(input_zeros)) #0's for training
input_ones_validation_rows <- sample(1:nrow(input_ones), 0.25 * nrow(input_ones)) #1's for validation
input_zeros_validation_rows <- sample(1:nrow(input_zeros), 0.25 * nrow(input_zeros)) #0's for validation
#pick as many as 0's and 1's
training_ones <- input_ones[input_ones_training_rows, ]
training_zeros <- input_zeros[input_zeros_training_rows, ]
validation_ones <- input_ones[input_ones_validation_rows, ]
validation_zeros <- input_zeros[input_zeros_validation_rows, ]
# We row bind the 0's and 1's
trainingData <- rbind(training_ones, training_zeros)
validationData <- rbind(validation_ones, validation_zeros)

# We create the test set data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
#1's and 0's and row binded 
testData <- rbind(test_ones, test_zeros)

table(trainingData$BAD)

prop.table(table(trainingData$BAD))

treeMod <- rpart(BAD ~., data = trainingData)
pred_treeMod <- predict(treeMod, newdata = testData)
pred_treeMod

accuracy.meas(testData$BAD, pred_treeMod[,2])


# We re-sample the data set 

data_balanced_over <- ovun.sample(BAD ~., data = trainingData, method = "over", N = 2912)$data
table(data_balanced_over$BAD)

data_balanced_both <- ovun.sample(BAD ~., data = trainingData, method = "both", p = 0.5, N = 1528)$data
table(data_balanced_both$BAD)


#Decision Tree Models are built
tree.over <- rpart(BAD ~., data = data_balanced_over)
tree.both <- rpart(BAD ~., data = data_balanced_both)


# make predictions on test data
pred_tree.over <- predict(tree.over, newdata = testData)
pred_tree.both <- predict(tree.both, newdata = testData)


#ROC Curves are created to show error rates for Decision Tree Algorithm
par(mfrow = c(2,2))

roc.curve(testData$BAD, pred_tree.over[,2], col = "BLACK", main = "ROC curve of oversampling")
roc.curve(testData$BAD, pred_tree.both[,2], col = "RED", main = "ROC curve of balanced sampling")

# We conduct a logistic regression on the training data
logisticModel <- glm(BAD ~., data = trainingData, family = binomial(link = "logit"))
pred_logit <- predict(logisticModel, testData)
summary(logisticModel)
summary(pred_logit)
#Doing a logistic Regression on the validation set to fine tune the model
mylogit1 <- glm(BAD ~., data=validationData , family =binomial(link = "logit"))
pred_logit1 <- predict(mylogit1, testData)
summary(mylogit1)
summary(pred_logit1)
# We look at the VIF to check for multicollinearity in the regression model 
vif(mylogit1)
#Doing a logistic Regression on the test set after fine tuning the regression model
mylogit2 <- glm(BAD ~ NINQ +VALUE +DEBTINC +CLAGE, data=testData, family = binomial(link = "logit"))
pred_logit2 <- predict(mylogit2, testData)
summary(pred_logit2)
vif(mylogit2)
summary(mylogit2)
# Returns the cutoff that gives minimum misclassification error.
cutoff  <- optimalCutoff(testData$BAD, pred_logit)[1] 
cutoff

# Calculating Error Rate for Logistic Regression
misClassError(testData$BAD, pred_logit2, threshold = cutoff)
misClassError(testData$BAD, pred_logit2, threshold = 0.5)
confusionMatrix(testData$BAD, pred_logit, cutoff)

specificity(testData$BAD, pred_logit2, cutoff)
sensitivity(testData$BAD, pred_logit2, cutoff)
accuracy.meas(testData$BAD, pred_logit2, cutoff)

plotROC(testData$BAD, pred_logit)
# We conduct an LDA analysis on training and validation datasets
Equity.lda <- lda(BAD ~., data = trainingData)
Equity.lda
fit.hat <- predict(Equity.lda,testData)
summary(fit.hat)
Equity.lda2 <- lda(BAD ~., data = validationData)
Equity.lda2
fit.hat2 <- predict(Equity.lda2,testData)
# There are vast differences in the means for the variable JOB
# We remove the variable JOB from the LDA analysis on the test to see if it improves accuracy of model
Equity.lda3 <- lda(BAD ~ LOAN+VALUE+REASON+YOJ+CLAGE+NINQ+CLNO+DEBTINC, data = testData)
Equity.lda3
fit.hat3 <- predict(Equity.lda3,testData)
# It turns out that re-sampling the data improves the efficiency of the LDA model
Equity.lda4 <- lda(BAD ~., data =data_balanced_both )
Equity.lda4

#Plots of Decision Trees

rpart.plot(treeMod)
rpart.plot(tree.over)
rpart.plot(tree.both)








# Set working directory
setwd("C:/ISE 5984")


# The libraries are read in 
library(caret)
library(naivebayes)
library(MASS)
library(ISLR)
library(pROC)
library(mlbench)
library(tm)
library("FNN")
library(e1071)
library(data.table)

# The training and test data is read in.
train <- read.table(file.path(getwd(), "zip.train.gz"))
test <- read.table(file.path(getwd(), "zip.test.gz"))
head(train)
## Filtering data set to 2's, 3's, and 5's 
train <- train[train[,1] %in% c(2, 3,5),]
test <- test[test[,1] %in% c(2, 3,5),]
# Reducing models to key variables 
pixel <- c("V1", "V3", "V5", "V7", "V15")
train <- train[,pixel]
test <- test[,pixel]


train$V1 <- as.numeric(train$V1)
train$V3 <- as.numeric(train$V3)
train$V5 <- as.numeric(train$V5)
train$V7 <- as.numeric(train$V7)
train$V15<- as.numeric(train$V15)


# Generate correlation matrix to detect for multicollinearity
cor(train[c("V1", "V3", "V5", "V7", "V15")])

# Building Naive Bayes Model on Training Set
TrainY = train[,1]
TrainY = as.matrix(TrainY)
TrainY = factor(TrainY)
TrainX = train[2:5]
head(TrainX)
bayes.model.train <- naive_bayes(TrainY~., data=train[,pixel])
bayes.model.train
summary(bayes.model.train)
predBayes <- predict(bayes.model.train, test)
summary(predBayes)



TestY = test[,1]
TestY = as.matrix(TestY)
TestY = factor(TestY)
TestX = test[2:5]
head(TestX)

# Applying Naive Bayes Model on Test Set
bayes.model.test <- naive_bayes(TestY~., data=test[,pixel])
bayes.model.test
summary(bayes.model.test)

# Conducting LDA analysis on training set
mylDA <- lda(formula =  V1 ~.,data=train)
mylDA 
summary(mylDA)
# Predicting LDA training model on test data
predLDA <- predict(mylDA, test)
predLDA

# Conducting LDA analysis on test set
mylDA2 <- lda(formula =  V1 ~.,data=test)
mylDA2 
summary(mylDA2)

# Conducting SVM regression on training data

svm.model <- svm(V1 ~ ., data = train, kernel = "radial", cost = 1, gamma = 0.1)
svm.model
svm.predict <- predict(svm.model, test)
svm.predict

# Conducting SVM regression on test data
svm.model.test <- svm(V1 ~ ., data = test, kernel = "radial", cost = 1, gamma = 0.1)
svm.model.test

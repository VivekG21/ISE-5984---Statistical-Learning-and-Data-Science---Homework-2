

# Set working directory
setwd("C:/ISE 5984")

# Libraries are read in 
library(data.table)
library(caret)
library(pROC)
library(mlbench)
library(tm)
library("FNN")



data = read.csv("hmeq.csv",stringsAsFactors = FALSE)


# Convert the variable 'BAD' from integer to factor
data$BAD <- as.factor(data$BAD)


# Convert integer variables to numeric variables
data$LOAN <- as.numeric(data$LOAN)
data$NINQ <- as.numeric(data$NINQ)
data$CLNO <- as.numeric(data$CLNO)
data$DEROG <- as.numeric(data$DEROG)
data$DELINQ <- as.numeric(data$DELINQ)
str(data)
summary(data)
data<-na.omit(data)
dim(data)


par(mfrow = c(2,3))

# Boxplots of all numeric variables
boxplot(data$MORTDUE, main = "MORTDUE")
boxplot(data$VALUE, main = "VALUE")
boxplot(data$YOJ, main = "YOJ")
boxplot(data$DEROG, main = "DEROG")
boxplot(data$DELINQ, main = "DELINQ")
boxplot(data$CLAGE, main = "CLAGE")
boxplot(data$NINQ, main = "NINQ")
boxplot(data$CLNO, main = "CLNO")
boxplot(data$DEBTINC, main = "DEBTINC")

# Function to remove outliers
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

# Removing outliers in the raw data set
data <- outliers_remover(data)
dim(data)


# Creating barplots
par(mfrow = c(2,2))
barplot(table(data$REASON), main = "REASON")
barplot(table(data$JOB), main = "JOB")
barplot(table(data$BAD), main = "BAD")



# Correct the errors in the barplots
data <- data[!(data$REASON ==""),]
data <- data[!(data$JOB ==""),]
par(mfrow = c(1,2))
barplot(table(data$REASON), main = "REASON")
barplot(table(data$JOB), main = "JOB")
dim(data)


# Correlation Matrix 
cor(data[c("LOAN","MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC")])



data$DEROG <- NULL
data$DELINQ <- NULL
#  Multicollinearity is present between variables MORTDUE and VALUE
data$MORTDUE <- NULL # delete MORTDUE
dim(data)

# The training data is created, which is 70% of the raw data
input_ones <- data[which(data$BAD == 1), ] #all 1's
input_zeros <- data[which(data$BAD == 0), ] # all 0's
set.seed(100) # for repeatability of sample
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7 * nrow(input_ones)) 
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7 * nrow(input_zeros)) 
#pick as many as 0's and 1's
training_ones <- input_ones[input_ones_training_rows, ]
training_zeros <- input_zeros[input_zeros_training_rows, ]
# We row bind the  0's and 1's for training data
trainingData <- rbind(training_ones, training_zeros)

# The test data is created, which is 30% of the raw data
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
# We row bind the  0's and 1's for test data
testData <- rbind(test_ones, test_zeros)


table(trainingData$BAD)

prop.table(table(trainingData$BAD))
dim(trainingData)
set.seed(111)
trControl <- trainControl(method  = "cv",
                          number  = 5)
s = function(seeds_list,k){
  seeds_list = lapply(seeds_list,"[",1:k)
  seeds_list[[length(seeds_list)+1]] = 999
  seeds_list
}


# Implement knn algorithm on training data for various values of k

k = 1
model0 <- train(BAD~.,data=data,method="knn",
                trControl = trainControl(method = 'LOOCV'),tuneGrid = expand.grid(k = 1:k))
model0

k = 5

model <- train(BAD~.,data=data,method="knn",
               trControl = trainControl(method = 'LOOCV'),tuneGrid = expand.grid(k = 1:k))
model

k = 10

model1 <- train(BAD~.,data=data,method="knn",
               trControl = trainControl(method = 'LOOCV'),tuneGrid = expand.grid(k = 1:k))
model1

k = 20

model2 <- train(BAD~.,data=data,method="knn",
                trControl = trainControl(method = 'LOOCV'),tuneGrid = expand.grid(k = 1:k))
model2

# The optimal value for k is 13, which gives the highest accuracy in predicting default

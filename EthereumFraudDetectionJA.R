#
#  Ethereum - Fraud Detection 
#  Jai Agrawal (agrawalj@uci.edu)
#  
#  
#
library(dplyr)
library(corrplot)
library(leaps)
library(e1071)
library(class)
library(randomForest)
library(caret)

# Read in Ethereum data set
dat <- read.csv("transaction_dataset.csv")

# Dummy Variable Conversion
#------------------------------------------------------------------------------------------
# Converting categorical variables to dummy variables
#sent.type <- dat$ERC20.most.sent.token.type 
#rec.type <- dat$ERC20_most_rec_token_type
#tmp_sent <- data.frame(model.matrix(~sent.type-1))
#tmp_rec <- data.frame(model.matrix(~rec.type-1))
#num<-data.frame(colSums(tmp_sent))
#dat1 <- cbind(dat[,1:51],tmp_rec[,1:467],tmp_sent[,1:305])

#Intial Cleaning
#------------------------------------------------------------------------------------------
# Removing #17 unnecessary variables, left with #34 (from #51 raw data)
dat<- dat[-c(1,2,3,19,21,22,25,26,35,36,37,38,45,46,47,50,51)]
table(is.na(dat))
# Distribution of 0s (7662) and 1s (2179)
d <- table(dat$FLAG)
d

# Removing NAs, left with #9012 records from #9814 original
dat <-na.omit(dat)
str(dat)
# Distribution after removing NAs 0s (7662) and 1s (1350)
d2 <- table(dat$FLAG)
d2

#------------------------------------------------------------------------------------------
# Correlation matrix
corr_simple <- function(data=df,sig=0.5){
  #convert data to numeric in order to run correlations
  #convert to factor first to keep the integrity of the data - each value will become a number rather than turn into NA
  df_cor <- data %>% mutate_if(is.character, as.factor)
  df_cor <- df_cor %>% mutate_if(is.factor, as.numeric)
  corr <- cor(df_cor, method = 'pearson')
  corr[lower.tri(corr,diag=TRUE)] <- NA 
  corr[corr == 1] <- NA 
  corr <- as.data.frame(as.table(corr))
  
  #remove the NA values from above 
  corr <- na.omit(corr) 
  corr <- subset(corr, abs(Freq) > 0.5) 
  corr <- corr[order(-abs(corr$Freq)),] 
  print(corr)
  mtx_corr <- reshape2::acast(corr, Var1~Var2, value.var="Freq")
  
  #plot correlations visually
  corrplot(mtx_corr, is.corr=FALSE, tl.col="black", na.label=" ",
           tl.cex = 0.3, type = 'upper')
}
corr_simple(dat, sig = 0.8)

# We see there is high correlation among ERC20's total ether sent and its
# min, max, avg values.
#------------------------------------------------------------------------------------------
# Split data set for into train and test  
set.seed(112233)

fraud <- subset(dat, dat$FLAG == 1)
real <- subset(dat, dat$FLAG == 0)

nrow(fraud)
nrow(real)

train.fraud <- sample(1:nrow(fraud),675)
train.real <- sample(1:nrow(real),675)

newreal<-real[-train.real,]
test.real <- newreal[sample(1:nrow(newreal),675),]

dat.test <- rbind(fraud[-train.fraud,],test.real)
table(dat.test$FLAG)
dat.train <- rbind(fraud[train.fraud,],real[train.real,])
str(dat.train)
dim(dat.train)
flag.count.train <- table(dat.train$FLAG)
flag.count.train
#------------------------------------------------------------------------------------------
# Logistic Regression 
logreg <- glm(FLAG ~ .-max.val.sent.to.contract, data = dat.train,
              family = "binomial")
summary(logreg)     # AIC = 1283.6
# 16 variables are significant
sum(summary(logreg)$coefficients[,4] < 0.1)

# Predicting on test data
y.test <- as.data.frame(predict(logreg, dat.test, 
        type = "response"))
yhat.test <- ifelse(y.test > 0.5,1,0)

# Confusion matrix,Accuracy,Precision,Recall   
result.log <- confusionMatrix(as.factor(yhat.test),as.factor(dat.test$FLAG))
Accuracy.log<-result.log$byClass['Balanced Accuracy']  
precision.log <- result.log$byClass['Pos Pred Value']    
recall.log <- result.log$byClass['Sensitivity']
test.error <- mean(dat.test$FLAG != yhat.test)
test.error    # 16.8% error rate
#---------------------------------------------------
# Naive Baye's 
nb.fit <- naiveBayes(FLAG ~ ., data = dat.train)
nb.pred <- predict(nb.fit, newdata = dat.test)
err.nb <- mean(dat.test$FLAG != nb.pred)
err.nb # Overall Error = 43.2% (Bayesian Poising, need to remove some features 
# that are not independent)
# Confusion matrix,Accuracy,Precision,Recall   
result.nb <- confusionMatrix(as.factor(nb.pred),as.factor(dat.test$FLAG))
Accuracy.nb<-result.nb$byClass['Balanced Accuracy']  
precision.nb <- result.nb$byClass['Pos Pred Value']    
recall.nb <- result.nb$byClass['Sensitivity']
#---------------------------------------------------
#KNN
dat.train.x <- dat.train[,2:34]
dat.train.y <- dat.train[,1]
dat.test.x <- dat.test[,2:34]
dat.test.y <- dat.test[,1]
# k=1
out1 <- knn(dat.train.x, dat.test.x, dat.train.y, k=1)
tab.knn1 <- table(dat.test.y, out1,
                  dnn = c("Actual", "Predicted"))
tab.knn1
knn1.err <- mean(dat.test.y != out1)
knn1.err # Overall error = 18.7%
# k=3
out3 <- knn(dat.train.x, dat.test.x, dat.train.y, k=3)
tab.knn3 <- table(dat.test.y, out3,
                  dnn = c("Actual", "Predicted"))
tab.knn3
knn3.err <- mean(dat.test.y != out3)
knn3.err # Overall Error = 17.5%
#k=5
out5 <- knn(dat.train.x, dat.test.x, dat.train.y, k=5)
tab.knn5 <- table(dat.test.y, out5,
                  dnn = c("Actual", "Predicted"))
tab.knn5
knn5.err <- mean(dat.test.y != out5)
knn5.err # Overall Error = 15.25%
#k=7
out7 <- knn(dat.train.x, dat.test.x, dat.train.y, k=7)
knn7.err <- mean(dat.test.y != out7)
knn7.err # Overall Error = 15.85%
# Confusion matrix,Accuracy,Precision,Recall   
result.knn <- confusionMatrix(as.factor(out7),as.factor(dat.test$FLAG))
Accuracy.knn<-result.knn$byClass['Balanced Accuracy']  
precision.knn <- result.knn$byClass['Pos Pred Value']    
recall.knn <- result.knn$byClass['Sensitivity']
#K= 5 performs the best, Overall Error = 15.25% 
#--------------------------------------------------------------------
# Classification Tree
library(tree)
dat.train[,1] <- as.factor(dat.train[,1])
dat.test[,1] <- as.factor(dat.test[,1])
tree1 <- tree(FLAG~., data = dat.train)
summary(tree1)
plot(tree1)
tree.pred.tst <- predict(tree1, dat.test, type = "class")
# Confusion matrix,Accuracy,Precision,Recall   
result.tree <- confusionMatrix(as.factor(tree.pred.tst),as.factor(dat.test$FLAG))
Accuracy.tree <-result.tree$byClass['Balanced Accuracy']  
precision.tree <- result.tree$byClass['Pos Pred Value']    
recall.tree <- result.tree$byClass['Sensitivity']
err.tree<-mean(dat.test$FLAG != tree.pred.tst)
err.tree # Overall Error Rate = 7.2%
#-------------------------------------------------------------------------------------
# Pruning the tree
prune1 <- prune.misclass(tree1)
plot(prune1$size, prune1$dev, xlab = "Size of Tree",
     ylab = "Deviation")
prune.tree1 <- prune.misclass(tree1, best = 7)
pt1.pred <- predict(prune.tree1, dat.test, type = "class")
# Confusion matrix,Accuracy,Precision,Recall   
result.ptree <- confusionMatrix(as.factor(pt1.pred),as.factor(dat.test$FLAG))
Accuracy.ptree <-result.ptree$byClass['Balanced Accuracy']  
precision.ptree <- result.ptree$byClass['Pos Pred Value']    
recall.ptree <- result.ptree$byClass['Sensitivity']
err.prune.tree<-mean(dat.test$FLAG != pt1.pred)
err.prune.tree # Overall Error Rate = 7.2%
# Pruned the tree from 8 to 7 
#--------------------------------------------------------------------------------
# Random Forest 
set.seed(223344)
bag.train.10 <- randomForest(FLAG ~ ., 
                             data = dat.train, 
                             mtry = 33, ntree = 10, 
                             importance = TRUE)
yhat.bag.10 <- predict(bag.train.10, dat.test)
tab.bag.10 <- table(dat.test$FLAG, yhat.bag.10)
tab.bag.10
err.bag10 <- mean(dat.test$FLAG != yhat.bag.10)
err.bag10 #Overall Error =4.1 %
# Finding the best value of ntree
rf.err<- 1:50
xrange<-1:50
for (i in 1:50) {
  n<-i*10
  bag.train<- randomForest(FLAG ~ ., 
                               data = dat.train, 
                               mtry = 33, ntree = n, 
                               importance = TRUE)
  yhat.bag <- predict(bag.train, dat.test)
  rf.err[[i]] <- mean(dat.test$FLAG != yhat.bag)
}
length(rf.err)
plot(xrange*10, rf.err, xlab = "Value of ntree",
     ylab = "Error from RF")
# Best Rf 
bag.train.best <- randomForest(FLAG ~ ., 
                             data = dat.train, 
                             mtry = 33, ntree = 190, 
                             importance = TRUE)
yhat.bag.best <- predict(bag.train.best, dat.test)
tab.bag.best <- table(dat.test$FLAG, yhat.bag.best)
tab.bag.best
result.rf <- confusionMatrix(as.factor(yhat.bag.best),as.factor(dat.test$FLAG))
Accuracy.rf <-result.rf$byClass['Balanced Accuracy']  
precision.rf <- result.rf$byClass['Pos Pred Value']    
recall.rf <- result.rf$byClass['Sensitivity']
err.bag.best <- mean(dat.test$FLAG != yhat.bag.best)
err.bag.best # Overall Error = 2.8%
# Variable Selection 
rf.train <- randomForest(FLAG ~ ., data = dat.train, 
                         mtry = 16, ntree = 430, 
                         importance = TRUE)
yhat.rf <- predict(rf.train, dat.test)
tab.rf <- table(dat.test$FLAG, yhat.rf)
tab.rf
err.rf <- mean(dat.test$FLAG != yhat.rf)
err.rf
importance(rf.train)
varImpPlot(rf.train, main = "Variable Importance Plot")
imp.rf <- importance(rf.train)
imp.rf[1:10,]
#
#  Top variables by accuracy and GINI index are
#
sort(importance(rf.train)[,3], decreasing = TRUE)[1:10]
sort(importance(rf.train)[,4], decreasing = TRUE)[1:10]


# Summarizing Results 
sumstats.tab <- as.data.frame(matrix(nrow = 3, ncol = 5))
sumstats.tab[1,1] <- Accuracy.log
sumstats.tab[1,2] <- Accuracy.nb
sumstats.tab[1,3] <- Accuracy.knn
sumstats.tab[1,4] <- Accuracy.tree
sumstats.tab[1,5] <- Accuracy.rf
sumstats.tab[2,1] <- precision.log
sumstats.tab[2,2] <- precision.nb
sumstats.tab[2,3] <- precision.knn
sumstats.tab[2,4] <- precision.tree
sumstats.tab[2,5] <- precision.rf
sumstats.tab[3,1] <- recall.log
sumstats.tab[3,2] <- recall.nb
sumstats.tab[3,3] <- recall.knn
sumstats.tab[3,4] <- recall.tree
sumstats.tab[3,5] <- recall.rf
colnames(sumstats.tab) <- c("Logistic Regression", "Naive Bayes", "KNN", "Decision Tree",
                            "Random Forest")
rownames(sumstats.tab) <- c("Accuracy","Precision","Recall")
sumstats.tab
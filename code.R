
## Content
* **[Step 1](#Step1)**

* [Treatment of N/A](#Treatment)
* [Why not impute Cabin discussion](#Cabindiscussion)
* [Comment on distribution of Varables](#Comment) 
* [Is sex a predictor or is it a policy over-ride?](#sexpredictor)

* **[Step 2](#Step2)**

* [Logit](#Logit) 
* [LDA](#lda) 
* [QDA](#qda) 
* [NB](#nb)  
* [KNN](#knn)

* [Feature selection using Lasso](#Lasso)
* [Comment on multi-collinearity](#collinearity)
* [Estimate errors using bootstrap](#bootstrap)

* *k-fold cross validation setup*
* [Logit CV](#Logitcv) 
* [LDA CV](#ldacv) 
* [QDA CV](#qdacv) 
* [NB CV](#nbcv)  
* [KNN CV](#knncv)

* [Explain models differences](#various)

* **[Step 3](#Step3)**
* *Compare performance using ROC and AUC measures*
* [Logit performance](#Logitp) 
* [LDA performance](#ldap) 
* [QDA performance](#qdap) 
* [NB performance](#nbp)  
* [KNN performance](#knnp)

* **[Step 4](#Step4)**

* [Treatments of test data](#testD)

I did the Treatments of the test data early to make sure a got the similar resort like the training data set
* [Identify which model is selected using x-validation](#Identify)
* [Rebuild the model using full train dataa](#Rebuild)
* [Submit results](#Submit)

### Needed libraries 

library(caret)
library(Amelia)
library(Rcpp)
library(ISLR2)
library(Matrix)
library(glmnet)
library(ROCR)
library(pROC)
library(ggplot2)
library(lattice)
library(kernlab)
library(regclass)



# Convert features to factors \
# Features: Name, Sex, Ticket, Cabin, Survived \

tf$Embarked = as.factor(tf$Embarked) 
tf$Sex = as.factor(tf$Sex)
tf$Ticket = as.factor(tf$Ticket)
tf$Cabin = as.factor(tf$Cabin)
tf$Survived = as.factor(tf$Survived)
attach(tf) ;
str(tf)



tef$Embarked = as.factor(tef$Embarked) 
tef$Sex = as.factor(tef$Sex)
tef$Ticket = as.factor(tef$Ticket)
tef$Cabin = as.factor(tef$Cabin)


# Check NA in Age \
# Use *sapply* to get % of NA for all variables \


sum(is.na(tf$Age)==TRUE) # 177 out of 891 are NA
sum(is.na(tf$Age)==TRUE)/length(tf$Age) # 20% are NA
sapply(tf,function(df){100*sum(is.na(df==TRUE)/length(df))}) # get a % of NA for all variables


sum(is.na(tef$Age)==TRUE) # 177 out of 891 are NA
sum(is.na(tef$Age)==TRUE)/length(tef$Age) # 20% are NA
sapply(tef,function(df){100*sum(is.na(df==TRUE)/length(df))}) # get a % of NA for all variables


## Impute missing Embarked by marking it using most counted port Southampton \


table(tf$Embarked, useNA="always") # to show number of NA values in the dataset
tf$Embarked[which(is.na(tf$Embarked))]='S' # assign missing value to most counted port Southampton
table(tf$Embarked, useNA="always") # no NA and S went up by two
  
  
  
table(tef$Embarked, useNA="always") # to show number of NA values in the dataset
tef$Embarked[which(is.na(tef$Embarked))]='S' # assign missing value to most counted port Southampton
table(tef$Embarked, useNA="always") # no NA and S went up by two



tf$Name = as.character(tf$Name)
table_words = table(unlist(strsplit(tf$Name, "\\s+"))) # tokenize the names
sort(table_words [grep('\\.',names(table_words))],decreasing=TRUE)
library(stringr)
tb=cbind(tf$Age, str_match(tf$Name, "[a-zA-Z]+\\.")) 
table(tb[is.na(tb[,1]),2]) 

mean.mr=mean(tf$Age[grepl(" Mr\\.", tf$Name) & !is.na(tf$Age)])
mean.mrs=mean(tf$Age[grepl(" Mrs\\.", tf$Name) & !is.na(tf$Age)])
mean.dr=mean(tf$Age[grepl(" Dr\\.", tf$Name) & !is.na(tf$Age)])
mean.miss=mean(tf$Age[grepl(" Miss\\.", tf$Name) & !is.na(tf$Age)])
mean.master=mean(tf$Age[grepl(" Master\\.", tf$Name) & !is.na(tf$Age)])

tf$Age[grepl(" Mr\\.", tf$Name) & is.na(tf$Age)]=mean.mr
tf$Age[grepl(" Mrs\\.", tf$Name) & is.na(tf$Age)]=mean.mrs
tf$Age[grepl(" Dr\\.", tf$Name) & is.na(tf$Age)]=mean.dr
tf$Age[grepl(" Miss\\.", tf$Name) & is.na(tf$Age)]=mean.miss
tf$Age[grepl(" Master\\.", tf$Name) & is.na(tf$Age)]=mean.master
sapply(tf,function(df){100*sum(is.na(df==TRUE)/length(df))}) # get a % of NA for all variables


tef$Name = as.character(tef$Name)
table_words = table(unlist(strsplit(tef$Name, "\\s+"))) # tokenize the names
sort(table_words [grep('\\.',names(table_words))],decreasing=TRUE)
library(stringr)
teb=cbind(tef$Age, str_match(tef$Name, "[a-zA-Z]+\\.")) 
table(teb[is.na(teb[,1]),2]) 

mean.mr=mean(tef$Age[grepl(" Mr\\.", tef$Name) & !is.na(tef$Age)])
mean.mrs=mean(tef$Age[grepl(" Mrs\\.", tef$Name) & !is.na(tef$Age)])
mean.dr=mean(tef$Age[grepl(" Dr\\.", tef$Name) & !is.na(tef$Age)])
mean.miss=mean(tef$Age[grepl(" Miss\\.", tef$Name) & !is.na(tef$Age)])
mean.master=mean(tef$Age[grepl(" Master\\.", tef$Name) & !is.na(tef$Age)])

tef$Age[grepl(" Mr\\.", tef$Name) & is.na(tef$Age)]=mean.mr
tef$Age[grepl(" Mrs\\.", tef$Name) & is.na(tef$Age)]=mean.mrs
tef$Age[grepl(" Dr\\.", tef$Name) & is.na(tef$Age)]=mean.dr
tef$Age[grepl(" Miss\\.", tef$Name) & is.na(tef$Age)]=mean.miss
tef$Age[grepl(" Master\\.", tef$Name) & is.na(tef$Age)]=mean.master
sapply(tef,function(df){100*sum(is.na(df==TRUE)/length(df))}) # get a % of NA for all variables


which(is.na(tef$Fare))

tef$Fare[153] <- median(tef$Fare, na.rm=TRUE) #changeing the fare missing value with the medina since there are only one missing fare value
tef$Age[89] <- median(tef$Age, na.rm=TRUE)


# Plot the data and explain the results before building the model \


hist(tf$Age, main="Passenger Age", xlab = "Age")
barplot(table(tf$SibSp),main="Passenger Siblings")
barplot(table(tf$Parch),main="Passenger Parch")
hist(table(tf$Far),main="Passenger Fare", xlab="Fare")
barplot(table(tf$Embarked),main="Port of Embarkation")
counts = table(tf$Survived,tf$Sex)
barplot(counts, col=c("darkblue","red"), legend=c("Perished","Survived"), main = "Passenger Survival by Sex")
countsC = table(tf$Survived,tf$Pclass)
barplot(countsC, col=c("darkblue","red"), legend=c("Perished","Survived"), main = "Titanic Class Bar Plot")
hist(tf$Age[which(tf$Survived=="0")], main = "Passenger Age Histogram", xlab="Age",ylab="Count", col="blue", breaks=seq(0,80, by=2))
hist(tf$Age[which(tf$Survived=="1")], col="red", add=T,breaks=seq(0,80, by=2))
boxplot(tf$Age ~ tf$Survived, main="Passenger Survival by Age",xlab="Survived",ylab="Age")

# Categorize passengers by age buckets and survival by age group \


tf.child=tf$Survived[tf$Age<13]
length(tf.child[which(tf.child==1)])/length(tf.child)
tf.youth=tf$Survived[tf$Age>=15 & tf$Age<25]
length(tf.youth[which(tf.youth==1)])/length(tf.youth)
tf.adult=tf$Survived[tf$Age>=25 & tf$Age<65]
length(tf.adult[which(tf.adult==1)])/length(tf.adult)
tf.senior=tf$Survived[tf$Age>=65]
length(tf.senior[which(tf.senior==1)])/length(tf.senior)


# **Why we do not impute Cabin ?**
# while every predector are important butThe cabin has big issue that make us don't impute the Cabin
# * column in training dataset is missing almost 80% of the values. I don't think it will be wise if we depend our production in predector that has big missing values. 
# * we don't know how the Cabins are close to the life boats and how the evacuating the vehicle was. 


# **age**

# From the graphics we can observe that most of the people are Passengers are between the age 20 and 40 years old, and the survivors follow the normal distribution where most of the survivors are in their 20s or late 30es. However, we can notice a peak on the children survivors, The mean age for the survivors are younger than the mean for the non-survivors so,from the plots the age group could be an important factor for forecast to be survived or not at the Titanic especially With the peaks in the 30s and 20s age.

# **siblings**

# when we look at the siblings we can see, most of the people have no siblings aboard the Titanic but there are around 200 people who have one sibling aboard the Titanic.

# **parents or children**

# The factor if the passenger is parents or children doesn't show a big significance where most of the people are not parents or children, However, 57% of the children are survivors and that could be because of the evacuation policies and the same thing we can see at the women's gender.

# **Ticket class**

# When we look at the gender most of the passengers are males where there are around 60% of passengers, However when we look at the survivors' gender most of them are females, again that could be because of the eviction policies in the provision of children and women.

# **Ticket**

# The Ticket class are significant in the graphs, most of the passengers are 3ed class but the survivors at the 1st class have the highest percentage compared to the numbers of the 3ed class ticket holders. Also, the 2ed class people have more percentage of survivors than the 3ed class but not as high as the 1st class; Which mean there is bias in the evacuation process or it could be easier to get evacuation depending on the passenger class. However, it is observable that there is an effect between the ticket class and the survived situation

# **now** we remove some varables that could construction process of building the model:

#  **names** , where there are many manes has no significant wxpet the age group or the gernder which is aredy got it.

# **Cabin** , where there are many missing cabins and some of them has diffren ordar.

# **Ticket number** , the order of the tickets number differ and it didn't follow some pattren.


# Is sex a predictor or is it a policy over-ride?**
#While we can observe that the percentage of the female gender servers are much higher than the males compared to the numbers, and the numbers of the females are around half of the males because of that, I think the reason for the higher females is because of the policy override which prioritizes women.


# *removing unwanted variables*

tf <- tf[-c(4,9,11)]

# *removing same variables from the testing file (tef)*

tef <- tef[-c(3,8,10)]

# We are setting the seed at 3131 to make sure all the following results going to be the same always
set.seed(3131)  

Using **createDataPartition** function from **caret** library we making 70% training sample and 30% test sample from our data

trainIndex <- createDataPartition(y = tf$Survived, p= 0.7, list = FALSE)
# 70%  training data
training <- tf[trainIndex,] 
# 30% testing data
testing <- tf[-trainIndex,]

# Now we will make final check if the data missing or lost while we are working before starting to build the models

dim(training); dim(testing)
missmap(tf)
anyNA(tf)
anyNA(tef)


  ## Step2
  
  ### Regressions without k-fold cross validation
  <div id="Logit"></div>
  **Model 1 : Logistic Regression**

Logit_fit <- train(Survived ~., data = training, method = "glm",
                   preProcess = c("center", "scale"),
                   tuneLength = 10,
                   family = "binomial"
)

Logit_fit


  **Model 2 : Linear Discriminant Analysis (LDA)**

lda_fit <- train(Survived ~ .,
                 data = training, method = "lda",
                 metric = "Accuracy",
                 preProcess = c("center", "scale")
)
lda_fit

<div id="qda"></div>
  **Model 3 : Quadratic Discriminant Analysis (QDA)**

qda_fit <- train(Survived ~., 
                 data = training, 
                 method = "qda",
                 preProcess = c("center", "scale"),
                 tuneLength = 10
)

qda_fit

  **Model 4 : Naive Bayes (NB)**

nb_fit <- train(Survived ~., 
                data = training, 
                method = "naive_bayes",
                preProcess = c("center", "scale"),
                tuneLength = 10)

nb_fit


  **Model 5 :  K-Nearest Neighbors (KNN)**

knn_fit <- train(Survived ~., 
                 data = training, 
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

knn_fit



  ### Feature selection using Lasso
  # We are using 10-fold cross validation to check the Lasso selections  

# Define predictor matrix
x <- model.matrix(Survived ~ ., training)[, -1] # define predictor matrix
y <- training$Survived 

grid <- 10^seq(10, -2, length = 100)
x.test <- model.matrix(Survived ~ ., data = testing)[, -1]
model.lasso <- glmnet(x, y, alpha = 1, lambda = grid, thresh = 1e-12, family = "binomial")

# plot the coefficients with the log of Lama for the predictors
plot(model.lasso, xvar = "lambda", label = TRUE)


cv.out <- cv.glmnet(x, y, 
                    family = "binomial", 
                    alpha = 1,
                    type.measure = "class")


# Plotting the misclassification error with the log of Lambda (??) 
plot(cv.out)


library(car)
VIF(glm(Survived~Pclass+Sex+Age+SibSp+Fare+Parch+Embarked,data=tf, family=binomial))


  ### The multi-collinearity
  
  #As we can see from the Variance Inflation Factor (VIF) check, all the predictors have low VIF, which means there is no reasonable multicollinearity; in the Lasso, there are four predictors that have the highest effect in the prediction of the model while there are some predictors is having a negligible impact. So our predictors are suitable to be used in the models.


  ### Estimate errors using bootstrap

library(tidyverse)
library(mosaic)

mean(~Survived, data=training)

Survived_bootstrap = mosaic::resample(Survived)

boot_Survived = do(100)*mean(~Survived, data=mosaic::resample(training))

#*Are we going to Estimate errors using bootstrap ?**
 # As we can see above, we will not get mean or standard errors because the Survived predictor is parameters of 0 and 1, and it is not a contumelious variable. So it is not reasonable to use the bootstrap in our case.

### Regressions with k-fold cross validation
# I choose **10-fold** cross-validation because it is popular to choose 5 or 10 folds

# to add the *10-fold* cross-validation I create the argument **trctrl** useing **trainControl** from caret backage whic is create Control parameters for train data, In our case the Control will be cross-validation with k=10


  # **Model 1 : Logistic Regression With CV**


# The Control  for the 10-fold CV
trctrl <- trainControl(method = "cv", number = 10 )

# The regression with the Control
Logit_CV <- train(Survived ~., 
                  data = training, 
                  method = "glm",
                  trControl=trctrl, # Applying the Control 
                  preProcess = c("center", "scale"),
                  tuneLength = 10,
                  family = "binomial"
)

Logit_CV

#  **Model 2 : LDA with CV**

#The Control 
trctrl <- trainControl(method = "cv", number = 10 )

# The regression with the Control
lda_CV <- train(Survived ~ .,
                data = training, 
                method = "lda",
                metric = "Accuracy",
                trControl = trctrl,  # Applying the Control
                preProcess = c("center", "scale")
)

lda_CV 


 #  **Model 3 : QDA with CV**

#The Control for the 10-fold CV 
trctrl <- trainControl(method = "cv", number = 10 )

# The regression with the Control
qda_CV <- train(Survived ~., 
                data = training, 
                method = "qda",
                trControl=trctrl, # Applying the Control
                preProcess = c("center", "scale"),
                tuneLength = 10)


qda_CV


#  **Model 4 : Naive Bayes with CV**

#The Control for the 10-fold CV 
trctrl <- trainControl(method = "cv", number = 10 )

# The regression with the Control
nb_CV <- train(Survived ~., data = 
                 training, 
               method = "naive_bayes",
               trControl=trctrl, # Applying the CV
               preProcess = c("center", "scale"),
               tuneLength = 10
)


nb_CV


#  **Model 5 : KNN with CV**

#The Control for the 10-fold CV 
trctrl <- trainControl(method = "cv", number = 10 )

# The regression with the Control
knn_cv <- train(Survived ~., 
                data = training, 
                method = "knn",
                trControl=trctrl,
                preProcess = c("center", "scale"),
                tuneLength = 10
)


knn_cv


These models are has good accuracy over all the logistic regression , lda ,QDA, NB and the KNN have more than 70% of the accuracy however that could improve after using the cross-valudation 



  ###Compare performance using ROC and AUC measures

#  **Model 1 : Logistic Regression ROC**

# Creating prediction using the Logistic Regression
myglmProb = predict(Logit_CV, training)
myglmProb
# create a vector of Drowned elements. Mark all as Drowned
myglmPred = rep("Drowned", 891) 
# convert to Up based on predicted probability > 0.5
myglmPred[myglmProb > .5] = "Survived" 

# create a confusion matrix
table(myglmPred, Survived) 
confusionMatrix(myglmProb, training$Survived )


#  **Model 2 : LDA ROC**

preds_xgb <- bind_cols(
  predict(lda_CV, newdata = testing, type = "prob"),
  Predicted = predict(lda_CV, newdata = testing, type = "raw"),
  Actual = testing$Survived
)

# Works
confusionMatrix(preds_xgb$Predicted, reference = preds_xgb$Actual)

#
mdl_auc <- Metrics::auc(actual = preds_xgb$Actual == "1", preds_xgb$`1`)
yardstick::roc_curve(preds_xgb, Actual, `1`) %>%
  autoplot() +
  labs(
    title = "LDA Model ROC Curve, Test Data",
    subtitle = paste0("AUC = ", round(mdl_auc, 4))
  )


#  **Model 3:  QDA ROC**

preds_qda <- bind_cols(
  predict(qda_CV, newdata = testing, type = "prob"),
  Predicted = predict(qda_CV, newdata = testing, type = "raw"),
  Actual = testing$Survived
)

# Works
confusionMatrix(preds_qda$Predicted, reference = preds_qda$Actual)

#
mdl_auc <- Metrics::auc(actual = preds_qda$Actual == "1", preds_qda$`1`)
yardstick::roc_curve(preds_qda, Actual, `1`) %>%
  autoplot() +
  labs(
    title = "QDA Model ROC Curve, Test Data",
    subtitle = paste0("AUC = ", round(mdl_auc, 4))
  )



#  **Model 3 :Naive Bayes ROC**

preds_nb <- bind_cols(
  predict(nb_CV, newdata = testing, type = "prob"),
  Predicted = predict(nb_CV, newdata = testing, type = "raw"),
  Actual = testing$Survived
)

# Works
confusionMatrix(preds_nb$Predicted, reference = preds_nb$Actual)

#
mdl_auc <- Metrics::auc(actual = preds_nb$Actual == "1", preds_nb$`1`)
yardstick::roc_curve(preds_nb, Actual, `1`) %>%
  autoplot() +
  labs(
    title = "QDA Model ROC Curve, Test Data",
    subtitle = paste0("AUC = ", round(mdl_auc, 4))
  )


#  **Model 3: KNN  ROC**

preds_knn <- bind_cols(
  predict(knn_cv, newdata = testing, type = "prob"),
  Predicted = predict(knn_cv, newdata = testing, type = "raw"),
  Actual = testing$Survived
)

# Works
confusionMatrix(preds_knn$Predicted, reference = preds_knn$Actual)

#
mdl_auc <- Metrics::auc(actual = preds_knn$Actual == "1", preds_knn$`1`)
yardstick::roc_curve(preds_knn, Actual, `1`) %>%
  autoplot() +
  labs(
    title = "QDA Model ROC Curve, Test Data",
    subtitle = paste0("AUC = ", round(mdl_auc, 4))
  )


  ##Step4
  
  ### Model Comparison
  which model are the best ? 
  From The Models Confusion Matrix Comparison and the ROC curve , we can see that the Linear Discriminant Analysis (LDA) model using 10-fold cross-validation is makes the best fit Accuracy in predicting the 30% training data from our 70% training model, so we will use it build the main model  to predict the future of the survivals in the survival rates for the Training titanic data passengers.


  ### Rebuilt the model Predict survival rates based on test data 
#  **Model LDA model using full train data with 10-kfol CV**

#The Control  for the CV
trctrl <- trainControl(method = "cv", number = 10 )



# The regression with the Control
lda_full <- train(Survived ~ .,
                  data = tf, 
                  method = "lda",
                  metric = "Accuracy",
                  trControl = trctrl,  # Applying the Control
                  preProcess = c("center", "scale")
)

lda_full 

predict the test data (tef) using the traingn data model with LDA model

Prediction <- predict(lda_full, newdata = tef)



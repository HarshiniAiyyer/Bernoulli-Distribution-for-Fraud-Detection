library(ranger)
library(caret)
library(data.table)
card_data <- read_excel("card_data.xls")
View(card_data)


dim(card_data)
head(card_data,6)
tail(card_data,6)
table(card_data$Class)
summary(card_data$Amount)
names(card_data)
var(card_data$Amount)
sd(card_data$Amount)


head(card_data)
card_data$Amount = scale(card_data$Amount)
newtable = card_data[,-c(1)]
head(newtable)

library(caTools)
set.seed(123)
sampleset = sample.split(newtable$Class,SplitRatio = 0.80)
train_data = subset(newtable,sampleset == TRUE)
test_data = subset(newtable,sampleset==FALSE)
dim(train_data)
dim(test_data)

logi_mod = glm(Class~.,test_data,family = binomial())
summary(logi_mod)
plot(logi_mod)
library(pROC)
lr.predict <- predict(logi_mod,test_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")

library(rpart)
library(rpart.plot)

dtree <- rpart(Class ~ . , card_data, method = 'class')
predicted_val <- predict(dtree, card_data, type = 'class')
probability <- predict(dtree, card_data, type = 'prob')
rpart.plot(dtree)

library(neuralnet)
ANN_model =neuralnet (Class~.,train_data,linear.output=FALSE)
plot(ANN_model)
predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)

library(gbm, quietly=TRUE)

# Get the time to train the GBM model
system.time(
  model_gbm <- gbm(Class ~. , distribution = "bernoulli" , data = rbind(train_data, test_data), n.trees = 500, interaction.depth = 3, n.minobsinnode = 100, shrinkage = 0.01, bag.fraction = 0.5, train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))))
# Determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")
model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
#Plot the gbm model

plot(model_gbm)
library(pROC)
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot = TRUE, col = "red")

print(gbm_auc)

library(caret)
library(doParallel)
library(gdm)

cl=makeCluster(6)
registerDoParallel(cl)

load('training.dat')

control=trainControl(method='repeatedcv',number=2,repeats=5,verboseIter=TRUE)
results=list()

#svm
results[[1]]=fit_svm_t=train(predictors,outcomes,method='svmRadial',
                             trControl=control,
                             preProcess=c('center','scale'), tuneLength=8)

#random forest
rf_tunegrid=expand.grid(mtry=seq(1,52,4))
results[[2]]=fit_rf_t<-train(predictors,outcomes,method="rf",
                             tuneGrid = rf_tunegrid,trControl=control)
#gbm
gbm_tunegrid=expand.grid(n.trees=c(300,400,500),shrinkage=c(0.1,0.2),interaction.depth=c(8,12,16))
results[[3]]=fit_gbm_t=train(predictors,outcomes,method='gbm',
                             preProcess=c('center','scale'), trControl=control,tuneGrid=gbm_tunegrid)

#lda
results[[4]]=fit_lda_t=train(predictors,outcomes,method='lda',
                             preProcess=c('center','scale'))

save(results,file='fitted_models.dat')

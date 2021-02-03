# nnm - an R package for building neural network models

## install
```
library(devtools)
install_bitbucket("chuanwen/nnm")
```
## example usage

```
library(nnm)
?nnm

# use nnm for logistic regression
x <- iris[, 1:4]
y <- iris[, 5]
mod <- nnm(x, y, list(Dense(4, 3, Activation.Identity), Softmax))
mod

# train a DNN for MNIST classification
mnist <- LoadMnist()
train <- mnist$train
test <- mnist$test
layerSpec <- Sequential(
  Dense(784, 128),
  Dropout(128, keepProb=0.8),
  Dense(128, 10, Activation.Identity),
  Softmax)
layerSpec
mod <- nnm(train$x, train$y, layerSpec, verbose=1, nEpoch=3)
# accuracy on train set
mean(mod$y == mod$fitted)
# accuracy on test set
mean(test$y == predict(mod, test$x, type="label"))
```
#' learning is a generic function
#'
#' @param object Object that supports this generic function
#' @param x input data, always assume there are multiple observations in x
#' @param y output data, always assume there are same number of observations as in x
#' @param weights weight of each observation, if missing, default 1 for each observation.
#' @param learningRate usually a small positive number, e.g. 0.1
#' @export
learning <- function(object, x, y, weights, learningRate, ...) UseMethod("learning")


#' A S3 class to represent a deep neuro net.
#'
#' @param layer a Layer object, which contains many layers. Usually you construct
# this Layer object using Sequential or Parallel or DAG.
#' @param loss a Loss object
#' @param type type of the network, for classification, it's predict Generic
#  accept extra parameter to output either "response" (probability) or "label".
#'
#' @examples
#' inputDim = 8
#' nObs = 100
#' x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
#' y = 0.1 * rnorm(nObs) + x[1, ]
#' layer1 = Parallel(Dense(4, 2, Activation.Identity), Dense(4, 2))
#' layer2 = Dense(4, 1, Activation.Identity)
#' layer = Sequential(layer1, layer2)
#' dnn = DNN(layer, MSELoss, type="regression")
#' for (i in 1:20) {
#'    dnn = learning(dnn, x, y)
#' }
#' dnn
#' plot(y, predict(dnn, x))
#'
#' @export
DNN <- function(layer, loss, type=c("regression", "classification", "other")) {
  type = match.arg(type)
  if (missing(loss)) {
    #loss = ifelse(type == "classification", EntropyLoss, MSELoss)
    loss = MSELoss
    if (type == "classification") {
      loss = EntropyLoss
    }
  }
  myclass = "DNN"
  if (type == "classification") {
    myclass = c("DNNClassifier", "DNN")
  }
  structure(list(layer=layer, loss=loss, type=type), class=myclass)
}


#' @describeIn DNN strip the net to only keep what is needed for Forward method.
#' @export
Strip.DNN <- function(object) {
  object$layer = Strip(object$layer)
  invisible(object)
}

#' @export
print.DNN <- function(x, ...) {
  cat("DNN, type =", x$type, "\n")
  print(x$loss, ...)
  print(x$layer, ...)
}

#' @describeIn DNN implement generic Forward function
#' @export
Forward.DNN <- function(object, x, ...) {
  object$layer = Forward(object$layer, x, ...)
  object$a = object$layer$a
  invisible(object)
}

#' @describeIn DNN implement  DNNBackward function.
#' @param x numeric matrix of dimension p x nobs, i.e. each col is a predictor vector.
#' @param y a numeric or factor vector, or a numeric matrix of dimension
#' q x nobs, i.e. each col is a response vector. If y is a vector, it will be
#' converted to a matrix (nrow=1).
# vector.
#' @export
Backward.DNN <- function(object, y, weights) {
  if (class(y) == "factor" && is.null(object$levels)) {
    object$levels = levels(y)
  }
  if (!is.matrix(y)) {
    y = matrix(as.numeric(y), nrow=1)
  }
  stopifnot(ncol(object$x) == ncol(y))
  if (missing(weights)) {
    weights = rep(1, ncol(object$x))
  }
  yhat <- object$a
  object$totLoss <- object$loss$f(y, yhat, weights)
  errorOut <- object$loss$errorf(y, yhat, weights)
  object$layer = Backward(object$layer, errorOut)
  object$errorIn = object$layer$errorIn
  invisible(object)
}

#' @describeIn DNN implement generic UpdateParameters function.
#' @export
UpdateParameters.DNN <- function(object, learningRate) {
  object$layer = UpdateParameters(object$layer, learningRate)
  invisible(object)
}

#' @describeIn DNN implement generic learning function.
#' @param object an DNN object
#' @param x numeric matrix of dimension p x nobs, i.e. each col is a predictor vector.
#' @param y a numeric or factor vector, or a numeric matrix of dimension
#' q x nobs, i.e. each col is a response vector. If y is a vector, it will be
#' converted to a matrix (nrow=1) in Backward.DNN.
#' @export
learning.DNN <- function(object, x, y, weights, learningRate, ...) {
  if (missing(weights)) {
    weights = rep(1, ncol(x))
  }
  if (missing(learningRate)) {
    learningRate = 0.2
  }
  object <- Forward.DNN(object, x)
  object <- Backward.DNN(object, y, weights)
  if (learningRate > 0) {
    object <- UpdateParameters.DNN(object, learningRate)
  }
  invisible(object)
}

#' @describeIn DNN prediction of Network
#' @export
predict.DNN <- function(object, newdata, ...) {
  object <- Forward(object, newdata, stage="predict")
  t(object$layer$a)
}

#' @describeIn DNN prediction of class probability or labels when last layer is softmax
#' @export
predict.DNNClassifier <- function(object, newdata, type = c("response", "label")) {
  type = match.arg(type)
  object <- Forward(object, newdata, stage="predict")
  ans <- object$a
  if (type == "label") {
    ans <- apply(ans, 2, which.max)
    if (!is.null(object$levels)) {
      ans <- factor(object$levels[ans], object$levels)
    }
  }
  ans
}

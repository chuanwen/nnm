#' A S3 class to represent Dense (full-connect) neuro layer.
#'
#' @importFrom truncnorm rtruncnorm
#'
#' @param inputDim integer, refers to the dimension of one observation.
#' @param outputDim integer, refers to the dimension of one output.
#' @param activation activation to be used for the neuro, default ReLU()
#' @examples
#' layer = Dense(inputDim=3, outputDim=5)
#' layer
#' names(layer)
#' layer$W # weight parameters
#' layer$b # bias parameters
#'
#' x = array(rnorm(3*2), dim=c(3,2))
#' errorOut=array(rnorm(10), dim=c(5,2))
#'
#' layer = Forward(layer, x)
#' layer$a # output
#'
#' layer = Backward(layer, errorOut)
#' layer$errorIn
#'
#' layer = UpdateParameters(layer, 0.1)
#' layer$b
#' @export
Dense <- function(inputDim, outputDim, activation = Activation.ReLU,
				  initWeightScale = 0.01, initBias = 0) {
  # (Fully-connected)Neuro Layer constructor.
  row <- outputDim
  col <- inputDim
  W = matrix(rtruncnorm(row * col, -2, 2, sd=initWeightScale), nrow = row)
  b = rep(initBias, row)
  structure(list(
         W = W,
         b = b,
		 inputDim = inputDim,
		 outputDim = outputDim,
		 activation = activation,
         learner = list(b=SGD(b), W=SGD(W)),
         done = "init"), class = c("Dense", "Layer"))
}

#' @export
Strip.Dense <- function(object) {
  structure(object[c("inputDim", "outputDim", "W", "b", "activation", "learner")],
            class = c("Dense", "Layer"))
}

#' @export
print.Dense <- function(x, left="", ...) {
  cat(left, "Dense:", x$inputDim, "->", x$outputDim,
	  x$activation$name, "\n")
}

#' @export
Forward.Dense <- function(object, x, ...) {
  object$x <- x
  # W %*% x has shape outputDim x nExamples, b is vector of length outputDim.
  # use sweep to add scale b[i] to vector row[i], i = 1,..,outputDim
  object$z <- sweep(object$W %*% x, 1, object$b, "+")
  object$a <- object$activation$f(object$z)
  object$done <- "Forward"
  invisible(object)
}

#' @export
Backward.Dense <- function(object, errorOut) {
  object$delta <- errorOut * object$activation$df(object$z, object$a)
  object$errorIn <- t(object$W) %*% object$delta
  object$done <- "Backward"
  invisible(object)
}

#' @export
UpdateParameters.Dense <- function(object, learningRate) {
  nExamples <- ncol(object$delta)
  if (is.null(object$learner)) {
    object$learner = list(W = SGD(object$W), b = SGD(object$b))
  }
  object$b <- object$learner$b(object$b, learningRate * apply(object$delta, 1, mean))
  object$W <- object$learner$W(object$W, (learningRate/nExamples) * (object$delta %*% t(object$x)))
  object$done <- "UpdateParameters"
  invisible(object)
}

#' @export
InitParameters.Dense <- function(object, initWeightScale=0.01, initBias=0.0) {
  row <- length(object$b)
  object$W = matrix(rtruncnorm(length(object$W), -2, 2, initWeightScale), nrow = row)
  object$b = rep(initBias, row)
	invisible(object)
}

#' @export
NumParameters.Dense <- function(object) {
  return(length(object$W) + length(object$b))
}
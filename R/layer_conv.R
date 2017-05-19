#' A S3 class to represent Convoluation neuro layer.
#'
#' @importFrom Rcpp evalCpp
#' @useDynLib nnm, .registration = TRUE
#'
#' @param inputDim a size 3 vector to indicate the dimension of one observation.
#' @param kernelDim a size 4 vector to indicate the dimension of kernel
#' @param activation activation to be usaed for the neuro, default ReLU()
#' @examples
#' inputDim = c(5, 5, 1)
#' outputDim = c(5, 5, 2)
#' kernelDim = c(3, 3, 1, 2)
#' layer = Conv(inputDim=inputDim, kernelDim=kernelDim)
#' layer
#'
#' nObs = 1
#'
#' x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
#' errorOut=array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))
#'
#' layer = Forward(layer, x)
#' layer$a # output
#'
#' layer = Backward(layer, errorOut)
#' layer$errorIn
#' @export
Conv <- function(inputDim, kernelDim, activation = Activation.ReLU,
				  initWeightScale = 0.1, initBias = 0.01) {
  # Convolution Neuro Layer constructor.
  WeightSampler <- function(n, scale) {
	truncatedAt = 2 * scale
    x <- rnorm(n, sd = scale)
    ind <- abs(x) > truncatedAt
    x[ind] <- truncatedAt * sign(x[ind])
    x
  }
  outputDim = c(inputDim[1:2], kernelDim[4])
  kernel = array(WeightSampler(prod(kernelDim), initWeightScale), dim=kernelDim)
  b = rep(initBias, kernelDim[4])
  structure(list(
         kernel = kernel,
         kernelInv = array(0.0, dim=dim(kernel)[c(1,2,4,3)]),
				 b = b,
				 inputDim = inputDim,
         outputDim = outputDim,
         kernelDim = kernelDim,
         H = (kernelDim[1]-1)/2,
         W = (kernelDim[2]-1)/2,
				 activation = activation,
         learner = list(kernel = SGD(kernel), b = SGD(b)),
         done = "init"),
			class = c("Conv", "Layer"))
}

#' @export
Strip.Conv <- function(object) {
  structure(object[c("inputDim", "outputDim", "kernel",
    "kernelInv", "b", "H", "W", "activation", "kernel")], class = c("Conv", "Layer"))
}

#' @export
print.Conv <- function(x, left="", ...) {
  str = function(vec) { paste(vec, collapse=",") }
  description = sprintf("Convoluation: (%s) * (%s) --> %s \t%s\n",
    str(x$inputDim), str(x$kernelDim), str(x$outputDim), x$activation$name)
  cat(left, description)
}

#' @export
Forward.Conv <- function(object, x, ...) {
  object$x = x;
  zDim = c(object$outputDim, dim(x)[4])
  object$z = array(0.0, dim=zDim)
  conv3d(x, object$kernel, object$z, "forward")
  object$z = sweep(object$z, 3, object$b, "+")
  object$a <- object$activation$f(object$z)
  object$done <- "Forward"
  invisible(object)
}

#' @export
Backward.Conv <- function(object, errorOut) {
  object$delta <- errorOut * object$activation$df(object$z, object$a)
  object$errorIn <- array(0.0, dim=dim(object$x))
  convFlip(object$kernel, object$kernelInv)
  conv3d(object$delta, object$kernelInv, object$errorIn, "backward")
  object$done <- "Backward"
  invisible(object)
}

#' @export
UpdateParameters.Conv <- function(object, learningRate) {
  if (is.null(object$learner)) {
    object$learner = list(kernel = SGD(object$kernel), b = SGD(object$b))
  }
  nExamples <- ncol(object$delta)
  object$b <- object$learner$b(object$b, learningRate * apply(apply(object$delta, c(3,4), sum), 1, mean))
  object$kernel <- object$learner$kernel(object$kernel, (learningRate/nExamples)*sumConvInv(object$delta, object$x, object$H, object$W))
  object$done <- "UpdateParameters"
  invisible(object)
}

#' @export
InitParameters.Conv <- function(object, initWeightScale = 0.1, initBias = 0.01) {
	WeightSampler <- function(n, scale) {
	truncatedAt = 2 * scale
    x <- rnorm(n, sd = scale)
    ind <- abs(x) > truncatedAt
    x[ind] <- truncatedAt * sign(x[ind])
    x
  }
	inputDim = object$inputDim
	kernelDim = object$kernelDim
  outputDim = c(inputDim[1:2], kernelDim[4])
  object$kernel = array(WeightSampler(prod(kernelDim), initWeightScale), dim=kernelDim)
  object$b = rep(initBias, kernelDim[4])
	invisible(object)
}

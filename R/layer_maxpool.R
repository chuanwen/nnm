#' A S3 class to represent Maxpool layer.
#'
#' @param inputDim a size 3 vector to indicate the dimension of one observation.
#' @param kernelDim a size 2 vector to indicate the dimension of kernel
#' @examples
#' inputDim = c(4, 4, 1)
#' kernelDim = c(2,2)
#' layer = MaxPool(inputDim=inputDim, kernelDim=kernelDim)
#' layer
#'
#' outputDim = layer$outputDim
#' nObs = 1
#' x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
#' errorOut = array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))
#'
#' layer = Forward(layer, x)
#' layer$a # output
#'
#' layer = Backward(layer, errorOut)
#' layer$errorIn
#' @export
MaxPool <- function(inputDim, kernelDim) {
  outputDim = inputDim / c(kernelDim, 1)
  structure(list(inputDim = inputDim, kernelDim = kernelDim, outputDim = outputDim),
            class = c("MaxPool", "Layer"))
}

#' @export
Strip.MaxPool <- function(object) {
  structure(object[c("inputDim", "outputDim", "kernelDim")], class = c("MaxPool", "Layer"))
}

#' @export
Forward.MaxPool <- function(object, x, ...) {
  object$x = x
  H = object$kernelDim[1]
  W = object$kernelDim[2]
  object$a = MaxPoolForwardC(x, H, W)
  invisible(object)
}

#' @export
Backward.MaxPool <- function(object, errorOut) {
  H = object$kernelDim[1]
  W = object$kernelDim[2]
  object$errorIn = MaxPoolBackwardC(object$x, object$a, errorOut)
  invisible(object)
}

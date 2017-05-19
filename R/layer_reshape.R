#' A S3 class to represent Reshape layer.
#'
#' @param inputDim an integer or vector to indicate the dimension of one input observation.
#' @param outputDim an integer or vector to indicate the dimension of one output observation
#' @examples
#' layer = Reshape(9, c(3, 3, 1))
#' layer
#'
#' nObs = 1
#' x = array(rnorm(9), c(9, 1))
#' layer = Forward(layer, x)
#' layer$x # input
#' layer$a # output
#' @export
Reshape <- function(inputDim, outputDim) {
  stopifnot(prod(inputDim) == prod(outputDim))
  structure(list(inputDim = inputDim, outputDim = outputDim), class = c("Reshape", "Layer"))
}

#' @export
Strip.Reshape <- function(object) {
  structure(object[c("inputDim", "outputDim")], class = c("Reshape", "Layer"))
}

#' @export
Forward.Reshape <- function(object, x, ...) {
  nObs = dim(x)[length(dim(x))]
  object$x = x
  object$a = array(x, dim=c(object$outputDim, nObs))
  invisible(object)
}

#' @export
Backward.Reshape <- function(object, errorOut) {
  object$errorIn = array(errorOut, dim=dim(object$x))
  invisible(object)
}

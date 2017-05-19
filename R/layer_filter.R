#' A S3 class to represent Filter layer.
#'
#' @param filter a logical array to indicate which element can pass.
#' @examples
#' layer = Filter(3, c(TRUE, TRUE, FALSE))
#' layer
#'
#' nObs = 1
#' x = array(rnorm(3), c(3, 1))
#' layer = Forward(layer, x)
#' layer$x # input
#' layer$a # output
#' @export
Filter <- function(inputDim, filter, outputDim) {
  stopifnot("logical" %in% class(filter))
  if (missing(outputDim)) {
    exampleInput = array(0, dim=c(inputDim, 1))
    outputDim = length(exampleInput[filter])
  }
  structure(list(inputDim=inputDim, outputDim=outputDim, filter=filter),
            class = c("Filter", "Layer"))
}

#' @export
Strip.Filter <- function(object) {
  structure(object[c("inputDim", "outputDim", "filter")], class = c("Filter", "Layer"))
}

#' @export
Forward.Filter <- function(object, x, ...) {
  nObs = dim(x)[length(dim(x))]
  object$x = x
  object$a = array(x[object$filter], dim=c(object$outputDim, nObs))
  invisible(object)
}

#' @export
Backward.Filter <- function(object, errorOut) {
  object$errorIn = array(0.0, dim=dim(object$x))
  object$errorIn[object$filter] = errorOut
  invisible(object)
}

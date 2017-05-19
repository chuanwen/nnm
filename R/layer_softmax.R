#' A S3 class to represent Softmax layer.
#'
#' @param numClasses an integer to indicate the number of classes.
#' @examples
#' layer = Softmax(3)
#' layer
#'
#' nObs = 2
#' x = array(rnorm(3*nObs), c(3, nObs))
#' layer = Forward(layer, x)
#' layer$x # input
#' layer$a # output
#' @export
Softmax <- function(numClasses) {
  # Softmax Layer constructor
  structure(list(inputDim = numClasses, numClasses = numClasses, outputDim = numClasses),
            class = c("Softmax", "Layer"))
}

#' @export
Strip.Softmax <- function(object) {
  Softmax(object$numClasses)
}

#' @export
print.Softmax <- function(x, left="", ...) {
  cat(left, "Softmax, numClasses = ", x$numClasses, "\n")
}

#' @export
Forward.Softmax <- function(object, x, ...) {
  # For each col, minus maximum value of the col.
  object$x <- sweep(x, 2, apply(x, 2, max))
  object$a <- exp(object$x)
  # For each col, normalize such that summation is 1.
  object$a <- sweep(object$a, 2, apply(object$a, 2, sum), "/")
  object$done <- "Forward"
  invisible(object)
}

#' @export
Backward.Softmax <- function(object, errorOut) {
  object$q <- apply(errorOut * object$a, 2, sum)
  object$errorIn <- object$a * sweep(errorOut, 2, object$q, "-")
  object$done <- "Backward"
  invisible(object)
}

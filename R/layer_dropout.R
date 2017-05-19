#' A S3 class to represent Dropout layer.
#'
#' @param inputDim an integer or vector to indicate dimension of one input.
#' @param keepProb probability to keep a neuron.
#' @examples
#' layer = Dropout(4, keepProb=0.5)
#' layer
#'
#' nObs = 2
#' x = array(rnorm(4*nObs), c(4, nObs))
#' x
#'
#' layer = Forward(layer, x) # Forward for training
#' layer$a
#'
#' layer = Forward(layer, x, stage="inference") # Forward for inference
#' layer$a
#' @export
Dropout <- function(inputDim, keepProb=0.5) {
  structure(list(inputDim = inputDim, outputDim = inputDim, keepProb=keepProb), class = c("Dropout", "Layer"))
}

#' @export
Strip.Dropout <- function(object) {
  structure(object[c("inputDim", "outputDim", "keepProb")], class = c("Dropout", "Layer"))
}

#' @export
print.Dropout <- function(x, left="", ...) {
  cat(left, "Dropout, keepProb = ", x$keepProb, "\n")
}

#' @export
Forward.Dropout <- function(object, x, stage="training", random=TRUE, ...) {
  object$x = x
  if (stage == "training") {
    if (random || length(object$maskIndicator) != length(x)) {
      object$maskIndicator = array(runif(length(x)) > object$keepProb, dim=dim(x))
    }
    object$a = object$x
    object$a[object$maskIndicator] = 0
  } else {
    object$maskIndicator = NULL
    object$a = object$x * object$keepProb
  }
  invisible(object)
}

#' @export
Backward.Dropout <- function(object, errorOut) {
  if (length(object$maskIndicator) == length(object$x)) {
    object$errorIn = errorOut
    object$errorIn[object$maskIndicator] = 0
  }
  invisible(object)
}

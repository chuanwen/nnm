#' Generic Forward function
#'
#' @param object Object that supports Forward function
#' @param x input data, always assume there are multiple observations in x
#' @param ... other optional parameter for specail layer like Dropout
#' @return a layer object with done = "Forward"
#' @export
Forward <- function(object, x, ...) UseMethod("Forward")

#' Generic Backward function
#'
#' @param object Object that supports Backward function
#' @param errorOut partial Cost / partial out
#' @return a layer object with done = "Backward"
#' @export
Backward <- function(object, errorOut) UseMethod("Backward")


#' Generic UpdateParameters function
#'
#' @param object Object that supports UpdateParameters function
#' @param learningRate float, between 0 and 1.0
#' @return a layer object with updated parameters (if applicable)
#' @export
UpdateParameters <- function(object, learningRate) UseMethod("UpdateParameters")

#' Generic InitParameters function
#'
#' @param object Object that supports InitParameters function
#' @param ... other arguments
#' @export
InitParameters <- function(object, ...) UseMethod("InitParameters")

#' Generic Strip function
#'
#' @param object Object that supports Strip function
#' @return a layer object where anything that is not needed for inference are removed.
#' @export
Strip <- function(object) UseMethod("Strip")

#' Default implementation of UpdateParameters generic function
#' @export
UpdateParameters.default <- function(object, learningRate) {
  invisible(object)
}

#' Default implementation of InitParameters generic function
#' @export
InitParameters.default <- function(object, ...) {
  invisible(object)
}

#' export
print.Layer <- function(x, left="", ...) {
  cat(left, class(x)[1], x$inputDim, "->", x$outputDim, "\n")
}

#' Default implementation of Strip generic function
#' @export
Strip.default <- function(object) {
  invisible(object)
}

#' @export
Identity <- function(inputDim) {
  structure(list(inputDim=inputDim, outputDim=inputDim), class=c("Identity", "Layer"))
}

#' @export
Forward.Identity <- function(object, x, ...) {
  object$x = x
  object$a = x
  invisible(object)
}

#' @export
Backward.Identity <- function(object, errorOut) {
  object$errorIn = errorOut
  invisible(object)
}

numericBackward <- function(object, errorOut, d = 1e-5, ...) {
  n = length(object$x)
  errorIn = array(0.0, dim=dim(object$x))
  for (i in 1:n) {
    x1 = object$x
    x2 = object$x
    x1[i] = x1[i] - d
    x2[i] = x2[i] + d
    layer1 = Forward(object, x1, ...)
    layer2 = Forward(object, x2, ...)
    errorIn[i] = sum((layer2$a - layer1$a) * errorOut)/(2*d)
  }
  object$errorIn = errorIn
  invisible(object)
}

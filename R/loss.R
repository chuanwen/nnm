# Loss class #
#' @export
Loss <- function(f, errorf, name) {
  # Loss constructor.
  #
  # Args:
  #   f: loss function, with signature f(y, yhat)
  #   errorf: partial derivative of f with regard to yhat.
  #
  # Returns:
  #   Loss object.
  structure(list(f = f, errorf = errorf, name = name), class = "Loss")
}

#' @export
print.Loss <- function(x, ...) {
  cat("Loss: ", x$name, "\n")
}

EPS <- 1e-05

#' @export
EntropyLoss <- (function() {
  f <- function(y, yhat, weights) {
	# Computes entropy loss for classification learning.
	#
	# Args:
  #   y: integer vector with length = nExamples.
  #   yhat: probability matrix of shape numClasses x nExamples
  #
  # Returns:
  #   sum of entropy loss for the nExamples.
    nExamples <- ncol(yhat)
    nClasses <- nrow(yhat)
    if (missing(weights)) {
      weights = rep(1, nExamples)
    }
    if (!is.matrix(y) || nrow(y) == 1) {
      index <- nClasses * (0:(nExamples - 1)) + as.integer(y)
      -sum(weights * log(yhat[index]))
    } else {
      -sum(apply(y*log(yhat), 2, sum) * weights)
    }
  }
  errorf <- function(y, yhat, weights) {
    nExamples <- ncol(yhat)
    nClasses <- nrow(yhat)
    if (missing(weights)) {
      weights = rep(1, nExamples)
    }
    if (!is.matrix(y) || nrow(y) == 1) {
      index <- nClasses * (0:(nExamples - 1)) + as.integer(y)
      ans <- matrix(0, nrow = nrow(yhat), ncol = ncol(yhat))
      ans[index] <- -weights/(yhat[index] + EPS)
      ans
    } else {
      -sweep(y/(yhat+EPS), 2, weights, "*")
    }
  }
  Loss(f, errorf, "entropy")
})()

#' @export
MSELoss <- (function() {
  f <- function(y, yhat, weights) {
    if (missing(weights)) {
      weights = rep(1, length(y))
    }
    sum(weights*(y - yhat)^2)
  }
  errorf <- function(y, yhat, weights) {
    if (missing(weights)) {
      weights = rep(1, length(y))
    }
    2.0 * weights * (yhat - y)
  }
  Loss(f, errorf, "MSE")
})()

#' @export
PoissonLoss <- (function() {
  f <- function(y, yhat, weights) {
    if (missing(weights)) {
      weights = rep(1, length(y))
    }
    sum(weights * (yhat - y * log(yhat)))
  }
  errorf <- function(y, yhat, weights) {
    if (missing(weights)) {
      weights = rep(1, length(y))
    }
    weights * (1 - y/yhat)
  }
  Loss(f, errorf, "Poisson")
})()

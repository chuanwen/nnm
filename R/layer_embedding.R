EmbeddingOne <- function(inputDim, inputlevels, embeddingDim,
                      initWeightScale=0.5) {
  stopifnot(inputDim == 1 &&
            length(inputlevels) == 1 &&
            length(embeddingDim) == 1)
  row = embeddingDim
  col = inputlevels
  W = matrix(rtruncnorm(row*col, -2, 2, sd=initWeightScale), nrow=row)
  structure(list(inputDim=inputDim, outputDim=embeddingDim, W=W,
                 inputlevels=inputlevels,
                 learner = list(W=SGD(W))),
            class=c("EmbeddingOne", "Layer"))
}

#' @export
print.EmbeddingOne <- function(x, left="", ...) {
  cat(left, "Embedding:", x$inputDim, "->", x$outputDim, "\n")
}

Strip.EmbeddingOne <- function(object) {
  structure(object[c("inputDim", "outputDim", "inputlevels", "W", "learner")],
            class = c("EmbeddingOne", "Layer"))
}

Forward.EmbeddingOne <- function(object, x) {
  x = factor(as.integer(x), 1:object$inputlevels)
  object$x = x
  object$a = object$W[, x]
  invisible(object)
}

Backward.EmbeddingOne <- function(object, errorOut) {
  x = object$x
  oneHot = model.matrix(~0+x, x, contrasts.arg=list(x=diag(object$inputlevels)))
  # adding 0.01 is to handle zero columns
  object$delta = sweep(errorOut %*% oneHot, 2, 0.01+apply(oneHot, 2, sum), "/")
  object$errorIn = rep(0.0, length(object$x))
  invisible(object)
}

UpdateParameters.EmbeddingOne <- function(object, learningRate) {
  if (learningRate > 0) {
    if (is.null(object$learner)) {
      object$learner = list(W = SGD(object$W))
    }
    object$W <- object$learner$W(object$W, learningRate * object$delta)
  }
  invisible(object)
}

#' @export
Embedding <- function(inputDim, inputlevels, embeddingDims,
                      initWeightScale=0.5) {
  stopifnot(inputDim == length(inputlevels))
  if (inputDim == 1) {
    return(EmbeddingOne(1, inputlevels, embeddingDims[1]))
  }

  if (inputDim != length(embeddingDims)) {
    stopifnot(inputDim %% length(embeddingDims) == 0)
    embeddingDims = rep(embeddingDims, inputDim/length(embeddingDims))
  }

  layers = lapply(1:inputDim, function(i){
      EmbeddingOne(1, inputlevels[i], embeddingDims[i], initWeightScale)
  })

  do.call(Parallel, layers)
}

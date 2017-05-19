#' neuron network model. It's a high level wrap of nnm.fit
#'
#' @importFrom stats  complete.cases predict rnorm runif model.matrix
#'
#' @param x matrix or data.frame with numeric, factor or character cols.
#' Each col is a variable or predictor, each row is an observation.
#' @param y response variable (vector or matrix). Quantitative for
#' ‘family="gaussian"’, or ‘family="poisson"’ (non-negative counts).
#' For ‘family="multinomial"’, should be either a factor/character vector, or
#' a matrix of N-Classes cols.
#' @param layerSpecs either a list of layer specs, or a (composite) layer
#' objectect (e.g. a Sequential or DAG layer). See \code{\link{Sequential}}
#' for how to specify a sequential of layers,
#' @param embeddingCols specify factor cols to be embedded. Factor cols
#' that are not embeded would be encoded using default contrast matrix.
#' @param embeddingDims specify the embedding dims, if length(embeddingDims) is
#' only a fraction of length(embeddingCols), then
#' embeddingDims would be repeated to the same length of embeddingCols.
#' @param weights weight of the obs, default 1 for each and every observation.
#' @param family Response type, should be one of c("gaussian", "poisson",
#' "multinomial"). If not specified, will be inferred from class(y).
#' @param nEpoch number of times to train over the whole data, default 5.
#' @param batchSize number of observations per train batch, default 100.
#' @param learningRate a float between 0 and 1, default 0.25.
#'
#' @examples
#' x = iris[, 1:4]
#' y = iris[, 5]
#' mod = nnm(x, y, list(Dense(4, 3, Activation.Identity), Softmax))
#' mod
#'
#' # example usage when x has factor cols
#' y = iris[, 1]
#' x = iris[, 2:5]
#' layerSpecs = list(Dense(3+nlevels(x$Species), 2), Dense(2, 1, Activation.Identity))
#' mod = nnm(x, y, layerSpecs)
#' mod
#'
#' # example usage of embedding
#' y = iris[, 1]
#' x = iris[, 2:5]
#' embeddingCols = "Species"
#' embeddingDims = 2
#' layerSpecs = list(Dense(3+embeddingDims, 2), Dense(2, 1, Activation.Identity))
#' mod = nnm(x, y, layerSpecs, embeddingCols, embeddingDims)
#' mod
#'
#' # example model for MNIST data
#' mnist <- LoadMnist("/home/ccw/learning/data/mnist")
#' train <- mnist$train
#' test <- mnist$test
#' mod = nnm(train$x, train$y, list(Dense(784, 15),
#'   Dense(15, 10, Activation.Identity), Softmax), verbose=1)
#' # accuracy on train set
#' mean(mod$y == mod$fitted)
#' # accuracy on test set
#' mean(test$y == predict(mod, test$x, type="label"))
#' @export
nnm <- function(x, y, layerSpecs, embeddingCols, embeddingDims, weights,
                family, ...) {
  # if the data is already good for nnm.fit, no need any work.
  if (is.matrix(x) && class(x[1,1]) == "numeric" ||
      is.data.frame(x) && all(sapply(x, class) == "numeric")) {
    stopifnot(missing(embeddingCols) && missing(embeddingDims))
    return(nnm.fit(x, y, layerSpecs, weights, family, ...))
  }

  if (is.null(dim(x))) {
    stop("x should be a data.frame or a matrix")
  }

  if (class(y) == "data.frame" && ncol(y) >= 2) {
    stopifnot(all(sapply(y, class) == "numeric"))
  }

  if (missing(weights)) {
      weights = rep.int(1, nrow(x))
  }

  ind = complete.cases(x, y, weights)
  if (sum(ind) != length(ind)) {
    x = x[ind, ]
    y = subset(y, ind)
    weights = weights[ind]
  }

  if (is.matrix(x)) {
    x = as.data.frame(x)
  }

  xlevels = lapply(x, function(col) {
    if (class(col) == "character") {
      return(unique(sort(col)))
    } else if(class(col) == "factor") {
      return(levels(col))
    } else if(class(col) == "numeric") {
      return(0)
    }
  })

  if (!missing(embeddingCols)) {
    if (is.character(embeddingCols)) {
      stopifnot(all(embeddingCols %in% names(x)))
      embeddingCols = which(embeddingCols==names(x))
    }
  } else {
    embeddingCols = NULL
  }

  numericDF <- function(x, xlevels, embeddingCols) {
    stopifnot(class(x) == "data.frame")
    xclass = sapply(x, class)
    stopifnot(all(xclass %in% c("numeric", "factor", "character")))
    if ("character" %in% xclass) {
      x <- sapply(1:ncol(x), function(i) {
        if (xclass[i] == "character") {
          factor(x[, i], xlevels[[i]])
        } else {
          x[, i]
        }
      })
    }
    if (is.null(embeddingCols)) {
      model.matrix(~0+., x)
    } else {
      em = sapply(x[, embeddingCols, drop=FALSE], as.integer)
      x = x[, -embeddingCols, drop=FALSE]
      if (ncol(x) > 0) {
        x = model.matrix(~0+., x)
      }
      cbind(x, em)
    }
  }

  x = numericDF(x, xlevels, embeddingCols)
  inputlevels = sapply(xlevels[embeddingCols], length)
  regularDims = ncol(x) - length(embeddingCols)

  if (length(layerSpecs) > 1) {
    layerSpecs = do.call(Sequential, layerSpecs)
  }

  if (regularDims == 0) {
    layer0 = Embedding(length(embeddingCols), inputlevels, embeddingDims)
    layerSpecs = Sequential(layer0, layerSpecs)
  } else if (regularDims < ncol(x)) {
    layer0a = Identity(regularDims)
    layer0b = Embedding(length(embeddingCols), inputlevels, embeddingDims)
    layerSpecs = Sequential(Parallel(layer0a, layer0b), layerSpecs)
  }

  fit = nnm.fit(x, y, layerSpecs, weights, family, ...)
  structure(list(fit=fit, fitted=fit$fitted, xlevels=xlevels,
                 embeddingCols=embeddingCols, numericDF=numericDF),
            class="nnm")
}

#' @export
print.nnm <- function(x, ...) {
    if (sum(x$embeddingDims) > 0) {
      cat("EmbeddingCols", x$embeddingCols, "embeddingDims", x$embeddingDims, "\n")
    }
    print.nnm.fit(x$fit, ...)
}

#' @export
predict.nnm <- function(object, x, type=c("response", "label"), ...) {
    type = match.arg(type)
    x = object$numericDF(as.data.frame(x), object$xlevels, object$embeddingCols)
    predict.nnm.fit(object$fit, x, type=type, ...)
}

#' fit a (deep) neuron network model. It's a wrap of the DNN class.
#'
#' @importFrom stats  complete.cases predict rnorm runif
#'
#' @param x numeric matrix, of dimension nobs x nvars; each row is an observation vector.
#' @param y response variable (vector or matrix). Quantitative for
#' ‘family="gaussian"’, or ‘family="poisson"’ (non-negative counts).
#' For ‘family="multinomial"’, should be either a factor/character vector, or
#' a matrix of N-Classes cols.
#' @param layerSpecs either a list of layer specs, or a (composite) layer
#' objectect (e.g. a Sequential or DAG layer). See \code{\link{Sequential}}
#' for how to specify a sequential of layers,
#' @param weights weight of the obs, default 1 for each and every observation.
#' @param family Response type, should be one of c("gaussian", "poisson",
#' "multinomial"). If not specified, will be inferred from class(y).
#' @param nEpoch number of times to train over the whole data, default 5.
#' @param batchSize number of observations per train batch, default 100.
#' @param learningRate a float between 0 and 1, default 0.25.
#' @param verbose default 0, if >= 1 will print out loss during training.
#'
#' @examples
#' x = iris[, 1:4]
#' y = iris[, 5]
#' mod = nnm.fit(x, y, list(Dense(4, 3, Activation.Identity), Softmax))
#' mod
#'
#' # example model for MNIST data
#' mnist <- LoadMnist("/home/ccw/learning/data/mnist")
#' train <- mnist$train
#' test <- mnist$test
#' mod = nnm.fit(train$x, train$y, list(Dense(784, 15),
#'   Dense(15, 10, Activation.Identity), Softmax), nEpoch=1)
#' # accuracy on train set
#' mean(mod$y == mod$fitted)
#' # accuracy on test set
#' mean(test$y == predict(mod, test$x, type="label"))
#' @export
nnm.fit <- function(x, y, layerSpecs, weights,
                    family=c("gaussian", "poisson", "multinomial"),
                    nEpoch=5, batchSize=100, learningRate=0.25, verbose=0)
{
  if (is.null(dim(x))) {
    stop("x should be a data.frame or matrix")
  }

  if (class(y) == "data.frame" && ncol(y) >= 2) {
    stopifnot(all(sapply(y, class) == "numeric"))
  }

  if (missing(weights)) {
      weights = rep.int(1, nrow(x))
  }

  ind = complete.cases(x, y, weights)
  if (sum(ind) != length(ind)) {
    x = x[ind, ]
    y = subset(y, ind)
    weights = weights[ind]
  }

  if (missing(family)) {
      family = switch(class(y),
        factor = "multinomial",
        character = "multinomial",
        numeric = "gaussian",
        integer = "poisson",
        data.frame = "gassuian",
        matrix = "gassuian"
      )
  } else {
      family = match.arg(family)
  }

  if (family == "multinomial") {
    if (is.null(dim(y))) { # y is not matrix or data.frame
      if (class(y) != "factor") {
        y = as.factor(y)
      }
    } else if (ncol(y) == 1) {
        y = as.factor(y[, 1])
    } else {
        stopifnot(all(y>=0) && nrow(y) == nrow(x))
        y = as.matrix(y)
    }
  }

  nvars = ncol(x)
  nobs = nrow(x)

  loss = switch(family,
    gaussian = MSELoss,
    poisson = PoissonLoss,
    multinomial = EntropyLoss
  )

  type = if(family == "multinomial") "classification" else "regression"

  if ("Layer" %in% class(layerSpecs)) {
    layer = layerSpecs
  } else {
    layer = do.call(Sequential, layerSpecs)
  }

  net = DNN(layer, loss, type)

  tx = t(x)
  ty = t(y)

  batchSize = min(batchSize, nobs)
  nextBatch = BuildBatch(list(x=tx, y=ty, weights=weights))
  nSteps = round(nEpoch * nobs / batchSize)

  for (i in 1:nSteps) {
    batch = nextBatch(batchSize)
    net = learning(net, batch$x, batch$y, batch$weights, learningRate)
    if (verbose >= 1 && i %% 100 == 1) {
      cat("iter ", i, "total loss = ", net$totLoss, "\n")
    }
  }
  if (family == "multinomial") {
      fitted = predict(net, tx, type="label")
  } else {
      fitted = predict(net, tx)
  }
  structure(list(net=Strip(net), x=x, y=y, fitted=fitted), class="nnm.fit")
}

#' @export
print.nnm.fit <- function(x, ...) {
    print(x$net, ...)
    cat("sample response and predictions\n")
    ids = sample.int(length(x$y), min(20, length(x$y)))
    print(data.frame(y=x$y[ids], fitted=x$fitted[ids]))
}

#' @export
predict.nnm.fit <- function(object, x, type=c("response", "label"), ...) {
    type = match.arg(type)
    predict(object$net, t(x), type)
}

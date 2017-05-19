#' Load the MNIST digit recognition dataset into R
#'
#' @importFrom grDevices gray
#' @importFrom graphics image
#'
#' @param dataDir directory where MNIST data is in.
#' @return a list(train=train, test=test), where
#'
#' train$x is 60000x784 matrix, each row is one digit (28x28), if byrow=TRUE,
#' otherwise, train$x is 784x60000 matrix, each col is one digit(28x28).
#' train$y is 60000-length vector.
#'
#' test$x is 10000x784 matrix, test$y is 10000-length vector, if byrow=TRUE.
#' @export
LoadMnist <- function(dataDir, byrow=TRUE) {
  # dataDir = getSrcDirectory(function(x) {x})
  LoadImages <- function(filename) {
    f <- file(filename, "rb")
    readBin(f, "integer", n = 1, size = 4, endian = "big")
    N <- readBin(f, "integer", n = 1, size = 4, endian = "big")
    nrow <- readBin(f, "integer", n = 1, size = 4, endian = "big")
    ncol <- readBin(f, "integer", n = 1, size = 4, endian = "big")
    x <- readBin(f, "integer", n = N * nrow * ncol, size = 1, signed = F)
    close(f)
    if (byrow) {
      return(matrix(x/255, ncol = nrow * ncol, byrow=TRUE))
    } else {
      return(matrix(x/255, nrow = nrow * ncol))
    }
  }
  LoadLabels <- function(filename) {
    f <- file(filename, "rb")
    readBin(f, "integer", n = 1, size = 4, endian = "big")
    n <- readBin(f, "integer", n = 1, size = 4, endian = "big")
    y <- readBin(f, "integer", n = n, size = 1, signed = F)
    close(f)
    factor(y)
  }

  train = list()
  test = list()

  train$x <- LoadImages(file.path(dataDir, "train-images-idx3-ubyte"))
  test$x <- LoadImages(file.path(dataDir, "t10k-images-idx3-ubyte"))

  train$y <- LoadLabels(file.path(dataDir, "train-labels-idx1-ubyte"))
  test$y <- LoadLabels(file.path(dataDir, "t10k-labels-idx1-ubyte"))

  invisible(list(train = train, test = test))
}

#' ShowDigit display a digit.
#'
#' @param x array of length 28x28.
#' @export
ShowDigit <- function(x) {
  image(matrix(x, nrow = 28)[, 28:1], col = gray(12:1/12))
}

#' BuildBatch is to build a batch "generator" given a dataset.
#
#' @param data a list with two required elements ("x" and "y") and one optional
#' element ("weights"), where data$x is a matrix with dimension nvars x nobs,
#' data$y is a vector or a matrix. If data$weights exists, its length should
#' same as the number of cols in data$x.
#' @export
BuildBatch <- function(data) {
    if (!is.matrix(data$y)) {
      data$y = t(data$y)
    }
    Shuffle <- function(data) {
        nObs <- dim(data$x)[2]
        ids <- sample.int(nObs)
        data$x <- data$x[, ids, drop=FALSE]
        data$y <- data$y[, ids, drop=FALSE]
        if (!is.null(data$weights)) {
          data$weights <- data$weights[ids]
        }
        data
    }
    data = Shuffle(data)
    start <- 1
    nExamples <- ncol(data$x)
    function(batchSize = 100) {
        end <- min(start + batchSize - 1, nExamples)
        if (is.null(data$weights)) {
          ans <- list(x = data$x[, start:end, drop=FALSE], y = data$y[, start:end, drop=FALSE])
        } else {
          ans <- list(x = data$x[, start:end, drop=FALSE], y = data$y[, start:end, drop=FALSE], weights = data$weights[start:end])
        }

        if (end < nExamples) {
        start <<- end + 1
        } else {
        data <<- Shuffle(data)
        start <<- 1
        }
        ans
    }
}

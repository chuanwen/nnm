\name{nnm-package}
\alias{nnm-package}
\alias{nnm}
\docType{package}
\title{
  Neuro Network Model (nnm) whose usage is as easy as glm in R.
}
\usage{
nnm(x, y, layerSpecs, embeddingCols, embeddingDims, weights, family, ...)
}

\arguments{
\item{x}{matrix or data.frame with numeric, factor or character cols.
Each col is a variable or predictor, each row is an observation.}

\item{y}{response variable (vector or matrix). Quantitative for
‘family="gaussian"’, or ‘family="poisson"’ (non-negative counts).
For ‘family="multinomial"’, should be either a factor/character vector, or
a matrix of N-Classes cols.}

\item{layerSpecs}{either a list of layer specs, or a (composite) layer
objectect (e.g. a Sequential or DAG layer). See \code{\link{Sequential}}
for how to specify a sequential of layers,}

\item{embeddingCols}{specify factor cols to be embedded. Factor cols
that are not embeded would be encoded using default contrast matrix.}

\item{embeddingDims}{specify the embedding dims, if length(embeddingDims) is
only a fraction of length(embeddingCols), then
embeddingDims would be repeated to the same length of embeddingCols.}

\item{weights}{weight of the obs, default 1 for each and every observation.}

\item{family}{Response type, should be one of c("gaussian", "poisson",
"multinomial"). If not specified, will be inferred from class(y).}

\item{nEpoch}{number of times to train over the whole data, default 5.}

\item{batchSize}{number of observations per train batch, default 100.}

\item{learningRate}{a float between 0 and 1, default 0.25.}
}
\description{
neuron network model. It's a high level wrap of nnm.fit
}

\details{
  This section should provide a more detailed overview of how to use the
  package, including the most important functions.
}

\author{
Chuanwen Chen

Maintainer: Chuanwen Chen<chuanwen@gmail.com>
}
\references{
  This optional section can contain literature or other references for
  background information.
}

\keyword{ package }
\seealso{
  Optional links to other man pages
}

\examples{
x = iris[, 1:4]
y = iris[, 5]
mod = nnm(x, y, list(Dense(4, 3, Activation.Identity), Softmax))
mod

# example usage when x has factor cols
y = iris[, 1]
x = iris[, 2:5]
layerSpecs = list(Dense(3+nlevels(x$Species), 2), Dense(2, 1, Activation.Identity))
mod = nnm(x, y, layerSpecs)
mod

# example usage of embedding
y = iris[, 1]
x = iris[, 2:5]
embeddingCols = "Species"
embeddingDims = 2
layerSpecs = list(Dense(3+embeddingDims, 2), Dense(2, 1, Activation.Identity))
mod = nnm(x, y, layerSpecs, embeddingCols, embeddingDims)
mod

# example model for MNIST data
mnist <- LoadMnist("/home/ccw/learning/data/mnist")
train <- mnist$train
test <- mnist$test
mod = nnm(train$x, train$y, list(Dense(784, 15),
  Dense(15, 10, Activation.Identity), Softmax), verbose=1)
# accuracy on train set
mean(mod$y == mod$fitted)
# accuracy on test set
mean(test$y == predict(mod, test$x, type="label"))
}

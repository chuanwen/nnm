% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layer_compose.R
\name{Parallel}
\alias{Parallel}
\alias{Forward.Parallel}
\alias{Backward.Parallel}
\alias{UpdateParameters.Parallel}
\title{A S3 class to represent a layer that is consistent of several parallel layers.}
\usage{
Parallel(layer1, layer2, ...)

\method{Forward}{Parallel}(object, x, ...)

\method{Backward}{Parallel}(object, errorOut)

\method{UpdateParameters}{Parallel}(object, learningRate)
}
\arguments{
\item{layer1}{a Layer}

\item{layer2}{another Layer}

\item{...}{additional layers.}
}
\description{
A S3 class to represent a layer that is consistent of several parallel layers.
}
\section{Methods (by generic)}{
\itemize{
\item \code{Forward}: implement generic Forward function.

\item \code{Backward}: implement generic Backward function.

\item \code{UpdateParameters}: implement generic UpdateParameters function.
}}

\examples{
layer1 = Sequential(Dense(10, 20), Dense(20, 8), Softmax)
layer2 = Dense(3, 5)
# net has inputDim of 10 + 3, outputDim of 8 + 5
net = Parallel(layer1, layer2)
net
}

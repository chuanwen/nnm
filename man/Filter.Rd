% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/layer_filter.R
\name{Filter}
\alias{Filter}
\title{A S3 class to represent Filter layer.}
\usage{
Filter(inputDim, filter, outputDim)
}
\arguments{
\item{filter}{a logical array to indicate which element can pass.}
}
\description{
A S3 class to represent Filter layer.
}
\examples{
layer = Filter(3, c(TRUE, TRUE, FALSE))
layer

nObs = 1
x = array(rnorm(3), c(3, 1))
layer = Forward(layer, x)
layer$x # input
layer$a # output
}

% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dnn.R
\name{learning}
\alias{learning}
\title{learning is a generic function}
\usage{
learning(object, x, y, weights, learningRate, ...)
}
\arguments{
\item{object}{Object that supports this generic function}

\item{x}{input data, always assume there are multiple observations in x}

\item{y}{output data, always assume there are same number of observations as in x}

\item{weights}{weight of each observation, if missing, default 1 for each observation.}

\item{learningRate}{usually a small positive number, e.g. 0.1}
}
\description{
learning is a generic function
}

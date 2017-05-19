#' Activation class constructor
#'
#' @param f A function to compute f(z).
#' @param df A function to compute df/dz, input parameters are z and a.
#' @param name name of the activation.
#' @return An Activation object.
#' @examples
#' f <- function(z) z
#' df <- function(z, a) rep(1, length(z))
#' identity <- Activation(f, df, "identity")
#' identity
#' @export
Activation <- function(f, df, name) {
  structure(list(f = f, df = df, name = name), class = "Activation")
}

#' Identity Activation object
#'
#' @examples
#' Dense(3, 4, Activation.Identity) # a 3 -> 4 Dense network with Identity activation.
#' @export
Activation.Identity <- (function() {
  f <- function(z) {
    z
  }
  df <- function(z, a) {
    rep(1, length(a))
  }
  Activation(f, df, "Identity")
})()

#' ReLU Activation object
#'
#' @examples
#' z <- seq(from=-5, to=5, length=201)
#' a <- Activation.ReLU$f(z)
#' plot(a ~ z, main='ReLU')
#' plot(Activation.ReLU$df(a) ~ a, main='ReLU df')
#' @export
Activation.ReLU <- (function() {
  f <- function(z) {
    pmax(z, 0)
  }
  df <- function(z, a) {
    (z > 0) + 0
  }
  Activation(f, df, "ReLU")
})()

#' Sigmoid Activation object
#'
#' @examples
#' z <- seq(from=-5, to=5, length=201)
#' plot(Activation.Sigmoid$f(z) ~ z)
#' @export
Activation.Sigmoid <- (function() {
  f <- function(z) {
    1/(1 + exp(-z))
  }
  df <- function(z, a) {
    a * (1 - a)
  }
  Activation(f, df, "Sigmoid")
})()

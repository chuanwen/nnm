SGD <- function(parameter) {
  delta0 <- rep(0.0, length(parameter))
  dim(delta0) <- dim(parameter)
  function(x, delta) {
    delta0 <<- (delta0 + delta)/2.0
    x - delta0
  }
}

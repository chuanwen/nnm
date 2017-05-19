
#' A S3 class to represent a "Layer" that contains a sequence of layers.
#'
#' @param layer1 a Layer
#' @param layer2 another Layer
#' @param ... additional layers.
#' @examples
#'
#' net <- Sequential(Dense(784, 15), Dense(15, 10, Activation.Identity), Softmax)
#' net
#' @export
Sequential <- function(layer1, layer2, ...) {
  createLayer <- function(inputDim, layerSpec) {
    if ("Layer" %in% class(layerSpec)) {
      return(layerSpec)
    }

    if (is.function(layerSpec)) {
      return(layerSpec(inputDim))
    }

    if (class(layerSpec) == "character") {
      layerSpec = list(layerSpec)
    }

    name = layerSpec[[1]]
    parameters = c(inputDim, layerSpec[-1])
    switch(tolower(name),
      dense = do.call(Dense, parameters),
      conv = do.call(Conv, parameters),
      maxpool = do.call(MaxPool, parameters),
      reshape = do.call(Reshape, parameters),
      softmax = do.call(Softmax, parameters)
    )
  }

  layerSpecs = list(layer1, layer2, ...)
  inputDim <- layer1$inputDim
  nLayers <- length(layerSpecs)

  layers <- vector(nLayers, mode = "list")
  layerInputDim <- inputDim
  myclass <- "Sequential"
  for (i in 1:nLayers) {
    spec <- layerSpecs[[i]]
    layers[[i]] <- createLayer(layerInputDim, spec)
    # inputDim for next layer #
    layerInputDim <- layers[[i]]$outputDim
  }
  outputDim = layers[[nLayers]]$outputDim

  structure(list(inputDim = inputDim, outputDim = outputDim, layers = layers),
            class = c("Sequential", "Layer"))
}

#' @describeIn Sequential strip the net to only keep what is needed for Forward method.
#' @export
Strip.Sequential <- function(object) {
  nLayers = length(object$layers)
  layers = vector(nLayers, mode="list")
  for (i in 1:nLayers) {
    layers[[i]] = Strip(object$layers[[i]])
  }
  inputDim = object$inputDim
  outputDim = object$outputDim
  structure(list(inputDim = inputDim, outputDim = outputDim, layers = layers), class = class(object))
}

#' @export
print.Sequential <- function(x, left="", ...) {
  cat(left, "Sequential Layer", x$inputDim, "->", x$outputDim, "\n")
  for (i in 1:length(x$layers)) {
    print(x$layers[[i]], left=paste0(left, "   |"), ...)
  }
}

#' @describeIn Sequential implement generic Forward function
#' @export
Forward.Sequential <- function(object, x, ...) {
  object$x = x
  n <- length(object$layers)
  input <- x
  for (i in 1:n) {
    if ("Dropout" %in% class(object$layers[[i]])) {
      object$layers[[i]] <- Forward(object$layers[[i]], input, ...)
    } else {
      object$layers[[i]] <- Forward(object$layers[[i]], input)
    }
    input <- object$layers[[i]]$a
  }
  object$a = object$layers[[n]]$a
  invisible(object)
}

#' @describeIn Sequential implement generic Backward function.
#' @export
Backward.Sequential <- function(object, errorOut) {
  for (i in length(object$layers):1) {
    object$layers[[i]] <- Backward(object$layers[[i]], errorOut)
    errorOut <- object$layers[[i]]$errorIn
  }
  object$errorIn <- object$layers[[1]]$errorIn
  invisible(object)
}

UpdateParameters.CompositeLayer <- function(object, learningRate) {
  if (learningRate > 0) {
    for (i in length(object$layers):1) {
      object$layers[[i]] <- UpdateParameters(object$layers[[i]], learningRate)
    }
  }
  invisible(object)
}

InitParameters.CompositeLayer <- function(object, ...) {
  for (i in length(object$layers):1) {
    object$layers[[i]] <- InitParameters(object$layers[[i]], ...)
  }
  invisible(object)
}

#' @describeIn Sequential implement generic UpdateParameters function.
#' @export
UpdateParameters.Sequential = UpdateParameters.CompositeLayer

#' @export
InitParameters.Sequential = InitParameters.CompositeLayer


indexFromDim <- function(dims) {
  if (length(dims) == 0) {
    return(integer(length=0))
  }
  stopifnot(all(dims > 0))
  n <- length(dims)
  ans <- vector(n, mode="list")
  start <- 1
  for (i in 1:n) {
    end <- start + dims[i] - 1
    ans[[i]] = start:end
    start <- end + 1
  }
  ans
}

#indexFromDim(c(3,5,2))

#' A S3 class to represent a layer that is consistent of several parallel layers.
#'
#' @param layer1 a Layer
#' @param layer2 another Layer
#' @param ... additional layers.
#' @examples
#' layer1 = Sequential(Dense(10, 20), Dense(20, 8), Softmax)
#' layer2 = Dense(3, 5)
#' # net has inputDim of 10 + 3, outputDim of 8 + 5
#' net = Parallel(layer1, layer2)
#' net
#' @export
Parallel <- function(layer1, layer2, ...) {
  layers <- list(layer1, layer2, ...)
  n = length(layers)
  for (layer in layers) {
    stopifnot("Layer" %in% class(layer))
  }

  inputDims = sapply(layers, function(layer)layer$inputDim)
  outputDims = sapply(layers, function(layer)layer$outputDim)

  inputIndex = indexFromDim(inputDims)
  outputIndex = indexFromDim(outputDims)

  inputDim = sum(inputDims)
  outputDim = sum(outputDims)

  structure(list(inputDim = inputDim, outputDim = outputDim, layers = layers,
                 inputIndex = inputIndex, outputIndex = outputIndex),
            class=c("Parallel", "Layer"))
}

#' @export
print.Parallel <- function(x, left="", ...) {
  cat(left, "Paralleled Layers", x$inputDim, "->", x$outputDim, "\n")
  for (i in 1:length(x$layers)) {
    print(x$layers[[i]], left=paste0(left, "   |"))
  }
}

#' @describeIn Parallel implement generic Forward function.
#' @export
Forward.Parallel <- function(object, x, ...) {
  object$x = x
  input <- x
  for (i in 1:length(object$layers)) {
    input <- x[object$inputIndex[[i]], ]
    if ("Dropout" %in% class(object$layers[[i]])) {
      object$layers[[i]] <- Forward(object$layers[[i]], input, ...)
    } else {
      object$layers[[i]] <- Forward(object$layers[[i]], input)
    }
  }
  object$a = do.call(rbind, lapply(object$layers, function(layer)layer$a))
  invisible(object)
}

#' @describeIn Parallel implement generic Backward function.
#' @export
Backward.Parallel <- function(object, errorOut) {
  for (i in 1:length(object$layers)) {
    layerErrorOut <- errorOut[object$outputIndex[[i]], ]
    object$layers[[i]] <- Backward(object$layers[[i]], layerErrorOut)
  }
  object$errorIn <- do.call(rbind, lapply(object$layers, function(layer)layer$errorIn))
  invisible(object)
}

#' @describeIn Parallel implement generic UpdateParameters function.
#' @export
UpdateParameters.Parallel <- UpdateParameters.CompositeLayer

#' @export
InitParameters.Parallel = InitParameters.CompositeLayer

#' A S3 class to represent a directed acycle graph (DAG) of a list of
# layer nodes.
#'
#' @param layers a list of layers
#' @param edges a vector of even intergers. The pair of integers at position
#' 2*i-1 and 2*i represents a directed edge in the DAG. We assumed the number
# at 2*i-1 is smaller the one in 2*i (to prevent cycle). Example edges:
#' c(1,2, 2,3) means 1->2, 2->3
#' c(1,2, 4,5) means 1->2, 4->5
#' c(1,2, 1,3, 2,3) means 1->2, 1->3, 2->3. Note that if output dim of 1 and 2 is
#' o1 and o2, respectively, then input dim of 3 is o1 + o2, with first o1 dim from
#' 1, second o2 dim from 2.
#'
#' @examples
#' layers = list(Dense(3,5), Dense(5, 8), Dense(8, 7), Dense(8, 6), Softmax(6))
#' edges = list(c(1,2,3), c(2,4,5))
#' # net has inputDim of 3, outputDim of 8 + 6
#' net = DAG(layers, c(1,2, 2,3, 2,4, 4,5))
#' net
#' @export
DAG <- function(layers, edges) {
  if (missing(edges) || length(edges) == 0) {
    return(do.call(Parallel, layers))
  }
  edges = data.frame(matrix(edges, ncol=2, byrow=TRUE))
  colnames(edges) = c("src", "dest")
  stopifnot(all(edges$dest > edges$src))

  getOutputDim <- function(i) layers[[i]]$outputDim

  n = length(layers)
  layerSNs = 1:n # layer serial numbers

  # serial No. of input layers for each layer
  inputSNs = lapply(layerSNs, function(i) sort(edges$src[edges$dest == i]))

  # serial No. of output layers for each layer
  outputSNs = lapply(layerSNs, function(i) sort(edges$dest[edges$src == i]))

  # For layer i, if inputDims is c(3,2,4), then inputIndex 1:3, 4:5, 6:9
  inputIndex = lapply(layerSNs, function(i) {
    inputDims = sapply(inputSNs[[i]], getOutputDim)
    indexFromDim(inputDims)
  })

  # For layer i, outputIndex[[i]] is a list of vectors, where j-th vector
  # gives the location of layer i output in the input matrix of outputSNs[i][j].
  outputIndex = lapply(layerSNs, function(i) {
    lapply(outputSNs[[i]], function(j) { # i -> j
      idx = which(inputSNs[[j]] == i)    # among j's input, location of i
      inputIndex[[j]][[idx]]
    })
  })

  # layer which get input from external layers
  headSNs = layerSNs[sapply(layerSNs, function(i) length(inputSNs[[i]])==0)]

  # layer whose output is to external layers
  tailSNs = layerSNs[sapply(layerSNs, function(i) length(outputSNs[[i]])==0)]

  #middleSNs = sort(setdiff(layerSNs, union(headSNs, tailSNs)))

  nonHeadSNs = sort(setdiff(layerSNs, headSNs))
  nonTailSNs = sort(setdiff(layerSNs, tailSNs))

  headDims = sapply(headSNs, function(i)layers[[i]]$inputDim)
  tailDims = sapply(tailSNs, function(i)layers[[i]]$outputDim)

  headIndex = indexFromDim(headDims)
  tailIndex = indexFromDim(tailDims)

  inputDim = sum(headDims)
  outputDim = sum(tailDims)
  structure(list(inputDim = inputDim, outputDim = outputDim, layers = layers,
                 inputSNs = inputSNs, outputSNs = outputSNs,
                 inputIndex = inputIndex, outputIndex = outputIndex,
                 headSNs = headSNs, tailSNs = tailSNs,
                 nonHeadSNs = nonHeadSNs, nonTailSNs = nonTailSNs,
                 headIndex = headIndex, tailIndex = tailIndex,
                 layers = layers, edges = edges),
            class=c("DAG", "Layer"))
}

Forward.DAG <- function(object, x, ...) {
  geta = function(i) object$layers[[i]]$a
  # forward headSNs first
  for (i in 1:length(object$headSNs)) {
    sn = object$headSNs[i]
    object$layers[[sn]] = Forward(object$layers[[sn]], x[object$headIndex[[i]], ], ...)
  }
  for (i in object$nonHeadSNs) {
    inputSNs = object$inputSNs[[i]]
    input = do.call(rbind, lapply(inputSNs, geta))
    object$layers[[i]] = Forward(object$layers[[i]], input, ...)
  }
  object$x = x
  object$a = do.call(rbind, lapply(object$tailSNs, geta))
  invisible(object)
}

Backward.DAG <- function(object, errorOut) {
  getErrorIn = function(i) object$layers[[i]]$errorIn
  for (i in 1:length(object$tailSNs)) {
    sn = object$tailSNs[i]
    object$layers[[sn]] = Backward(object$layers[[sn]], errorOut[object$tailIndex[[i]], ])
  }
  for (i in sort(object$nonTailSNs, decreasing=TRUE)) {
    outputSNs = object$outputSNs[[i]]
    outputIndex = object$outputIndex[[i]]
    errorOut = Reduce('+', lapply(1:length(outputSNs), function(j){
        sn = outputSNs[j]
        rows = outputIndex[[j]]
        object$layers[[sn]]$errorIn[rows, ]
    }))
    object$layers[[i]] = Backward(object$layers[[i]], errorOut)
  }
  object$errorIn = do.call(rbind, lapply(object$headSNs, getErrorIn))
  invisible(object)
}

#' @export
UpdateParameters.DAG = UpdateParameters.CompositeLayer

#' @export
InitParameters.DAG = InitParameters.CompositeLayer

#' @export
print.DAG <- function(x, left="", ...) {
  cat(left, "Directed Acycle Graph", x$inputDim, "->", x$outputDim, "\n")
  for (i in 1:length(x$layers)) {
    print(x$layers[[i]], left=sprintf("%s   | node %d: ", left, i), ...)
  }
  for (i in 1:nrow(x$edges)) {
    cat(left, "  | edge: node", x$edges$src[i], "-> node", x$edges$dest[i], "\n")
  }
}

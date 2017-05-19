context("layer")

test_that("Dense layer", {
    layer = Dense(inputDim=3, outputDim=5)
    x = array(rnorm(3*2), dim=c(3,2))
    errorOut=array(rnorm(10), dim=c(5,2))
    layer = Forward(layer, x)

    expect_equal(dim(layer$a), c(5,2))
    expect_true(all(layer$a >= 0))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), c(3,2))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("Conv layer", {
    inputDim = c(8,8,2)
    outputDim = c(8,8,5)
    kernelDim = c(3,3,2,5)
    nObs = 3

    layer = Conv(inputDim=inputDim, kernelDim=kernelDim)
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut=array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))

    layer = Forward(layer, x)

    expect_equal(dim(layer$a), c(outputDim, nObs))
    expect_true(all(layer$a >= 0))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), c(inputDim,nObs))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("Softmax layer", {
    layer = Softmax(numClasses=10)
    x = array(rnorm(10*2), dim=c(10,2))
    errorOut=array(rnorm(10*2), dim=c(10,2))
    layer = Forward(layer, x)

    expect_equal(dim(layer$a), c(10,2))
    expect_true(all(layer$a >= 0))
    expect_equal(apply(layer$a, 2, sum), c(1, 1))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), c(10,2))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("MaxPool", {

    inputDim = c(16, 16, 3)
    kernelDim = c(2,2)
    outputDim = c(8, 8, 3)
    nObs = 10
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut = array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))

    layer = MaxPool(inputDim=inputDim, kernelDim=kernelDim)
    layer = Forward(layer, x)

    expect_equal(dim(layer$a), c(outputDim, nObs))
    expect_equal(max(layer$a), max(x))
    expect_true(min(layer$a) >= min(x))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("Reshape", {

    inputDim = c(16, 16, 3)
    outputDim = c(16*16*3)
    nObs = 10
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut = array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))

    layer = Reshape(inputDim=inputDim, outputDim=outputDim)
    layer = Forward(layer, x)

    expect_equal(dim(layer$a), c(outputDim, nObs))
    expect_equal(as.vector(layer$a), as.vector(x))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("Filter", {

    inputDim = c(4, 6)
    outputDim = c(2, 6)
    filter = c(TRUE, FALSE)
    nObs = 10
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut = array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))

    layer = Filter(inputDim=inputDim, filter, outputDim=outputDim)
    layer = Forward(layer, x)

    expect_equal(dim(layer$a), c(outputDim, nObs))
    expect_equal(as.vector(layer$a), as.vector(x[filter]))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("Dropout", {

    inputDim = 8
    outputDim = inputDim
    nObs = 2
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut = array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))

    layer = Dropout(inputDim=inputDim, keepProb=0.5)
    layer = Forward(layer, x)

    expect_equal(dim(layer$a), c(outputDim, nObs))
    keepIndicator = !layer$maskIndicator
    expect_equal(as.vector(layer$a[keepIndicator]), as.vector(x[keepIndicator]))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut, random=FALSE)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("Sequential", {
    inputDim = 8
    outputDim = 5
    nObs = 2
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut = array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))

    layer = Sequential(Dense(inputDim, outputDim, Activation.Identity), Softmax)
    layer = Forward(layer, x)
    expect_equal(dim(layer$a), c(outputDim, nObs))
    expect_true(all(layer$a >= 0))
    expect_equal(apply(layer$a, 2, sum), rep(1, nObs))

    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("Parallel", {
    inputDim = 8
    outputDim = 5
    nObs = 2
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut=array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))
    layer = Parallel(Dense(inputDim-4, 3), Dense(4, outputDim-3))
    layer = Forward(layer, x)
    expect_equal(dim(layer$a), dim(errorOut))
    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("DAG-Sequential", {
    inputDim = 8
    outputDim = 5
    nObs = 2
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut=array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))
    layer = DAG(list(Dense(inputDim, outputDim, Activation.Identity), Softmax(outputDim)), c(1,2))
    layer = Forward(layer, x)
    expect_equal(dim(layer$a), dim(errorOut))
    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("DAG: 1 points to 2,3", {
    inputDim = 8
    outputDim = 5
    nObs = 2
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut=array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))
    layer = DAG(list(Dense(inputDim, 5), Dense(5, outputDim-3), Dense(5, 3)),
                c(1,2,1,3))
    layer = Forward(layer, x)
    expect_equal(dim(layer$a), dim(errorOut))
    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("DAG: 1,2 point to 3", {
    inputDim = 8
    outputDim = 5
    nObs = 2
    x = array(rnorm(prod(inputDim)*nObs), dim=c(inputDim, nObs))
    errorOut=array(rnorm(prod(outputDim)*nObs), dim=c(outputDim, nObs))
    layer = DAG(list(Dense(inputDim-2, 3), Dense(2, 6), Dense(9, outputDim)),
                c(1,3,2,3))
    layer = Forward(layer, x)
    expect_equal(dim(layer$a), dim(errorOut))
    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$errorIn), dim(x))

    layer1 = numericBackward(layer, errorOut)
    expect_equal(layer1$errorIn, layer$errorIn)
})

test_that("EmbeddingOne", {
    x = factor(c("a", "b", "a", "c", "b"), levels=letters)
    embeddingDim = 4
    errorOut = matrix(rnorm(embeddingDim*length(x)), nrow=embeddingDim)
    layer = EmbeddingOne(1, nlevels(x), embeddingDim)
    layer = Forward(layer, x)
    expect_equal(dim(layer$a), c(embeddingDim, length(x)))
    layer = Backward(layer, errorOut)
    expect_equal(dim(layer$delta), dim(layer$W))
    W = layer$W
    layer = UpdateParameters(layer, 0.1)
    expect_equal(dim(W), dim(layer$W))

    # We do not have data for the levels that are not in x, hence
    # their weights should not change after UpdateParameters
    indx = unique(as.integer(x))
    expect_equal(as.vector(W[, -indx]), as.vector(layer$W[, -indx]))
})

test_that("Embedding", {
    x = matrix(c(1,2,1,3,
                 2,3,5,4), nrow=2, byrow=T)
    inputDim = nrow(x)
    inputlevels = c(4, 8)
    embeddingDims = c(2, 3)
    outputDim = sum(embeddingDims)
    nObs = ncol(x)
    errorOut = matrix(rnorm(nObs*outputDim), ncol=nObs)

    layer = Embedding(inputDim, inputlevels, embeddingDims)
    expect_equal(layer$outputDim, outputDim)

    layer = Forward(layer, x)
    expect_equal(dim(layer$a), c(outputDim, nObs))

    layer = Backward(layer, errorOut)
    oldW1 = layer$layers[[1]]$W
    oldW2 = layer$layers[[2]]$W
    layer = UpdateParameters(layer, 0.25)
    newW1 = layer$layers[[1]]$W
    newW2 = layer$layers[[2]]$W
    expect_equal(as.vector(oldW1[, -x[1,]]), as.vector(newW1[, -x[1,]]))
    expect_equal(as.vector(oldW2[, -x[2,]]), as.vector(newW2[, -x[2,]]))
})

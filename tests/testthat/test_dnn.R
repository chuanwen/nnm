context("dnn")

test_that("DNN", {
    x = t(as.matrix(iris[, 1:4]))
    y = iris[, 5]
    inputDim = nrow(x)
    hiddenDim = 5
    outputDim = length(levels(y))
    nObs = ncol(x)

    net = DNN(Sequential(Dense(inputDim, hiddenDim),
                         Dense(hiddenDim, outputDim, Activation.Identity), Softmax),
              type="classification")
    expect_equal(class(net$loss), "Loss")
    for (i in 1:200) {
      net = learning(net, x, y, learningRate=0.1)
    }
    expect_equal(dim(net$a), c(outputDim, nObs))
    expect_true(all(net$a >= 0))
    expect_equal(apply(net$a, 2, sum), rep(1, nObs))
    expect_equal(Forward(net, x)$a, predict(net, x))
    #expect_gt(mean(y==predict(net, x, type="label")), 0.9)
})

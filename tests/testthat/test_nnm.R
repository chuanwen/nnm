context("nnm")

test_that("nnm.fit classification", {
    x = iris[, 1:4]
    y = iris[, 5]
    mod = nnm.fit(x, y, list(Dense(4, 6), Dense(6, 3, Activation.Identity), Softmax))
    expect_equal(length(mod$fitted), length(y))
})

test_that("nnm.fit regression", {
    x = iris[, 1:3]
    y = iris[, 4]
    mod = nnm.fit(x, y, list(Dense(3, 6), Dense(6, 1, Activation.Identity)))
    expect_equal(length(mod$fitted), length(y))
})

test_that("nnm.fit regression with parallel layers", {
    x = iris[, 1:3]
    y = iris[, 4]
    layer1 = Parallel(Identity(1), Identity(2))
    layer2 = Sequential(Dense(3, 6), Dense(6, 1, Activation.Identity))
    mod = nnm.fit(x, y, list(layer1, layer2))
    expect_equal(length(mod$fitted), length(y))
})

test_that("nnm with all numeric data.frame", {
    x = iris[, 1:4]
    y = iris[, 5]
    mod = nnm(x, y, list(Dense(4, 6), Dense(6, 3, Activation.Identity), Softmax))
    expect_equal(length(mod$fitted), length(y))
})

test_that("nnm with factor + numeric data.frame", {
    x = iris[, 2:5]
    y = iris[, 1]
    layerSpec = list(Dense(4+2, 6), Dense(6, 1, Activation.Identity))
    mod = nnm(x, y, layerSpec)
    expect_equal(length(mod$fitted), length(y))
})

test_that("nnm with embedding factor + numeric data.frame", {
    x = iris[, 2:5]
    y = iris[, 1]
    layerSpec = list(Dense(3+3, 6), Dense(6, 1, Activation.Identity))
    mod = nnm(x, y, layerSpec, embeddingCols="Species", embeddingDims=3)
    expect_equal(length(mod$fitted), length(y))
})

import system, json, options, random, times, terminal
import strutils, terminal, strformat
import nn, neo

# Whatever. We can always break the training process
# TODO: actually save the trained network after each epoch.
const max_epochs = 1000

proc main() =
    echo "Iniatilizing randomness..."
    randomize(int64(epochTime() * 1000))
    echo "Starting up..."
    let network = makeNeuralNetwork(
        (LayerType.Normal, @[28 * 28]), 
        (LayerType.Convolution, @[16, 7]),
        (LayerType.Normal, @[16]),
        (LayerType.Normal, @[10]))
    echo "Loading training data..."
    let data = parseJson(readFile("../data/mnist_handwritten_train.json"))
    var dataSet = to(data, DataSet)
    echo "Loading testing data..."
    let testData = parseJson(readFile("../data/mnist_handwritten_test.json"))
    let testDataSet = to(testData, DataSet)
    let size = dataSet.len
    let testSize = testDataSet.len
    for epoch in 0..max_epochs:
        echo "Shuffling data set..."
        shuffle(dataSet)
        for i in 0..(size - 1):
            let loss = network.train(0.25, dataSet[i])
            stdout.eraseLine()
            stdout.write &"Epoch [{epoch}]: progress [{i}/{size}], loss = {loss}"
            stdout.flushFile()
        echo ""
        echo "Epoch [", epoch, "]: Testing on test dataset..."
        var correct = 0
        for i in 0..(testDataSet.len - 1):
            let result = network.run(testDataSet[i].image).get().maxIndex
            stdout.eraseLine()
            stdout.write &"Epoch [{epoch}], Test [{i}/{testSize}]: should be {testDataSet[i].label}, got {result}"
            stdout.flushFile()
            if result.i.byte == testDataSet[i].label:
                correct += 1
        echo ""
        echo "Epoch [", epoch, "]: Hit rate: ", correct.float64 / testDataSet.len.float64 * 100.float64, "%"

when isMainModule:
    main()

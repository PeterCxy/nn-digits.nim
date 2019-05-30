import system, json, options, random, times
import nn, neo

proc main() =
    echo "Iniatilizing randomness..."
    randomize(int64(epochTime() * 1000))
    echo "Starting up..."
    let network = makeNeuralNetwork(28 * 28, 16, 16, 10)
    # TODO: Change to use the training dataset. Use test now because it's smaller
    # and quicker to load
    let data = parseJson(readFile("../data/mnist_handwritten_train.json"))
    let dataSet = to(data, DataSet)
    for i in 0..(dataSet.len - 1):
        #echo "Sample ", i
        network.train(0.5, dataSet[i])
    echo "Testing on test dataset..."
    let testData = parseJson(readFile("../data/mnist_handwritten_test.json"))
    let testDataSet = to(testData, DataSet)
    var correct = 0
    for i in 0..(testDataSet.len - 1):
        let result = network.run(testDataSet[i].image).get().maxIndex
        echo "Test: ", i, " should be ", testDataSet[i].label, ", got ", result
        if result.i.byte == testDataSet[i].label:
            correct += 1
    echo "Hit rate: ", correct.float64 / testDataSet.len.float64 * 100.float64, "%"

when isMainModule:
    main()

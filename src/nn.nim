# Thanks: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=2s
import neo
import options, sugar, math

type
    Layer = ref object
        # prevLen = 0 means it is input layer
        prevLen: int
        # Column vector values represent the activation
        neurons*: Matrix[float64]
        d: Matrix[float64]
        # every row represents the weight from the previous layer
        # to one neuron in the current layer
        weights: Option[Matrix[float64]]
        biases: Matrix[float64]
    NeuralNetwork* = ref object
        # The first layer is input layer, the last is output layer
        layers: seq[Layer]
    Sample* = object
        image*: seq[byte]
        label*: byte
    DataSet* = seq[Sample]

proc makeLayer(len: int; prevLen: int): Layer =
    return Layer(
        prevLen: prevLen,
        neurons: constantMatrix(len, 1, 0.float64),
        biases: constantMatrix(len, 1, 0.float64), # Not implemented yet
        d: constantMatrix(len, 1, 0.float64),
        weights: if prevLen == 0:
            none[Matrix[float64]]()
        else:
            some(randomMatrix(len, prevLen, rowMajor))
    )

proc makeNeuralNetwork*(layerSizes: varargs[int]): NeuralNetwork =
    var layers: seq[Layer]
    newSeq(layers, layerSizes.len)
    result = NeuralNetwork(layers: layers)
    var prevLen = 0
    for i, size in layerSizes:
        result.layers[i] = makeLayer(size, prevLen)
        prevLen = size

proc sigmoid(x: float64): float64 = 1.float64 / ((-x).exp + 1.float64)
proc dSigmoid(x: float64): float64 = sigmoid(x) * (1.float64 - sigmoid(x))
proc crossEntropy(target: float64, predict: float64): float64 =
    - (target * ln(1e-15 + predict) + (1 - target) * ln(1e-15 + 1 - predict))
proc dCrossEntropy(target: float64, predict: float64): float64 =
    - (target / (1e-15 + predict) + (1 - target) / (1e-15 + predict - 1))

proc calculateActivation*(self: Layer, prev: Layer) =
    if self.prevLen == 0:
        return # We cannot propagate for input layer
    self.neurons = (self.weights.get() * prev.neurons + self.biases)
        .map((x) => x / self.prevLen.float64)
        .map(sigmoid) # Sigmoid Rectifier
    # Calculate the derivative of the output in terms of input (weighted sum of previous layer)
    # i.e. the derivative of sigmoid
    self.d = (self.weights.get() * prev.neurons + self.biases)
        .map((x) => x / self.prevLen.float64)
        .map(dSigmoid)

proc run*(self: NeuralNetwork, input: seq[byte]): Option[Vector[float64]] =
    if input.len != self.layers[0].neurons.column(0).len:
        return none[Vector[float64]]() # We cannot do anything if input length differ
    # Set the input layer
    for i, b in input:
        self.layers[0].neurons[i, 0] = b.float64 / 255.float64
    # Propagate through all layers
    for i in 1..(len(self.layers) - 1):
        self.layers[i].calculateActivation(self.layers[i - 1])
    # Now we have the output
    return some(self.layers[self.layers.len - 1].neurons.column(0))

proc backPropagate*(self: Layer, prev: Layer, target: seq[float64], step: float64): seq[float64] =
    newSeq(result, self.prevLen)
    for i in 0..(self.prevLen - 1):
        result[i] = 0
    for i in 0..(self.neurons.column(0).len - 1):
        # Derivative of loss function in terms of the activation of the current neuron
        let dEdO = dCrossEntropy(target[i], self.neurons[i, 0])
        # Derivative of the activation in terms of its input (weighted sum of previous layer)
        # i.e. the derivative of the sigmoid function (calculated along with the activation)
        let dOdO: float64 = self.d[i, 0]
        for j in 0..(self.prevLen - 1):
            # Calculate changes in terms of all the partial derivatives
            let dW = dEdO * dOdO * prev.neurons[j, 0]
            # Change previous layer's activation proportional to weight
            result[j] -= dEdO * dOdO * self.weights.get()[i, j]
            # Change weight proportional to activation
            self.weights.get()[i, j] -= step * (dW)
            
            #echo result[j]
    for i in 0..(self.prevLen - 1):
        result[i] = prev.neurons[i, 0] + result[i] / self.neurons.column(0).len.float64

proc train*(self: NeuralNetwork, step: float64, sample: Sample) =
    let outLen = self.layers[self.layers.len - 1].neurons.column(0).len
    var target: seq[float64]
    newSeq(target, outLen)
    let result = self.run(sample.image).get()
    var loss: float64 = 0
    for i in 0..(outLen - 1):
        target[i] = if i.byte == sample.label:
            1
        else:
            0
        # Record the actual loss of this pass
        loss += crossEntropy(target[i], result[i])
    echo "loss = ", loss / outLen.float64
    for i in 0..(self.layers.len - 2):
        let index = self.layers.len - i - 1
        target = self.layers[index].backPropagate(self.layers[index - 1], target, step)
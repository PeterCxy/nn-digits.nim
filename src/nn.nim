# Thanks: https://www.youtube.com/watch?v=tIeHLnjs5U8&t=2s
import neo
import options, sugar, math

type
    # Abstract definition of a layer in neural network
    AbsLayer = ref object of RootObj
        # prevLen = 0 means it is input layer
        prevLen: int
        # Column vector values represent the activation
        neurons*: Matrix[float64]
        d: Matrix[float64]
    # A normal neural layer: no convolution, no fancy stuff
    Layer = ref object of AbsLayer
        # every row represents the weight from the previous layer
        # to one neuron in the current layer
        weights: Option[Matrix[float64]]
        biases: Matrix[float64]
    NeuralNetwork* = ref object
        # The first layer is input layer, the last is output layer
        layers: seq[AbsLayer]
    Sample* = object
        image*: seq[byte]
        label*: byte
    DataSet* = seq[Sample]

proc makeLayer(len: int; prevLen: int): Layer =
    return Layer(
        prevLen: prevLen,
        neurons: constantMatrix(len, 1, 0.float64),
        biases: constantMatrix(len, 1, 0.float64),
        d: constantMatrix(len, 1, 0.float64),
        weights: if prevLen == 0:
            none[Matrix[float64]]()
        else:
            some(randomMatrix(len, prevLen, 0.1, rowMajor))
    )

proc makeNeuralNetwork*(layerSizes: varargs[int]): NeuralNetwork =
    var layers: seq[AbsLayer]
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

# Any implementation of AbsLayer should implement this function
method calculateActivation*(self: AbsLayer, prev: AbsLayer) {.base.} =
    discard
# Implementation for a normal NN layer
method calculateActivation*(self: Layer, prev: AbsLayer) =
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

# Any implementation of AbsLayer should implement this function
method backPropagate*(self: AbsLayer, prev: AbsLayer, target: seq[float64], step: float64): seq[float64] {.base.} =
    return @[]
# Implementation for a normal NN layer
method backPropagate*(self: Layer, prev: AbsLayer, target: seq[float64], step: float64): seq[float64] =
    let selfLen = self.neurons.column(0).len
    newSeq(result, self.prevLen)
    for i in 0..(self.prevLen - 1):
        result[i] = 0
    for i in 0..(selfLen - 1):
        # Note: weighted sum is activation BEFORE the sigmoid function
        # Derivative of loss function in terms of the activation of the current neuron
        let dEdO = dCrossEntropy(target[i], self.neurons[i, 0])
        # Derivative of the activation in terms of its input (weighted sum of previous layer)
        # i.e. the derivative of the sigmoid function (calculated along with the activation)
        let dOdO: float64 = self.d[i, 0]
        let dEdOdOdO: float64 = dEdO * dOdO
        # Change in bias should be the previous two derivatives
        # times the derivative of the weighted sum against the
        # bias, which is basically 1. Note that biases do not
        # directly correlate to one specific neuron in the previous
        # layer, so we just do it here.
        self.biases[i, 0] -= step * dEdOdOdO
        for j in 0..(self.prevLen - 1):
            # Calculate changes in terms of all the partial derivatives
            # prev.neurons[j, 0] is the derivative of the weighted sum
            # in terms of the activation of neuron j in previous layer
            let dW = dEdOdOdO * prev.neurons[j, 0]
            # Change previous layer's activation proportional to weight
            # the weight is the derivative of the weighted sum against
            # the weight applied by the current neuron to neuron j in previous layer
            result[j] -= dEdOdOdO * self.weights.get()[i, j]
            # Change weight proportional to activation
            self.weights.get()[i, j] -= step * (dW)
            
            #echo result[j]
    for i in 0..(self.prevLen - 1):
        result[i] = prev.neurons[i, 0] + result[i] / selfLen.float64

proc train*(self: NeuralNetwork, step: float64, sample: Sample): float64 =
    let outLen = self.layers[self.layers.len - 1].neurons.column(0).len
    var target: seq[float64]
    newSeq(target, outLen)
    let predict = self.run(sample.image).get()
    var loss: float64 = 0
    for i in 0..(outLen - 1):
        target[i] = if i.byte == sample.label:
            1
        else:
            0
        # Record the actual loss of this pass
        loss += crossEntropy(target[i], predict[i])
    for i in 0..(self.layers.len - 2):
        let index = self.layers.len - i - 1
        target = self.layers[index].backPropagate(self.layers[index - 1], target, step)
    return loss / outLen.float64
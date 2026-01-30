using BasicNeuralNetwork.Extensions;
using BasicNeuralNetwork.Models;
using MathNet.Numerics.LinearAlgebra;

var nn = CreateNeuralNetwork(new List<int> { 2, 2, 2, 1 });
nn = ConnectNeuralNetwork(nn);
nn.Dump("Neural Network Structure");

var inputs = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
inputs.Dump("Input Matrix");

var outputs = Vector<double>.Build.DenseOfEnumerable(new List<double> { 1, 1, 1, 0 });
outputs.Dump("Output Vector");

// number of interactions for training
int epochs = 2000;

// essentially for learning in neural networks
double learningRate = 0.1;

nn = Train(nn, epochs, learningRate, inputs, outputs);
nn.Dump("Trained Neural Network");

// NAND gate prediction
nn = Predict(nn, Vector<double>.Build.DenseOfArray(new double[] { 1, 0 }));
nn.Prediction.Dump("NAND Gate Prediction for [1, 1]");

/// <summary>
/// Creates the neural network.
/// </summary>
/// <param name="numOfLayers">The number of layers.</param>
/// <returns>BasicNeuralNetwork.Models.NeuralNetwork.</returns>
NeuralNetwork CreateNeuralNetwork(List<int> numOfNeuronsInEachLayer)
{
    return new NeuralNetwork(numOfNeuronsInEachLayer);
}

NeuralNetwork ConnectNeuralNetwork(NeuralNetwork nn)
{
    // for each layer, it takes all the neurons and connects them to the neurons in the next layer
    nn.Layers.Take(nn.Layers.Count - 1).Select((layer, index) =>
    {
        layer.Neurons.ForEach(neuron =>
        {
            neuron.Connections.Select((connection, connIndex) =>
            {
                connection.TargetNeuron = nn.Layers[index + 1].Neurons[connIndex];
                return connection;
            }).ToList();
        });
        return layer;
    }).ToList();

    return nn;
}

#region Training the Neural Network

/// <summary>
/// Trains the specified neural network.
/// </summary>
/// <param name="nn">The neural network.</param>
/// <param name="epochs">The epochs.</param>
/// <param name="learningRate">The learning rate.</param>
/// <param name="inputs">The inputs.</param>
/// <param name="outputs">The outputs.</param>
/// <returns>BasicNeuralNetwork.Models.NeuralNetwork.</returns>
NeuralNetwork Train(NeuralNetwork nn, int epochs, double learningRate, Matrix<double> inputs, Vector<double> outputs)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int trainRowIndex = 0; trainRowIndex < inputs.RowCount; trainRowIndex++)
        {
            nn = FeedForward(nn, inputs.Row(trainRowIndex));
            nn = BackwardPass(nn, outputs[trainRowIndex]);
            nn = AdjustWeights(nn, learningRate);
        }
    }

    return nn;
}

#endregion

NeuralNetwork AdjustWeights(NeuralNetwork nn, double learningRate)
{
    nn.Layers.ToList().ForEach(layer =>
    {
        layer.Neurons!.ForEach(neuron =>
        {
            if (!layer.IsOutputLayer)
            {
                if (!layer.IsInputLayer)
                    neuron.Bias -= learningRate * neuron.LocalDelta;

                neuron.Connections!.ForEach(connection =>
                {
                    connection.Weight -= learningRate * connection.TargetNeuron!.LocalDelta * neuron.Output;
                });
            }
        });
    });
    return nn;
}

NeuralNetwork BackwardPass(NeuralNetwork nn, double correctOutput)
{
    // output neuron's derivative
    var outputNeuron = nn.Layers.Last().Neurons!.First();
    var lossDerivative = 2 * (nn.Prediction - correctOutput);

    var outputActFuncDerivative = outputNeuron.Activation!.Derivative(outputNeuron.Input);
    var localDelta = lossDerivative * outputActFuncDerivative;

    outputNeuron.LocalDelta = localDelta;

    // rest of the neurons
    nn.Layers.Reverse();
    nn.Layers.Skip(1).ToList().ForEach(layer =>
    {
        layer.Neurons!.ForEach(neuron =>
        {
            // calculate the local delta (error gradient) based on the next layer neuron deltas
            neuron.LocalDelta = neuron.Connections!.Sum(con => con.Weight * con.TargetNeuron!.LocalDelta);
            neuron.LocalDelta *= neuron.Activation!.Derivative(neuron.Input);
        });
    });

    nn.Layers.Reverse();
    return nn;
}

NeuralNetwork FeedForward(NeuralNetwork nn, Vector<double> inputRow)
{
    // see the input values for the first layer
    for (int i = 0; i  < nn.Layers[0].Neurons!.Count; i++)
    {
        nn.Layers[0]!.Neurons[i].Input = inputRow[i];
        nn.Layers[0]!.Neurons[i].Output = inputRow[i];
    }

    for (int layerIndex = 1; layerIndex < nn.Layers.Count; layerIndex++)
    {
        var previousLayer = nn.Layers[layerIndex - 1];
        var currentLayer = nn.Layers[layerIndex];

        for (int neuronIndex = 0; neuronIndex < currentLayer.Neurons!.Count; neuronIndex++)
        {
            var neuron = currentLayer.Neurons[neuronIndex];
            neuron.Input = 0;

            // calculate the weight of the neuron based on the previous layer's outputs
            for (int prevNeuronIndex = 0; prevNeuronIndex < previousLayer.Neurons!.Count; prevNeuronIndex++)
            {
                var prevNeuron = previousLayer.Neurons[prevNeuronIndex];
                neuron.Input += prevNeuron.Output * prevNeuron.Connections[neuronIndex].Weight;
            }

            neuron.Input += neuron.Bias;
            neuron.Output = neuron.Activation!.Activate(neuron.Input);
            if(currentLayer.IsOutputLayer)
                nn.Prediction = neuron.Output;
        }
    }
    return nn;
}

NeuralNetwork Predict(NeuralNetwork nn, Vector<double> inputRow)
{
    nn = FeedForward(nn, inputRow);
    Console.WriteLine($"Prediction for input [{string.Join(", ", inputRow.ToArray())}] is: {nn.Prediction:F4}");
    return nn;
}
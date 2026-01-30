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
int epochs = 100;

// essentially for learning in neural networks
double learningRate = 0.01;

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
                connection.TargetNeuron = nn.Layers[connIndex + 1].Neurons[connIndex];
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
            //nn = FeedForward();
            //nn = BackwardPass();
            //nn = AdjustWeights();
        }
    }

    return nn;
}

#endregion

NeuralNetwork FeedForward(NeuralNetwork nn, Vector<double> inputRow)
{
    return nn;
}
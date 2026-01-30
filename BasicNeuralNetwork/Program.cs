using BasicNeuralNetwork.Models;

var nn = CreateNeuralNetwork(new List<int> { 2, 2, 2, 1 });
nn = ConnectNeuralNetwork(nn);

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
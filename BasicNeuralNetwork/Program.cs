using BasicNeuralNetwork.Models;

var nn = CreateNeuralNetwork(4);

/// <summary>
/// Creates the neural network.
/// </summary>
/// <param name="numOfLayers">The number of layers.</param>
/// <returns>BasicNeuralNetwork.Models.NeuralNetwork.</returns>
NeuralNetwork CreateNeuralNetwork(int numOfLayers)
{
    return new NeuralNetwork(numOfLayers);
}
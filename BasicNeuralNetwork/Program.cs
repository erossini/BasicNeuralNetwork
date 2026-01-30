using BasicNeuralNetwork.Models;

var nn = CreateNeuralNetwork(new List<int> { 1, 4, 4, 1 });

/// <summary>
/// Creates the neural network.
/// </summary>
/// <param name="numOfLayers">The number of layers.</param>
/// <returns>BasicNeuralNetwork.Models.NeuralNetwork.</returns>
NeuralNetwork CreateNeuralNetwork(List<int> numOfNeuronsInEachLayer)
{
    return new NeuralNetwork(numOfNeuronsInEachLayer);
}
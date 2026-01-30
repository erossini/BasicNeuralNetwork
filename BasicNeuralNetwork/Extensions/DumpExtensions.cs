using BasicNeuralNetwork.Models;
using MathNet.Numerics.LinearAlgebra;
using System.Text;

namespace BasicNeuralNetwork.Extensions;

public static class DumpExtensions
{
    public static T Dump<T>(this T obj, string? label = null)
    {
        if (!string.IsNullOrEmpty(label))
            Console.WriteLine($"--- {label} ---");

        Console.WriteLine(obj switch
        {
            NeuralNetwork nn => FormatNeuralNetwork(nn),
            Layer layer => FormatLayer(layer, 0),
            Neuron neuron => FormatNeuron(neuron, 0, 0),
            Matrix<double> matrix => FormatMatrix(matrix),
            _ => obj?.ToString() ?? "null"
        });

        return obj;
    }

    private static string FormatNeuralNetwork(NeuralNetwork nn)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"NeuralNetwork: {nn.Layers?.Count ?? 0} layers, Prediction: {nn.Prediction}");

        if (nn.Layers is not null)
        {
            for (int i = 0; i < nn.Layers.Count; i++)
                sb.AppendLine(FormatLayer(nn.Layers[i], i));
        }

        return sb.ToString();
    }

    private static string FormatLayer(Layer layer, int layerIndex)
    {
        var sb = new StringBuilder();
        var layerType = layer.IsInputLayer ? "Input" : layer.IsOutputLayer ? "Output" : "Hidden";
        sb.AppendLine($"  Layer {layerIndex} ({layerType}): {layer.Neurons?.Count ?? 0} neurons");

        if (layer.Neurons is not null)
        {
            for (int i = 0; i < layer.Neurons.Count; i++)
                sb.Append(FormatNeuron(layer.Neurons[i], layerIndex, i));
        }

        return sb.ToString();
    }

    private static string FormatNeuron(Neuron neuron, int layerIndex, int neuronIndex)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"    Neuron {neuronIndex}: Bias={neuron.Bias:F4}, Input={neuron.Input:F4}, Output={neuron.Output:F4}, Connections={neuron.Connections?.Count ?? 0}");
        return sb.ToString();
    }

    private static string FormatMatrix(Matrix<double> matrix)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Matrix [{matrix.RowCount}x{matrix.ColumnCount}]:");

        for (int row = 0; row < matrix.RowCount; row++)
        {
            sb.Append("  [");
            sb.Append(string.Join(", ", Enumerable.Range(0, matrix.ColumnCount).Select(col => matrix[row, col].ToString("F4"))));
            sb.AppendLine("]");
        }

        return sb.ToString();
    }
}
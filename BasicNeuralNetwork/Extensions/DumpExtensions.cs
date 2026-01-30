using BasicNeuralNetwork.Models;
using MathNet.Numerics.LinearAlgebra;
using System.Text;

namespace BasicNeuralNetwork.Extensions;

public static class DumpExtensions
{
    private static readonly ConsoleColor[] LevelColors =
    [
        ConsoleColor.Cyan,      // Level 0: NeuralNetwork / Matrix
        ConsoleColor.Green,     // Level 1: Layer
        ConsoleColor.Yellow,    // Level 2: Neuron
        ConsoleColor.Magenta,   // Level 3: Connection
        ConsoleColor.Blue       // Level 4: Target
    ];

    public static T Dump<T>(this T obj, string? label = null, bool showConnections = true)
    {
        if (!string.IsNullOrEmpty(label))
        {
            WriteColored($"--- {label} ---", ConsoleColor.White);
            Console.WriteLine();
        }

        switch (obj)
        {
            case NeuralNetwork nn:
                FormatNeuralNetwork(nn, showConnections);
                break;
            case Layer layer:
                FormatLayer(layer, 0, showConnections);
                break;
            case Neuron neuron:
                FormatNeuron(neuron, 0, showConnections);
                break;
            case Connection connection:
                FormatConnection(connection, 0);
                break;
            case Matrix<double> matrix:
                FormatMatrix(matrix);
                break;
            default:
                Console.WriteLine(obj?.ToString() ?? "null");
                break;
        }

        return obj;
    }

    private static void WriteColored(string text, ConsoleColor color)
    {
        var originalColor = Console.ForegroundColor;
        Console.ForegroundColor = color;
        Console.Write(text);
        Console.ForegroundColor = originalColor;
    }

    private static void WriteLineColored(string text, int level)
    {
        var color = LevelColors[Math.Min(level, LevelColors.Length - 1)];
        WriteColored(text, color);
        Console.WriteLine();
    }

    private static void FormatNeuralNetwork(NeuralNetwork nn, bool showConnections)
    {
        WriteLineColored($"NeuralNetwork: {nn.Layers?.Count ?? 0} layers, Prediction: {nn.Prediction}", 0);

        if (nn.Layers is not null)
        {
            for (int i = 0; i < nn.Layers.Count; i++)
                FormatLayer(nn.Layers[i], i, showConnections);
        }
    }

    private static void FormatLayer(Layer layer, int layerIndex, bool showConnections)
    {
        var layerType = layer.IsInputLayer ? "Input" : layer.IsOutputLayer ? "Output" : "Hidden";
        WriteLineColored($"  Layer {layerIndex} ({layerType}): {layer.Neurons?.Count ?? 0} neurons", 1);

        if (layer.Neurons is not null)
        {
            for (int i = 0; i < layer.Neurons.Count; i++)
                FormatNeuron(layer.Neurons[i], i, showConnections);
        }
    }

    private static void FormatNeuron(Neuron neuron, int neuronIndex, bool showConnections)
    {
        WriteLineColored($"    Neuron {neuronIndex}: Bias={neuron.Bias:F4}, Input={neuron.Input:F4}, Output={neuron.Output:F4}, Connections={neuron.Connections?.Count ?? 0}", 2);

        if (showConnections && neuron.Connections is not null && neuron.Connections.Count > 0)
        {
            for (int i = 0; i < neuron.Connections.Count; i++)
                FormatConnection(neuron.Connections[i], i);
        }
    }

    private static void FormatConnection(Connection connection, int connectionIndex)
    {
        WriteLineColored($"      Connection {connectionIndex}: Weight={connection.Weight:F4}", 3);

        if (connection.TargetNeuron is not null)
        {
            var target = connection.TargetNeuron;
            WriteLineColored($"        -> Target: Id={target.Id.ToString()[..8]}..., Bias={target.Bias:F4}, Input={target.Input:F4}, Output={target.Output:F4}", 4);
        }
        else
        {
            WriteLineColored("        -> Target: null", 4);
        }
    }

    private static void FormatMatrix(Matrix<double> matrix)
    {
        WriteLineColored($"Matrix [{matrix.RowCount}x{matrix.ColumnCount}]:", 0);

        for (int row = 0; row < matrix.RowCount; row++)
        {
            var rowText = "  [" + string.Join(", ", Enumerable.Range(0, matrix.ColumnCount).Select(col => matrix[row, col].ToString("F4"))) + "]";
            WriteLineColored(rowText, 1);
        }
    }
}
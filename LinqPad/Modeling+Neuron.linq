<Query Kind="Statements" />

var nn = CreateNeuralNetwork(4);
nn.Dump();

NeuralNetwork CreateNeuralNetwork(int numOfLayers)
{
	return new NeuralNetwork(numOfLayers);
}

public class NeuralNetwork
{
	public List<Layer> Layers { get; set; }
	public double Prediction { get; set; }
	public ILossFunction Loss { get; set; }
	public NeuralNetwork(int numOfLayers)
	{
		Layers = new List<Layer>();
		for (int i = 0; i < numOfLayers; i++) Layers.Add(new Layer());
	}

}
public class Layer
{
	public List<Neuron> Neurons { get; set; }
	public bool IsInputLayer { get; set; }
	public bool IsOutputLayer { get; set; }
	
	public Layer(int numOfNeuronsInLayer)
	{
		Neurons = new List<Neuron>();
		for(int i=0; i<numOfNeuronsInLayer;i++)Neurons.Add(new Neuron())
	}

}
public class Neuron
{
	public Guid Id {get; set;}
	public double Input {get; set; }
	public double Output {get; set; }
	public double Bias {get; set;}
	public List<Connection> Connections {get; set; }
	
	public IActivationFunction Activation{get;set;}
	public Neuron(IActivationFunction activation)
	{
		Bias = 0;
		Activation = activation;
		Connections = new List<Connection>();
	}
	
}
public class Connection{
	
}
public interface ILossFunction
{

}
public interface IActivationFunction
{

}
<Query Kind="Statements" />

var nn = CreateNeuralNetwork(new List<int>(){1,4,4,1});
nn.Dump();

NeuralNetwork CreateNeuralNetwork(List<int> numOfNeuronsInEachLayer) // {1,4,4,1}
{
	return new NeuralNetwork(numOfNeuronsInEachLayer);
}

public class NeuralNetwork
{
	public List<Layer> Layers { get; set; }
	public double Prediction { get; set; }
	public ILossFunction Loss { get; set; }
	public NeuralNetwork(List<int> numOfNeuronsInEachLayer)
	{
		Layers = new List<Layer>();
		//for (int i = 0; i < numOfNeuronsInEachLayer.Count(); i++) Layers.Add(new Layer());
		Layers = numOfNeuronsInEachLayer.Select((neuronsInLayer, index) => {   //{1,4,4,1}
			//Determine the number of neurons in the next layer(if it exists)
			int num0fneuronsInNextLayer = (index < numOfNeuronsInEachLayer.Count -1) ? numOfNeuronsInEachLayer[index+1] : 0;
			return new Layer(neuronsInLayer,num0fneuronsInNextLayer);
		}).ToList();;
	}

}
public class Layer
{
	public List<Neuron> Neurons { get; set; }
	public bool IsInputLayer { get; set; }
	public bool IsOutputLayer { get; set; }

	public Layer(int numOfNeuronsInLayer, int num0fneuronsInNextLayer)
	{
		if(num0fneuronsInNextLayer == 0) IsOutputLayer = true;
		Neurons = new List<Neuron>();
		for (int i = 0; i < numOfNeuronsInLayer; i++) Neurons.Add(new Neuron());
	}

}
public class Neuron
{
	public Guid Id { get; set; }
	public double Input { get; set; }
	public double Output { get; set; }
	public double Bias { get; set; }
	public List<Connection> Connections { get; set; }

	public IActivationFunction Activation { get; set; }
	public Neuron(IActivationFunction activation)
	{
		Id = Guid.NewGuid();
		Bias = 0;
		Activation = activation;
		Connections = new List<Connection>();
	}

}
public class Connection
{

}
public interface ILossFunction
{

}
public interface IActivationFunction
{
	public double Activate(double x);
}







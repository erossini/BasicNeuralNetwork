<Query Kind="Statements" />

var nn = CreateNeuralNetwork(new List<int>() { 1, 4, 4, 1 });
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
		Layers = numOfNeuronsInEachLayer.Select((neuronsInLayer, index) =>
		{   //{1,4,4,1}
			//Determine the number of neurons in the next layer(if it exists)
			int num0fneuronsInNextLayer = (index < numOfNeuronsInEachLayer.Count - 1) ? numOfNeuronsInEachLayer[index + 1] : 0;
			var isInputLayer = index == 0 ? true : false;
			return new Layer(neuronsInLayer, num0fneuronsInNextLayer, isInputLayer);
		}).ToList(); ;
	}

}
public class Layer
{
	public List<Neuron> Neurons { get; set; }
	public bool IsInputLayer { get; set; }
	public bool IsOutputLayer { get; set; }

	public Layer(int numOfNeuronsInLayer, int num0fneuronsInNextLayer, bool isInputLayer)
	{
		if (num0fneuronsInNextLayer == 0) IsOutputLayer = true;
		IsInputLayer = isInputLayer;
		IActivationFunction activation = IsOutputLayer ? new LogisticSigmoid() : new Tanh();
		activation = IsInputLayer ? new NoActivation() : activation;
		Neurons = new List<Neuron>();
		for (int i = 0; i < numOfNeuronsInLayer; i++) Neurons.Add(new Neuron(activation));
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

public class LogisticSigmoid : IActivationFunction
{
	public double Activate(double x)
	{
		return 1.0 / (1.0 + Math.Exp(-x));//compresses the input it receives into output between 0 and 1 (0.5> output is 1)
	}
}
public class Tanh : IActivationFunction
{
	public double Activate(double x)
	{
		return Math.Tanh(x);//compresses the input it receives into output between -1 and 1..used in our hidden layers
	}
}
public class NoActivation : IActivationFunction
{
	public double Activate(double x)
	{
		return x;
	}
}







<Query Kind="Statements">
  <NuGetReference>MathNet.Numerics</NuGetReference>
  <Namespace>MathNet.Numerics.LinearAlgebra</Namespace>
</Query>

var nn = CreateNeuralNetwork(new List<int>() { 2, 2, 2, 1 });
nn = ConnectNeuralNetwork(nn);
var inputs = Matrix<double>.Build.DenseOfArray(new double[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
var outputs = Vector<double>.Build.DenseOfEnumerable(new List<double>() { 0, 1, 1, 0 });//XOR f(x,y)=x*x +y*y+3																			
int epochs = 150;
double learningRate = 0.1;
var graphData = new GraphData();
graphData.Data = new List<Data>();
nn = Train(nn, epochs, learningRate, inputs, outputs);
nn = Predict(nn, Vector<double>.Build.DenseOfArray(new double[] { 0, 1 }));//
nn.Prediction.Dump();
graphData.Data.Chart(c=>c.Epoch,c=>c.Loss,Util.SeriesType.Line).Dump();

NeuralNetwork Predict(NeuralNetwork nn, Vector<double> input)
{
	nn = FeedForward(nn, input);
	return nn;
}
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
		graphData.Data.Add(new Data { Loss=nn.CurrentLoss,Epoch=epoch});
	}
	return nn;
}
NeuralNetwork CreateNeuralNetwork(List<int> numOfNeuronsInEachLayer)
{
	return new NeuralNetwork(numOfNeuronsInEachLayer);
}
NeuralNetwork ConnectNeuralNetwork(NeuralNetwork nn)
{
	nn.Layers.Take(nn.Layers.Count - 1).Select((layer, index) =>
	{
		layer.Neurons.ForEach(neuron =>
		{
			neuron.Connections.Select((connection, connIndex) =>
			{
				connection.TargetedNeuron = nn.Layers[index + 1].Neurons[connIndex];
				return connection;
			}).ToList();
		});
		return layer;
	}).ToList();
	return nn;
}
NeuralNetwork FeedForward(NeuralNetwork nn, Vector<double> inputRow)
{
	//see the input values for the first layer
	for (int i = 0; i < nn.Layers[0].Neurons.Count; i++)
	{
		nn.Layers[0].Neurons[i].Input = inputRow[i];
		nn.Layers[0].Neurons[i].Output = inputRow[i];
	}
	for (int layerIndex = 1; layerIndex < nn.Layers.Count; layerIndex++)
	{

		var previousLayer = nn.Layers[layerIndex - 1];
		var currentLayer = nn.Layers[layerIndex];
		for (int neuronIndex = 0; neuronIndex < currentLayer.Neurons.Count; neuronIndex++)
		{
			var neuron = currentLayer.Neurons[neuronIndex];
			neuron.Input = 0;
			//calculate the weighted sum for this neuron based on the previos layers outputs
			for (int prevNeuronIndex = 0; prevNeuronIndex < previousLayer.Neurons.Count; prevNeuronIndex++)
			{
				var prevNeuron = previousLayer.Neurons[prevNeuronIndex];
				neuron.Input += prevNeuron.Output * prevNeuron.Connections[neuronIndex].Weight;
			}
			neuron.Input += neuron.Bias;
			neuron.Output = neuron.Activation.Activate(neuron.Input);
			if (currentLayer.IsOutputLayer) nn.Prediction = neuron.Output;
		}
	}

	return nn;

}
NeuralNetwork BackwardPass(NeuralNetwork nn, double correctOutput)
{
	//output neuron's derivative
	var outputNeuron = nn.Layers.Last().Neurons.First();
	nn.CurrentLoss = Math.Pow(correctOutput-nn.Prediction,2);
	var lossDerative = 2 * (nn.Prediction - correctOutput);//MSE derivative
	var outputActFuncDerivativative = outputNeuron.Activation.Derivative(outputNeuron.Input);
	var localDelta = lossDerative * outputActFuncDerivativative;
	outputNeuron.LocalDelta = localDelta;
	//rest of the neurons
	nn.Layers.Reverse();
	nn.Layers.Skip(1).ToList().ForEach(layer =>
	{
		layer.Neurons.ForEach(neuron =>
		{
			//calculate the local delta(error gradient) based on next layer neurons deltas
			neuron.LocalDelta = neuron.Connections.Sum(con => con.Weight * con.TargetedNeuron.LocalDelta);
			neuron.LocalDelta = neuron.LocalDelta * neuron.Activation.Derivative(neuron.Input);
		});

	});
	nn.Layers.Reverse();
	return nn;
}
NeuralNetwork AdjustWeights(NeuralNetwork nn, double learningRate)
{
	nn.Layers.ToList().ForEach(layer =>
	{
		layer.Neurons.ForEach(neuron =>
		{
			if (!layer.IsOutputLayer)
			{
				if (!layer.IsInputLayer) neuron.Bias -= learningRate * neuron.LocalDelta;
				neuron.Connections.ForEach(conn =>
				{
					conn.Weight -= learningRate * conn.TargetedNeuron.LocalDelta * neuron.Output;
				});

			}
		});
	});
	return nn;
}
public class NeuralNetwork
{
	public List<Layer> Layers { get; set; }
	public double Prediction { get; set; }
	public ILossFunction Loss { get; set; }
	public double CurrentLoss{get;set;}
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
		for (int i = 0; i < numOfNeuronsInLayer; i++) Neurons.Add(new Neuron(activation, num0fneuronsInNextLayer));
	}

}
public class Neuron
{
	public Guid Id { get; set; }
	public double Input { get; set; }
	public double Output { get; set; }
	public double Bias { get; set; }
	public double LocalDelta { get; set; }
	public List<Connection> Connections { get; set; }

	public IActivationFunction Activation { get; set; }
	public Neuron(IActivationFunction activation, int num0fneuronsInNextLayer)
	{
		Id = Guid.NewGuid();
		Bias = 0;
		Activation = activation;
		Connections = new List<Connection>();
		for (int i = 0; i < num0fneuronsInNextLayer; i++) Connections.Add(new Connection());
	}

}
public class Connection
{
	public double Weight { get; set; }
	public Neuron TargetedNeuron { get; set; }
	public Connection()
	{
		var random = new Random();
		Weight = random.NextDouble();
	}
}
public interface ILossFunction
{

}
public interface IActivationFunction
{
	public double Activate(double x);
	public double Derivative(double x);
}

public class LogisticSigmoid : IActivationFunction
{
	public double Activate(double x)
	{
		return 1.0 / (1.0 + Math.Exp(-x));//compresses the input it receives into output between 0 and 1 (0.5> output is 1)
	}
	public double Derivative(double x)
	{
		var sigmoid = Activate(x);
		return sigmoid * (1.0 - sigmoid);
	}
}
public class Tanh : IActivationFunction
{
	public double Activate(double x)
	{
		return Math.Tanh(x);//compresses the input it receives into output between -1 and 1..used in our hidden layers
	}
	public double Derivative(double x)
	{
		var tanh = Activate(x);
		return 1 - 0 - (tanh * tanh);
	}

}
public class NoActivation : IActivationFunction
{
	public double Activate(double x)
	{
		return x;
	}

	public double Derivative(double x)
	{
		return 1.0;
	}
}
public class GraphData{
	public List<Data> Data{get;set;}
}
public class Data{
	public double Epoch{get;set;}
	public double Loss{get;set;}
}







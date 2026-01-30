<Query Kind="Statements" />

var nn = CreateNeuralNetwork();


NeuralNetwork CreateNeuralNetwork()
{
	return new NeuralNetwork();
}

public class NeuralNetwork{
	public List<Layer> Layers{get;set;}
	public double Prediction {get;set;}
	public ILossFunction Loss{get;set;}
	
}
public class Layer{
	
}
public interface ILossFunction{
	
}
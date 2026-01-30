using BasicNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Models
{
    /// <summary>
    /// Class Neuron.
    /// </summary>
    public class Neuron
    {
        public Guid Id { get; set; }
        public IActivationFunction? Activation { get; set; }
        public double Bias { get; set; }
        public double Input { get; set; }
        public double Output { get; set; }
        public List<Connection>? Connections { get; set; }

        public Neuron(IActivationFunction activation, int numOfNeuronsInNextLayer)
        {
            Id = Guid.NewGuid();
            Activation = activation;
            Bias = 0;
            Connections = new List<Connection>();

            for (int i = 0; i < numOfNeuronsInNextLayer; i++)
                Connections.Add(new Connection());
        }
    }
}
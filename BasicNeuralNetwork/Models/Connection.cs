using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Models
{
    /// <summary>
    /// Class Connection.
    /// </summary>
    public class Connection
    {
        public double Weight { get; set; }
        public Neuron? TargetNeuron { get; set; }

        public Connection()
        {
            var randon = new Random();
            Weight = randon.NextDouble();
        }
    }
}

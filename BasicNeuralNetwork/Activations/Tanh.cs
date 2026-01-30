using BasicNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Activations
{
    public class Tanh : IActivationFunction
    {
        public double Activate(double x)
        {
            // compress the input it receives into outputs between -1 and 1 used in our hidden layers
            return Math.Tanh(x);
        }

        public double Derivative(double x)
        {
            var tanh = Activate(x);
            return 1 - tanh * tanh;
        }
    }
}
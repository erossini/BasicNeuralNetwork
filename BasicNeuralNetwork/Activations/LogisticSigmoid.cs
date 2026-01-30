using BasicNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Activations
{
    public class LogisticSigmoid : IActivationFunction
    {
        public double Activate(double x)
        {
            // compress the input it receives into outputs between 0 and 1 (0.5 output is 1
            return 1 / (1 + Math.Exp(-x));
        }
    }
}
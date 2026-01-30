using BasicNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Activations
{
    public class NoActivation : IActivationFunction
    {
        public double Activate(double x)
        {
            // no activation function applied
            return x;
        }
    }
}

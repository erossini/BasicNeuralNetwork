using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Interfaces
{
    /// <summary>
    /// Interface IActivationFunction
    /// </summary>
    public interface IActivationFunction
    {
        public double Activate(double x);
        public double Derivative(double x);
    }
}
using BasicNeuralNetwork.Activations;
using BasicNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Models
{
    /// <summary>
    /// Class Layer.
    /// </summary>
    public class Layer
    {
        /// <summary>
        /// Gets or sets the neurons.
        /// </summary>
        /// <value>The neurons.</value>
        public List<Neuron>? Neurons { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this instance is input layer.
        /// </summary>
        /// <value><c>true</c> if this instance is input layer; otherwise, <c>false</c>.</value>
        public bool IsInputLayer { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this instance is output layer.
        /// </summary>
        /// <value><c>true</c> if this instance is output layer; otherwise, <c>false</c>.</value>
        public bool IsOutputLayer { get; set; }

        public Layer(int numOfNeurons, int numOfNeuronsInNextLayer, bool isInputLayer)
        {
            if (numOfNeuronsInNextLayer == 0)
                IsOutputLayer = true;

            IsInputLayer = isInputLayer;

            IActivationFunction activationFunction = IsOutputLayer ? new LogisticSigmoid() : new Tanh();
            activationFunction = IsInputLayer ? new NoActivation() : activationFunction;

            Neurons = new List<Neuron>();
            for (int i = 0; i < numOfNeurons; i++)
                Neurons.Add(new Neuron(activationFunction));
        }
    }
}
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
    }
}

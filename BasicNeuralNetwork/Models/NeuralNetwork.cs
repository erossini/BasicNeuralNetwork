using BasicNeuralNetwork.Interfaces;
using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeuralNetwork.Models
{
    /// <summary>
    /// Class NeuralNetwork.
    /// </summary>
    public class NeuralNetwork
    {
        /// <summary>
        /// Gets or sets the layers.
        /// </summary>
        /// <value>The layers.</value>
        public List<Layer>? Layers { get; set; }

        /// <summary>
        /// Gets or sets the prediction.
        /// </summary>
        /// <value>The prediction.</value>
        public double Prediction { get; set; }

        /// <summary>
        /// Gets or sets the loss function.
        /// </summary>
        /// <value>The loss function.</value>
        public ILossFunction? LossFunction { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNetwork"/> class.
        /// </summary>
        /// <param name="numOfLayers">The number of layers.</param>
        public NeuralNetwork(int numOfLayers) {
            Layers = new List<Layer>();
            Enumerable.Range(0, numOfLayers).ToList().ForEach(_ => Layers!.Add(new Layer()));
        }
    }
}

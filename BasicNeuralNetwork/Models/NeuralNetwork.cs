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
        public NeuralNetwork(List<int> numOfNeuronsInEachLayer) {
            Layers = new List<Layer>();
            Layers = numOfNeuronsInEachLayer.Select((neuronsInLayer, index) => {
                // determine the number of neurons in the next layer (if it exists)
                int neuronsInNextLayer = (index < numOfNeuronsInEachLayer.Count - 1) ? numOfNeuronsInEachLayer[index + 1] : 0;
                bool isInputLayer = index == 0;

                return new Layer(neuronsInLayer, neuronsInNextLayer, isInputLayer);
            }).ToList();
        }
    }
}

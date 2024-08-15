"""
This file contains a very simple TDNN module to use for speaker-id.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.

Authors
 * Nauman Dawalatabad 2020
 * Mirco Ravanelli 2020
"""

import torch  # noqa: F401
import torch.nn as nn
import speechbrain as sb


class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of xvector features.
    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 192])
    >>> classify = Classifier(input_shape=input_feats.shape)
    >>> output = classify(input_feats)
    >>> output.shape
    torch.Size([5, 28])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=192,
        out_neurons=28,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        self.append(
            sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        )

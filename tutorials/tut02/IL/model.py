import torch
from torch import nn


class MLP(nn.Module):
    """
    Multilayer perceptron with ReLU activations
    https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron
    """

    def __init__(self, in_features, hidden_dims, out_features):
        """
        initializes the MLP modules
        :param in_features: the number of input features
        :param hidden_dims: a list of hidden layer output features
        :param out_features: the number of output features
        """
        super().__init__()

        # flatten dimensions
        all_dims = [in_features, *hidden_dims, out_features]

        # create layers
        layers = []
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                nn.ReLU()
            ]

        # remove last non-linearity
        layers = layers[:-1]

        # create sequential model
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        runs a sequence of fully connected layers with ReLU non-linearities.
        :param x: a batch of vectors of shape (batch_size, in_features)
        """
        # flatten batch input to batch of flat inputs
        x = torch.flatten(x, start_dim=1)

        return self.fc_layers(x)

import torch.nn as nn
from .regressor import ACTIVATION_KEY
import torch


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layer,
        num_output,
        dropout_prob,
        activation,
        config_env,
        transformerCall=False,
    ):
        super(MLP, self).__init__()
        """
        Args:
            configuration specific to surrogate model
            env (required for specifying input dimensions of model)

        Initlaise the model given the parameters defined in the config
        """
        self.activation = ACTIVATION_KEY[activation]

        # TODO: this is grid specific for now, make it general (for apatamers and torus)
        self.input_max_length = config_env.n_dim
        self.input_classes = config_env.length
        self.out_dim = num_output

        if transformerCall == False:
            self.hidden_layers = [hidden_dim] * num_layer
            self.dropout_prob = dropout_prob
            self.init_layer_depth = int((self.input_classes) * (self.input_max_length))

        else:
            # this clause is entered only when transformer specific config is passed
            self.hidden_layers = [self.mlp.hidden_dim] * self.mlp.num_layer
            self.dropout_prob = self.mlp.dropout_prob
            self.init_layer_depth = self.embed_dim

        layers = [
            nn.Linear(self.init_layer_depth, self.hidden_layers[0]),
            self.activation,
        ]
        layers += [nn.Dropout(self.dropout_prob)]
        for i in range(1, len(self.hidden_layers)):
            layers.extend(
                [
                    nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]),
                    self.activation,
                    nn.Dropout(self.dropout_prob),
                ]
            )
        layers.append(nn.Linear(self.hidden_layers[-1], self.out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            tensor of dimension batch x max_seq_length_in_batch x k,
            where k is num_tokens if input is required to be one-hot encoded

        """
        # Pads the tensor till maximum length of dataset
        input = torch.zeros(x.shape[0], self.input_max_length*self.input_classes)
        input[:, : x.shape[1]] = x
        # TODO: Check if the following is reqd for aptamers. Reshapes the tensor to batch_size x init_layer_depth
        # x = input.reshape(x.shape[0], -1)
        # Performs a forward call
        return self.model(x)

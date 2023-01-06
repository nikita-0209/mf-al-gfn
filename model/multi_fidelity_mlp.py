import torch.nn as nn
from .mlp import ACTIVATION_KEY
import torch
import torch.nn.functional as F

class MultiFidelityMLP(nn.Module):
    def __init__(self, multi_head, base_num_hidden, base_num_layer, fid_num_hidden, fid_num_layer, activation, dropout_prob, num_output, num_fid, config_env):
        super(MultiFidelityMLP, self).__init__()

        input_classes = config_env.length  
        input_max_length = config_env.n_dim 
        self.num_output = num_output
        self.init_layer_depth = int((input_classes) * (input_max_length))

        self.base_hidden_layers = [base_num_hidden] * base_num_layer
        self.fid_hidden_layers = [fid_num_hidden] * fid_num_layer
        self.activation = ACTIVATION_KEY[activation]
        self.num_fid = num_fid

        embed_layers = [
            nn.Linear(self.init_layer_depth, self.embed_hidden_layers[0]),
            self.activation,
        ]
        embed_layers += [nn.Dropout(self.dropout_prob)]
        for i in range(1, len(self.embed_hidden_layers)):
            embed_layers.extend(
                [
                    nn.Linear(self.embed_hidden_layers[i - 1], self.embed_hidden_layers[i]),
                    self.activation,
                    nn.Dropout(dropout_prob),
                ]
            )
        self.embed_module = nn.Sequential(*embed_layers)

        fid_layers = [
            nn.Linear(self.embed_hidden_layers[-1], self.fid_hidden_layers[0]),
            self.activation,
        ]
        fid_layers += [nn.Dropout(self.dropout_prob)]
        for i in range(1, len(self.fid_hidden_layers)):
            fid_layers.extend(
                [
                    nn.Linear(self.fid_hidden_layers[i - 1], self.fid_hidden_layers[i]),
                    self.activation,
                    nn.Dropout(dropout_prob),
                ]
            )

        for i in range(self.num_fid):
            setattr(self, f"fid_{i}_module", nn.Sequential(*fid_layers))

        if multi_head == True:
            self.forward = self.forward_multiple_head
        else:
            self.forward = self.forward_shared_head

    def preprocess(self, x, fid):
        input = torch.zeros(x.shape[0], self.input_max_length * self.input_classes)
        input[:, : x.shape[1]] = x
        fid_ohe = F.one_hot(fid, num_classes=self.num_fid+1)[:, 1:].to(torch.float32)
        fid_ohe = fid_ohe.to(self.device)
        return input, fid_ohe

    def forward_shared_head(self, inputs, inputLens, fid):
        x, fid_ohe = self.preprocess(inputs, inputLens, fid)
        embed = self.embed_module(x)
        # TODO: Check if the below line is required
        embed = embed.view(x.size(0), -1)
        embed_with_fid = torch.concat([embed, fid_ohe], dim=1)
        out = self.fid_0_module(embed_with_fid)
        return out

    def forward_multiple_head(self, inputs, inputLens, fid):
        """
        Currently compatible with at max two fidelities. 
        Need to extend functionality to more fidelities if need be.
        """
        x, fid_ohe = self.preprocess(inputs, inputLens, fid)
        # For debugging
        # fid_ohe = torch.FloatTensor([1, 0]).repeat(x.shape[0], 1)
        embed = self.embed_module(x)
        # TODO: Check if the below line is required
        embed = embed.view(x.size(0), -1)
        embed_with_fid = torch.concat([embed, fid_ohe], dim=1)
        # [1, 0]
        output_fid_1 = self.fid_0_module(embed_with_fid)
        # [0, 1]
        output_fid_2 = self.fid_1_module(embed_with_fid)
        out = torch.stack([output_fid_1, output_fid_2], dim=-1).squeeze(-2)
        output = out[fid_ohe==1].unsqueeze(dim=-1)
        return output
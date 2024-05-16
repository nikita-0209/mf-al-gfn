import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from gflownet.proxy.base import Proxy


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers,
        dropout_prob,
        device,
        transformerCall=False,
        num_token=None,
        seq_len=None,
    ):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transformerCall = transformerCall
        self.d_numToken = num_token
        self.d_seqLength = seq_len
        self.device = device
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()]
        layers += [nn.Dropout(dropout_prob)]
        for i in range(1, len(hidden_layers)):
            layers.extend(
                [
                    nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
            )
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.transformerCall == False:
            x = self.preprocess(x)
        output = self.model(x)
        return output

    def preprocess(self, inputs):
        inp_x = F.one_hot(inputs, num_classes=self.d_numToken + 1)[:, :, :-1].to(
            torch.float32
        )
        inp = torch.zeros(inputs.shape[0], self.d_seqLength, self.d_numToken)
        inp[:, : inp_x.shape[1], :] = inp_x
        inputs = inp.reshape(inputs.shape[0], -1).to(self.device).detach()
        return inputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(
        self,
        num_hid,
        max_len,
        vocab_size,
        num_outputs,
        dropout,
        num_layers,
        num_head,
        pre_ln,
        factor,
        varLen,
        device,
    ):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 2)
        self.embedding = nn.Embedding(vocab_size + 1, num_hid)
        # Embedding was (26, 128) in that code
        encoder_layers = nn.TransformerEncoderLayer(
            num_hid, num_head, num_hid, dropout=dropout, norm_first=pre_ln
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = MLP(
            num_hid,
            num_outputs,
            [factor * num_hid, factor * num_hid],
            dropout,
            device,
            transformerCall=True,
        )
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        self.num_hid = num_hid
        self.varLen = varLen

    def model_params(self):
        return (
            list(self.pos.parameters())
            + list(self.embedding.parameters())
            + list(self.encoder.parameters())
            + list(self.output.parameters())
        )

    def forward(self, x, mask):
        x = x.to(torch.int64)
        x = self.preprocess(x)
        x = self.embedding(x)  # (100, 64, 128)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        pooled_x = x[0, torch.arange(x.shape[1])]
        y = self.output(pooled_x)
        return y

    def preprocess(self, inputs):
        # inputs need to be integer so that you can feed them into embedding layer and get embeddings for each integer
        x = torch.transpose(inputs, 1, 0)
        return x


class AptamersNN(Proxy):
    def __init__(self, cost, **kwargs):
        super().__init__(**kwargs)
        NUM_TOKEN = 4
        NUM_OUTPUT = 1
        DROPOUT = 0.0

        # Next three variables are specifically for fixed length 30 sequences
        self.MIN = -30
        self.MAX = 0
        self.max_seq_length = 30

        self.cost = cost
        # Came from the dataset creation in ActiveLearningPipeline
        # https://github.com/alexhernandezgarcia/ActiveLearningPipeline/blob/main/oracle.py#L376
        # as the indices were used to train the MLP
        # Amd the indices were assigned in the numbers2letters() function linked above
        self.lookup = {"A": 1, "T": 2, "C": 3, "G": 4}
        NUM_HIDDEN_TRANSFORMER = 1024
        NUM_HEAD = 16
        PRE_LN = True
        FACTOR = 4
        VARIABLE_LEN = False
        NUM_TRANSFORMER_LAYER = 8
#         self.model = Transformer(
#             num_hid=NUM_HIDDEN_TRANSFORMER,
#             max_len=self.max_seq_length,
#             vocab_size=NUM_TOKEN,
#             num_outputs=NUM_OUTPUT,
#             dropout=DROPOUT,
#             num_layers=NUM_TRANSFORMER_LAYER,
#             num_head=NUM_HEAD,
#             pre_ln=PRE_LN,
#             factor=FACTOR,
#             varLen=VARIABLE_LEN,
#             device=self.device,
#         )
#         oraclePath = "/home/mila/n/nikita.saxena/activelearning/storage/dna/length30/oracleSearch_electricSweep/0.0_iter197.pt"
#         self.model.load_state_dict(torch.load(oraclePath))
#         self.model.eval()
#         self.model.to(self.device)

    def __call__(self, sequences):
        """
        Args:
            x: torch.Tensor of shape (batch_size, input_dim)
        Returns:
            torch.Tensor of shape (batch_size, output_dim)
        Transformation (one-hot etc) done within model
        """
        return torch.zeros(len(sequences), device=self.device, dtype=self.float)
        encoded_readable = [
            torch.tensor([self.lookup[el] for el in seq]) for seq in sequences
        ]
        states = torch.nn.utils.rnn.pad_sequence(
            encoded_readable, batch_first=True, padding_value=0.0
        )
        states = states.to(self.device)
        scores = []
        with torch.no_grad():
            for batch in range(0, len(states), 32):
                state_batch = states[batch : batch + 32]
                result = self.model(state_batch, None)
                scores.append(result)
        scores = torch.cat(scores, dim=0)
        scores = scores * (self.MAX - self.MIN) + self.MIN
        """"""
        return scores.squeeze(-1)

import torch
import attrs
from torch import nn, optim
from torch.nn.utils import clip_grad_norm


# @attrs.define(eq=False)
# class simpleLSTM(nn.Module):

#     num_hiddens: int
#     vocab_size: int

#     learning_rate: float = 0.1
#     weight_decay: float = 0.01

#     def __attrs_post_init__(self):
#         super().__init__()
#         self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
#         self.lstm = nn.LSTM(self.vocab_size, self.num_hiddens)
#         self.layer_norm = nn.LayerNorm(self.num_hiddens)
#         self.initialize_parameters(self)

#     def forward(self, inputs, state=None):
#         output, _ = self.lstm(inputs, state)
#         output = self.layer_norm(output)
#         output = self.linear(output)
#         return output

#     def loss(self, y_hat, y):
#         loss_function = nn.CrossEntropyLoss()
#         return loss_function(y_hat, y)

#     def initialize_parameters(self, module):
#         def initialize_xavier(layer):
#             if type(layer) == nn.Linear:
#                 nn.init.xavier_normal_(layer.weight)
#                 nn.init.zeros_(layer.bias)
#         module.apply(initialize_xavier)

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

#     def training_step(self, batch):
#         l = self.loss(self(*batch[:-1]), batch[-1])
#         return l

#     def validation_step(self, batch):
#         l = self.loss(self(*batch[:-1]), batch[-1])
#         return l


@attrs.define(eq=False)
class simpleRNN(nn.Module):

    num_hiddens: int
    vocab_size: int

    learning_rate: float = 0.1
    weight_decay: float = 0.01

    def __attrs_post_init__(self):
        super().__init__()
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        self.rnn = nn.RNN(self.vocab_size, self.num_hiddens)
        self.layer_norm = nn.LayerNorm(self.num_hiddens)
        self.softmax = nn.LogSoftmax(dim=1)
        # self.initialize_parameters(self)

    def forward(self, inputs, state=None):
        output, _ = self.rnn(inputs, state)
        output = self.layer_norm(output)
        output = self.linear(output)
        # output = self.softmax(output)
        return output

    def loss(self, y_hat, y):
        # y_hat: [batch_size, seq_len, vocab_size], y: [batch_size, seq_len]
        y_hat = y_hat.reshape(-1, y_hat.size(-1))
        y = y.reshape(-1)
        loss_function = nn.CrossEntropyLoss()
        return loss_function(y_hat, y)

    def initialize_parameters(self, module):

        def initialize_xavier(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        module.apply(initialize_xavier)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

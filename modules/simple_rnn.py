import torch
import attrs
from torch import nn, optim

@attrs.define(eq=False)
class simpleRNN(nn.Module):

    num_hiddens: int
    vocab_size: int

    learning_rate: float = 0.1
    weight_decay: float = 0.01

    def __attrs_post_init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(self.vocab_size)
        self.rnn = nn.RNN(self.vocab_size, self.num_hiddens)

    def forward(self, inputs, state=None):
        output, _ = self.rnn(inputs, state)
        return self.linear(output).swapaxes(0, 1)

    def loss(self, y_hat, y):
        loss_function = nn.CrossEntropyLoss()
        return loss_function(y_hat, y.type(torch.long))

    def initialize_parameters(self, module):

        def initialize_xavier(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        module.apply(initialize_xavier)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, batch):
        # target is the last element in the batch
        inputs, targets = batch[:-1], batch[-1]
        l = self.loss(self(inputs), targets)
        return l

    def validation_step(self, batch): 
        # target is the last element in the batch
        inputs, targets = batch[:-1], batch[-1]
        l = self.loss(self(inputs), targets)
        return l
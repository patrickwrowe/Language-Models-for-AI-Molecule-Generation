import torch
import attrs
from torch import nn, optim


@attrs.define(eq=False)
class simpleRNN(nn.Module):

    num_hiddens: int
    vocab_size: int
    num_layers: int = 1

    learning_rate: float = 0.1
    weight_decay: float = 0.01

    def __attrs_post_init__(self):
        super().__init__()
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        self.rnn = nn.RNN(self.vocab_size, self.num_hiddens, num_layers = self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.initialize_parameters(self)

    def forward(self, inputs, state=None):
        output, _ = self.rnn(inputs, state)
        output = self.dropout(output)
        output = self.linear(output)
        return output

    def loss(self, y_hat, y):
        loss_function = nn.CrossEntropyLoss()
        return loss_function(y_hat, y)

    def initialize_parameters(self, module):

        def initialize_xavier(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif type(layer) == nn.RNN:
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

        module.apply(initialize_xavier)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, batch):
        # print(batch[0].shape)  # Debugging: Check input shape]
        # print(batch[1].shape)  # Debugging: Check target shape
        preds = self(*batch[:-1])
        labels = batch[-1]
        
        # Flatten for cross-entropy loss
        # preds: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # labels: (batch_size, seq_len) -> (batch_size * seq_len)
        preds = preds.reshape(-1, self.vocab_size)
        labels = labels.reshape(-1)
        
        return self.loss(preds, labels)

    def validation_step(self, batch):
        preds = self(*batch[:-1])
        labels = batch[-1]
        
        # Flatten for cross-entropy loss
        # preds: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # labels: (batch_size, seq_len) -> (batch_size * seq_len)
        preds = preds.reshape(-1, self.vocab_size)
        labels = labels.reshape(-1)
        
        return self.loss(preds, labels)

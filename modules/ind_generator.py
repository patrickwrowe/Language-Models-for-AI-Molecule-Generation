import torch
import attrs
from torch import nn, optim

@attrs.define(eq=False)
class SmilesIndGeneratorRNN(nn.Module):

    vocab_size: int
    num_indications: int
    num_hiddens: int
    num_layers: int

    learning_rate: float = 0.001
    weight_decay: float = 0.01

    dropout_pc: float = 0.2

    def __attrs_post_init__(self):
        super().__init__()

        # One hidden layer per state to prep
        self.ind_to_state = nn.Linear(
            in_features=self.num_indications, 
            out_features=self.num_hiddens
        )

        self.rnn_to_out = nn.Linear(
            self.num_hiddens, self.vocab_size
        )

        self.rnn = nn.RNN(
            self.vocab_size,
            self.num_hiddens,
            num_layers = self.num_layers,
            batch_first = True
        )
        
        self.dropout = nn.Dropout(0.2)

    def init_state(self, ind_tensor: torch.Tensor):
        state = self.ind_to_state(ind_tensor.unsqueeze(0).repeat(self.num_layers, 1, 1))
        return state

    def forward(self, seq_tensor: torch.Tensor, ind_tensor: torch.Tensor):
        # First, condition the state
        state = self.ind_to_state(ind_tensor.unsqueeze(0).repeat(self.num_layers, 1, 1))

        output, state = self.rnn(seq_tensor, state)
        output = self.dropout(output)
        output = self.rnn_to_out(output)
        return output, state

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
        preds, _ = self(*batch[:-1])
        labels = batch[-1]
        
        # Flatten for cross-entropy loss
        preds = preds.reshape(-1, self.vocab_size)
        labels = labels.reshape(-1)
        
        return self.loss(preds, labels)

    def validation_step(self, batch):
        return self.training_step(batch)

    

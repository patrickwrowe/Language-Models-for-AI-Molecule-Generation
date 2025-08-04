import torch
import attrs
from torch import nn
from typing import Optional, Union
from modules import SmilesGenerativeLanguageModel, SmilesGLMConfig

# NOTE: Due to a bug, models defined as attrs classes can't be loaded from weights. 

# NOTE: Model was used as a stepping stone for more functional model and should not be used.
# Leaving this it here as-is for posterity and reference.

@attrs.define(frozen=True)
class simpleRNNConfig(SmilesGLMConfig):
    num_hiddens: int = attrs.field()
    num_layers: int = attrs.field(default=1)
    output_dropout: float = attrs.field(default=0.2)
    rnn_dropout: float = attrs.field(default=0.2)

@attrs.define(eq=False)
class simpleRNN(SmilesGenerativeLanguageModel):

    config: simpleRNNConfig

    def __attrs_post_init__(self, ):
        super().__init__(
            config=self.config
        )

        self.linear = nn.Linear(self.config.num_hiddens, self.config.vocab_size)
        self.rnn = nn.RNN(self.vocab_size, self.num_hiddens, num_layers=self.num_layers, batch_first=True, 
                          dropout=self.config.rnn_dropout, nonlinearity='relu')
        self.dropout = nn.Dropout(self.config.output_dropout)
        self.initialize_parameters(self)

    def forward(self, input: Union[torch.Tensor, tuple[torch.Tensor, ...]], state: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle tuple input by taking first element (for compatibility)
        if isinstance(input, tuple):
            inputs = input[0]
        else:
            inputs = input
            
        output, new_state = self.rnn(inputs, state)
        output = self.dropout(output)
        output = self.linear(output)
        return output, new_state
    

@attrs.define(eq=False)
class simpleLSTM(simpleRNN):

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.rnn = nn.LSTM(self.vocab_size, self.num_hiddens, num_layers=self.num_layers, batch_first=True,
                           dropout=self.rnn_dropout)
        self.initialize_parameters(self)
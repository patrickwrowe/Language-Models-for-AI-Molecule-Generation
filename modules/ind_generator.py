import torch
from torch import nn
from typing import Union
from modules import SmilesGenerativeLanguageModel, SmilesGLMConfig
import utilities
import attrs

@attrs.define(frozen=True)
class SmilesIndGeneratorRNNConfig(SmilesGLMConfig):
    num_indications: int = attrs.field(validator=attrs.validators.ge(1))
    num_hiddens: int = attrs.field(validator=attrs.validators.ge(1))
    num_layers: int = attrs.field(validator=attrs.validators.ge(1))
    output_dropout: float = attrs.field()
    rnn_dropout: float = attrs.field()
    state_dropout: float = attrs.field()


class SmilesIndGeneratorRNN(SmilesGenerativeLanguageModel):

    def __init__(self, config: SmilesIndGeneratorRNNConfig):
        """
        Recurrent architecture with initial condition set by embedding of
        indication for which the drug molecule is intended.
        """
        super().__init__(
            config=config
        )
        
        self.config = config

        # One hidden layer per state to prep
        self.ind_to_h_0 = nn.Linear(
            in_features=self.config.num_indications, 
            out_features=self.config.num_hiddens
        )

        self.ind_to_c_0 = nn.Linear(
            in_features=self.config.num_indications, 
            out_features=self.config.num_hiddens
        )

        self.rnn_to_out = nn.Linear(
            self.config.num_hiddens, self.config.vocab_size
        )

        self.rnn = nn.LSTM(
            self.config.vocab_size,
            self.config.num_hiddens,
            num_layers = self.config.num_layers,
            batch_first = True,
            dropout=self.config.rnn_dropout
        )
        
        self.dropout = nn.Dropout(self.config.output_dropout)
        self.init_state_dropout = nn.Dropout(self.config.state_dropout)
        
        self.in_emb_h0 = nn.Embedding(
            num_embeddings=self.config.num_indications,
            embedding_dim=self.config.num_hiddens
        )

        self.in_emb_c0 = nn.Embedding(
            num_embeddings=self.config.num_indications,
            embedding_dim=self.config.num_hiddens
        )

        self.initialize_parameters(self)

    def init_state(self, input: Union[torch.Tensor, tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
        """
        Initialize LSTM hidden and cell states from indication tensor.
        
        Args:
            ind_tensor: One-hot encoded indication tensor of shape (batch_size, num_indications)
            
        Returns:
            tuple: (h_0, c_0) where each has shape (num_layers, batch_size, num_hiddens)
        """
        
        if isinstance(input, tuple):
            if len(input) == 1:
                input = input[0]
            else:
                raise ValueError(f"Expected exactly one tensor as input, got {len(input)}.")

        # Temporarily cast back to ordinal encoding...
        input_tensor = input.argmax(dim=-1)

        input_tensor = input_tensor.repeat(self.config.num_layers, 1).type(torch.long)

        # Transform indication vector to initial hidden and cell states 
        h_0 = self.init_state_dropout(self.in_emb_h0(input_tensor))
        c_0 = self.init_state_dropout(self.in_emb_c0(input_tensor))

        return (h_0, c_0)

    def forward(self, input: Union[torch.Tensor, tuple[torch.Tensor, ...]], state=None):
        
        if isinstance(input, torch.Tensor):
            seq_tensor = input
            ind_tensor = None
        elif isinstance(input, tuple):
            # Expect exactly 2 tensors for this specific implementation
            if len(input) != 2:
                raise ValueError(f"Expected exactly 2 tensors (seq_tensor, ind_tensor), got {len(input)}")
            seq_tensor, ind_tensor = input
        else: raise ValueError(f"Expected type torch.Tensor or tuple[torch.Tensor], got {type(input)}")

        if state is None and ind_tensor is not None:
            # First, condition the state with indication tensor
            initial_state = self.init_state(ind_tensor)
        else:
            initial_state = state
        
        output, state = self.rnn(seq_tensor, initial_state)
        output = self.dropout(output)
        output = self.rnn_to_out(output)

        return output, state
    




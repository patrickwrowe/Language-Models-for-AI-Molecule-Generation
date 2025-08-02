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


def simple_generate(prefix, num_chars, model, char_to_idx_mapping, idx_to_char_mapping, temperature = 0.0, device=None):
    """
    Simple character-by-character generation function.
    """

    def decode_indices_to_string(encoded_indices: list, idx_to_char_mapping: dict[int, str]):
        decoded = ''.join([idx_to_char_mapping[int(inx)] for inx in encoded_indices])
        return decoded

    def encode_string_to_indices(smiles_string: str, char_to_idx_mapping: dict[str, int]):
        encoded = [char_to_idx_mapping[c] for c in smiles_string]
        return encoded

    model.eval()
    generated = prefix
    
    with torch.no_grad():
        for i in range(num_chars):
            # Encode current text
            encoded = torch.nn.functional.one_hot(torch.tensor(encode_string_to_indices(generated, char_to_idx_mapping)), num_classes=len(char_to_idx_mapping))
            input_tensor = torch.tensor(encoded, device=device, dtype=torch.float32)
            
            # Get prediction
            output = model(input_tensor.unsqueeze(0))  # Add batch dim
            
            # Get most likely next token
            if temperature > 0:
                # Apply temperature scaling
                output = output / temperature
                probabilities = torch.softmax(output, dim=-1)
                next_token = torch.multinomial(probabilities[0, -1, :], num_samples=1).item()
            else:
                # Default to argmax if temperature is 0
                next_token = output[0, -1, :].argmax().item()
            
            # Decode and append
            next_char = decode_indices_to_string([next_token], idx_to_char_mapping)

            if next_char == ' ' or next_char == '': # EOS token
                break

            generated += next_char
            
            # print(f"Step {i+1}: Added '{next_char}' -> '{generated}'")
            
    return generated
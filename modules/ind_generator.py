import torch
from torch import nn
from typing import Union
from modules import SmilesGenerativeLanguageModel, SmilesGLMConfig
import utilities
import attrs

@attrs.define(frozen=True)
class SmilesIndGeneratorRNNConfig(SmilesGLMConfig):
    num_indications: int = attrs.field()
    num_hiddens: int = attrs.field()
    num_layers: int = attrs.field()
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

        self.in_emb_h0 = nn.Embedding(
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
        c_0 = self.init_state_dropout(self.in_emb_h0(input_tensor))

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
    

# Model Sampling

def simple_generate(prefix, num_chars, model, indications_tensor, char_to_idx_mapping, idx_to_char_mapping, temperature = 0.0, device=None):
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
        # Initialize state with indications
        state = model.init_state(indications_tensor.unsqueeze(0).to(device))  # Add batch dim

        # First, process the prefix to get the proper state
        if len(prefix) > 0:
            prefix_encoded = encode_string_to_indices(prefix, char_to_idx_mapping)
            prefix_tensor = torch.nn.functional.one_hot(
                torch.tensor(prefix_encoded), 
                num_classes=len(char_to_idx_mapping)
            ).float().to(device)
            
            # Process prefix through model to get proper state
            _, state = model(prefix_tensor.unsqueeze(0), state=state)
        
        # Now generate new characters one by one
        for i in range(num_chars - len(prefix)):
            # For generation, we need to feed the last character (or a dummy if this is the first step)
            if len(generated) > 0:
                last_char = generated[-1]
                last_char_idx = char_to_idx_mapping[last_char]
            else:
                # If no prefix, start with some default (this shouldn't happen with your use case)
                last_char_idx = 0
            
            # Create one-hot encoding for single character
            char_tensor = torch.nn.functional.one_hot(
                torch.tensor([last_char_idx]), 
                num_classes=len(char_to_idx_mapping)
            ).float().to(device)
            
            # Get prediction for next character
            output, state = model(char_tensor.unsqueeze(0), state=state)  # Add batch dim
            
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

            if next_char == '£' or next_char == '': # EOS token
            # if next_char == ' ' or next_char == '': # EOS token
                break

            generated += next_char
            
            # print(f"Step {i+1}: Added '{next_char}' -> '{generated}'")
            
    return generated

def robust_generate(generate_function, max_attempts: int, **kwargs):
    n_chars = 100

    attempts = 0
    valid = False
    output = None

    while attempts < max_attempts and valid == False:
        output = generate_function(**kwargs)

        valid = utilities.validate_smiles_string(output)

        if valid:
            return output
        else:
            attempts += 1
        
    print(f"Could not generate valid molecular sample in {max_attempts} attemtps. Aborting.")
    return output


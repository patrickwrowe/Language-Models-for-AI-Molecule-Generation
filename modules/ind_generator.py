import torch
import attrs
from torch import nn, optim
from typing import Optional
import utilities

@attrs.define(eq=False)
class SmilesIndGeneratorRNN(nn.Module):

    vocab_size: int
    num_indications: int
    num_hiddens: int
    num_layers: int

    learning_rate: float = 0.001
    weight_decay: float = 0.01

    output_dropout: float = 0.2
    rnn_dropout: float = 0.2
    state_dropout: float = 0.2

    def __attrs_post_init__(self):
        super().__init__()

        # One hidden layer per state to prep
        self.ind_to_h_0 = nn.Linear(
            in_features=self.num_indications, 
            out_features=self.num_hiddens
        )

        self.ind_to_c_0 = nn.Linear(
            in_features=self.num_indications, 
            out_features=self.num_hiddens
        )

        self.rnn_to_out = nn.Linear(
            self.num_hiddens, self.vocab_size
        )

        self.rnn = nn.LSTM(
            self.vocab_size,
            self.num_hiddens,
            num_layers = self.num_layers,
            batch_first = True,
            dropout=self.rnn_dropout
        )
        
        self.dropout = nn.Dropout(self.output_dropout)
        self.init_state_dropout = nn.Dropout(self.state_dropout)
        
        self.in_emb_h0 = nn.Embedding(
            num_embeddings=self.num_indications,
            embedding_dim=self.num_hiddens
        )

        self.in_emb_h0 = nn.Embedding(
            num_embeddings=self.num_indications,
            embedding_dim=self.num_hiddens
        )

    def init_state(self, ind_tensor: torch.Tensor):
        """
        Initialize LSTM hidden and cell states from indication tensor.
        
        Args:
            ind_tensor: One-hot encoded indication tensor of shape (batch_size, num_indications)
            
        Returns:
            tuple: (h_0, c_0) where each has shape (num_layers, batch_size, num_hiddens)
        """
        
        # Temporarily cast back to ordinal encoding...
        input_tensor = ind_tensor.argmax(dim=-1)

        input_tensor = input_tensor.repeat(self.num_layers, 1).type(torch.long)

        # Transform indication vector to initial hidden and cell states
        # h_0 = self.init_state_dropout(self.ind_to_h_0(input_tensor))  
        # c_0 = self.init_state_dropout(self.ind_to_c_0(input_tensor)) 
        
        h_0 = self.init_state_dropout(self.in_emb_h0(input_tensor))
        c_0 = self.init_state_dropout(self.in_emb_h0(input_tensor))

        return (h_0, c_0)

    def forward(self, seq_tensor: torch.Tensor, ind_tensor: Optional[torch.Tensor] = None, state=None):
        
        if not state and ind_tensor is not None:
            # First, condition the state with indication tensor
            initial_state = self.init_state(ind_tensor)
        else:
            initial_state = state
        
        output, state = self.rnn(seq_tensor, initial_state)
        # output, _ = self.rnn(seq_tensor)
        output = self.dropout(output)
        output = self.rnn_to_out(output)

        return output, state

    def loss(self, y_hat, y):
        loss_function = nn.CrossEntropyLoss(ignore_index=0)  
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

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    

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


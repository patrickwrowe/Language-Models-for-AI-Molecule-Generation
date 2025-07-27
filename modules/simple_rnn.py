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
    output_dropout: float = 0.2
    rnn_dropout: float = 0.2

    def __attrs_post_init__(self):
        super().__init__()
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        self.rnn = nn.RNN(self.vocab_size, self.num_hiddens, num_layers = self.num_layers, batch_first=True, 
                          dropout=self.rnn_dropout, nonlinearity='relu')
        self.dropout = nn.Dropout(self.output_dropout)
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
        # These don't need to be different for this model
        return self.training_step(batch)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

@attrs.define(eq=False)
class simpleLSTM(simpleRNN):
    num_hiddens: int = attrs.field(default=100)
    num_layers: int = attrs.field(default=1)

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
            generated += next_char
            
            # print(f"Step {i+1}: Added '{next_char}' -> '{generated}'")
            
    return generated
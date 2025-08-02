import torch
from torch import nn, optim
from typing import Optional
from typing import Union

class SmilesGenerativeLanguageModel(nn.Module):
    """ 
    (Largely abstract) base class for SMILEs based generative language models
    """

    def __init__(self, vocab_size: int, learning_rate: float = 0.001, weight_decay: float = 0.01):
        super().__init__()
        
        self.vocab_size: int = vocab_size
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay

    def init_state(self, input: Union[torch.Tensor, tuple[torch.Tensor, ...]]) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        raise NotImplementedError()

    def forward(self, input: Union[torch.Tensor, tuple[torch.Tensor, ...]], state: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        raise NotImplementedError()

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_function = nn.CrossEntropyLoss()
        return loss_function(y_hat, y)

    def initialize_parameters(self, module: nn.Module) -> None:

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
            elif type(layer) == nn.Embedding:
                nn.init.xavier_normal(layer.weight)

        module.apply(initialize_xavier)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        preds, _ = self(batch[:-1])
        labels = batch[-1]
        
        # Usually, we want to flatten our lanuage model predictions
        preds = preds.reshape(-1, self.vocab_size)
        labels = labels.reshape(-1)
        
        return self.loss(preds, labels)

    def validation_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.training_step(batch)

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()
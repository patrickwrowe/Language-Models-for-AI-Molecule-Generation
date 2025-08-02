import torch
from torch import nn, optim
from typing import Optional
from typing import Union
import attrs

@attrs.define(frozen=True)
class SmilesGLMConfig:
    learning_rate: float = attrs.field(validator=attrs.validators.gt(0))
    weight_decay: float = attrs.field(validator=attrs.validators.ge(0.0))
    vocab_size: int = attrs.field(validator=attrs.validators.ge(1))

class SmilesGenerativeLanguageModel(nn.Module):
    """ 
    (Largely abstract) base class for SMILEs based generative language models
    """

    def __init__(self, config: SmilesGLMConfig):
        super().__init__()
        self.config = config

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
        return optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def training_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        preds, _ = self(batch[:-1])
        labels = batch[-1]
        
        # Usually, we want to flatten our lanuage model predictions
        preds = preds.reshape(-1, self.config.vocab_size)
        labels = labels.reshape(-1)
        
        return self.loss(preds, labels)

    def validation_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.training_step(batch)

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()
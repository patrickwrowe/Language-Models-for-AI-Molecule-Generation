import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import pytest
import attr


import sys
sys.path.append('..')  # Adjust the path as necessary to import modules
from training.trainer import Trainer

@pytest.fixture
def mock_linear_dataset(size=100, noise=0.001, weights = torch.tensor([1., 2., 3., 4., 5.]), bias: float = 5.):

    @attr.define
    class MockDataset(Dataset):
        
        X: torch.Tensor
        y: torch.Tensor

        batch_size: int = 10

        @classmethod
        def new_dataset(cls, noise=noise, size=size, weights=weights, bias=bias):
            X = torch.randn((size, len(weights)))
            noise = torch.randn((size, len(weights))) * noise

            y = X * weights + bias + noise

            return MockDataset(X, y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

        def get_dataloader(self, train: bool):
            total_len = len(self)
            train_len = int(0.8 * total_len)  # 80% for training
        
            if train:
                # Training uses first 80% of data
                subset = Subset(self, range(train_len))
            else:
                # Validation uses last 20% of data
                subset = Subset(self, range(train_len, total_len))
                
            return DataLoader(
                subset,
                batch_size = self.batch_size,
                shuffle = train
            )
            

        def train_dataloader(self):
            return self.get_dataloader(train=True)

        def val_dataloader(self):
            return self.get_dataloader(train=False)

    return MockDataset.new_dataset()

@pytest.fixture
def mock_model():
    
    @attr.define(eq=False)
    class MockModel(nn.Module):
        
        input_size: int
        output_size: int

        def __attrs_post_init__(self):
            super().__init__()
            self.linear = nn.Linear(self.input_size, self.output_size)

        def forward(self, x):
            return self.linear(x)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.01)
        
        def loss(self, y_hat, y):
            return nn.MSELoss()(y_hat, y)

        def training_step(self, batch):
            X, y = batch
            preds = self(X)
            loss = self.loss(preds, y)
            return loss

        def validation_step(self, batch):
            return self.training_step(batch)

    return MockModel(input_size=5, output_size=1)

def test_train(mock_linear_dataset, mock_model):
    trainer = Trainer(max_epochs=1)
    trainer.fit(mock_model, mock_linear_dataset)

    assert trainer.model is not None
    assert isinstance(trainer.optim, torch.optim.Optimizer)
    assert len(trainer.train_dataloader) > 0
    assert len(trainer.val_dataloader) > 0

    # Check if the model can make predictions
    batch = next(iter(trainer.train_dataloader))
    
    X, y = [b.to(trainer.device) for b in batch]

    preds = trainer.model(X)
    loss = trainer.model.loss(preds, y)
    assert loss.item() >= 0  # Loss should be non-negative
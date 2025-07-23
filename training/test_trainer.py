import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytest
import attr

import sys
sys.path.append('..')  # Adjust the path as necessary to import modules
from training.trainer import Trainer

@pytest.fixture
def mock_linear_dataset(size=100, noise=0.001, weights = torch.tensor([1., 2., 3., 4., 5.]), biases = torch.tensor([-1., 0., 2., 5., 10.])):

    @attr.define
    class MockDataset(Dataset):
        
        X: torch.Tensor
        y: torch.Tensor

        @classmethod
        def new_dataset(cls, noise=noise, size=size, weights=weights, biases=biases):
            X = torch.randn((size, len(weights)))
            noise = torch.randn((size, len(weights))) * noise

            y = torch.matmul(X, weights).reshape(-1, 1) + biases + noise

            return MockDataset(X, y)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return MockDataset.new_dataset()

@pytest.fixture
def mock_linear_dataloader(mock_linear_dataset, batch_size=10):
    return DataLoader(mock_linear_dataset, batch_size=batch_size, shuffle=True)

@pytest.fixture
def mock_model():
    
    @attr.define
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

    return MockModel(input_size=5, output_size=1)

def test_train(mock_linear_dataloader, mock_model):
    trainer = Trainer(max_epochs=1, init_random=42)
    trainer.fit(mock_model, mock_linear_dataloader.dataset)

    assert trainer.model is not None
    assert isinstance(trainer.optim, torch.optim.Optimizer)
    assert len(trainer.train_dataloader) > 0
    assert len(trainer.val_dataloader) > 0

    # Check if the model can make predictions
    for batch in trainer.train_dataloader:
        X, y = batch
        preds = trainer.model(X)
        loss = trainer.model.loss(preds, y)
        assert loss.item() >= 0  # Loss should be non-negative
        break  # Only check the first batch
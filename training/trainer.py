import attrs
import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim.optimizer import Optimizer
import numpy as np
from typing import Optional

@attrs.define(slots=False)
class Trainer:

    # training options
    max_epochs: int
    init_random: Optional[int] = None
    clip_grads_norm: Optional[float] = None

    # model params
    model: Module = attrs.field(init=False)
    optim: Optimizer = attrs.field(init=False)

    # Data parameters
    train_dataloader: DataLoader = attrs.field(init=False)
    val_dataloader: DataLoader = attrs.field(init=False)
    num_train_batches: int = attrs.field(init=False)
    num_val_batches: int = attrs.field(init=False)

    # Counter
    epoch: int = attrs.field(init=False)
    train_batch_idx: int = attrs.field(init=False)
    val_batch_idx: int = attrs.field(init=False)

    # Metadata
    metadata: dict = {}

    def prepare_state(self):
        if self.init_random:
            torch.manual_seed(self.init_random)
            torch.use_deterministic_algorithms(True)

    def prepare_data(self, data):
        """Data must inherit from the DataSet class and have special classes for returning loaders"""
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = len(self.val_dataloader)

    def prepare_model(self, model):
        """Prepare the model, attempt to put it on the GPU if we can"""
        model.trainer = self
        device = try_gpu()
        model.to(device)
        self.model = model

    def fit(self, model, data):
        self.prepare_state()
        self.prepare_data(data)
        self.prepare_model(model)

        # Model should specify own optimizer methods
        self.optim = model.configure_optimizers()
        
        # Initialise counters
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        # initialise metadata used for printing/plotting
        self.metadata = {
            "max_epochs": self.max_epochs,
            "num_train_batches": self.num_train_batches,
            "num_val_batches": self.num_val_batches,
            "training_epochs": [],
        }

        for self.epoch in range(self.max_epochs):
            # one epoch fits over the entire training dataset
            # then evaluates on validation dataset for us
            self.fit_epoch()

    def maybe_batch_to_gpu(self, batch):
        batch = [b.to(try_gpu()) for b in batch]
        return batch

    def fit_epoch(self):
        """Fit one epoch of the model, this is where the real learning happens"""
        self.model.train()  # ensure model is set to training mode
        train_loss = 0.0  # Initialise loss to 0

        # main training loop over batches in train_dataloader
        for self.train_batch_idx, batch in enumerate(self.train_dataloader):

            end = "\r" if self.train_batch_idx < self.num_train_batches else "\n"
            print(f"Training batch {self.train_batch_idx + 1}/{self.num_train_batches}..."
                  f" (Epoch {self.epoch + 1}/{self.max_epochs})", end=end)

            # must manually move batch from main memory to GPU if we have access to a GPU
            batch = self.maybe_batch_to_gpu(batch)

            # We have to manually reset the gradients to zero each step
            # or else they'll just be added to the previous ones
            self.optim.zero_grad()

            # the model implements a method which computes the loss on a particular batch
            loss = self.model.training_step(batch)
            
            # loss.backward computes the gradients fo rhthe step using analytical gradients
            # and stores these in the pytorch tensor objects for us to use
            loss.backward()

            if self.clip_grads_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grads_norm)

            # use the optimizer to move the parameters in the direction of the gradient
            self.optim.step()

            train_loss += loss.item()
    
        # average the loss over all the batches
        avg_train_loss = train_loss / self.num_train_batches

        # Just evaluate the model on the validation loop if we have a validation dataset
        if self.num_val_batches > 0:
            self.model.eval()  # Swap the model from training mode to evaluation mode
            val_loss = 0.0
            
            with torch.no_grad():  # Don't want gradients in evaluation mode
                for self.val_batch_id, batch in enumerate(self.val_dataloader):
                    batch = self.maybe_batch_to_gpu(batch)
                    val_loss += self.model.validation_step(batch)

            avg_val_loss = val_loss / self.num_val_batches
        else:
            avg_val_loss = None
            val_loss = None

        # Update metadata with training epoch information
        self.metadata["training_epochs"].append(
            {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": avg_val_loss,
            }
        )

        self.print_training_status()


    def print_training_status(self):
        train_loss = self.metadata["training_epochs"][-1]["avg_train_loss"]
        val_loss = self.metadata["training_epochs"][-1]["avg_val_loss"] 
        val_loss = val_loss if val_loss is not None else np.NaN

        # end = "\n" if self.epoch == self.max_epochs -1 else "\r"  # Carriage return unless last epoch
        end = "\n"
        print(f"Epoch {self.epoch + 1}/{self.max_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}", end=end)


def try_gpu():
    """Return gpu if exists, otherwise return cpu"""
    if torch.cuda.device_count() >= 1:
        return torch.device("cuda")
    return torch.device("cpu")

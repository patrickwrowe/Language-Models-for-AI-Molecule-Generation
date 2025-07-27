import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_training_validation_loss(
    training_losses: np.ndarray,
    validation_losses: np.ndarray,
    title="Training and Validation Loss",
):
    """
    Plot the training and validation loss.
    """

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(training_losses, marker="o", label="Training Loss")

    if validation_losses.any():
        ax.plot(validation_losses, marker="x", label="Validation Loss")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)

    return fig, ax
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error


def plot_correlations(correlations: pd.DataFrame, colormap: str = "seismic"):
    """
    Plots the result of calling pandas.DataFrame.corr() as a heatmap.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(correlations, cmap=colormap)

    # We want to show all ticks...
    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.columns)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(correlations.columns)
    ax.set_yticklabels(correlations.columns)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(correlations.columns)):
        for j in range(len(correlations.columns)):
            text = ax.text(
                j,
                i,
                f"{correlations.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                size=10,
                weight="bold",
            )

    ax.set_title("Diabetes Correlations")
    fig.tight_layout()

    return fig, ax


def plot_two_feature_correlation(
    feature_1: pd.Series, feature_2: pd.Series, title: str = "Feature Correlation"
):
    """
    Plot the correlation between two features.
    """

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(feature_1, feature_2)

    ax.set_title(title)
    ax.set_xlabel(feature_1.name)
    ax.set_ylabel(feature_2.name)

    return fig, ax


def plot_umap(
    umap_coords: np.ndarray,
    color_by: np.ndarray,
    colormap: str = "viridis",
    title: str = "UMAP",
):
    """
    Plot the UMAP embedding of the dataset.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1], c=color_by, cmap=colormap
    )

    ax.set_title(title)

    # No markers on axes
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    return fig, ax


def correlate_y_vs_yhat(
    y: np.ndarray,
    y_hat: np.ndarray,
    title="Correlation",
    show_metrics=True,
):

    r2 = r2_score(y, y_hat)
    mse = mean_squared_error(y, y_hat)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))

    ax.scatter(y, y_hat, alpha=0.6, label="Predicted")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    if show_metrics:
        ax.set_title(f"{title}\nR2: {r2:.2f}, MSE: {mse:.2f}")

    ax.set_box_aspect(1)

    return fig, ax


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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from analysis import similarity

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

def plot_umap(
    embedding_smiles: list[str],
    query_smiles: list[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    colour_categories: list[str] = None,
    title="UMAP Projection of SMILES",
    tanimoto: bool = True,
    **kwargs
):
    """
    Plot UMAP projection of SMILES strings.

    """

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta", "yellow", "black"]
    
    # Compute UMAP embedding
    embedding, umap_model = similarity.fit_umap(embedding_smiles, n_neighbors, min_dist, tanimoto, **kwargs)

    # Create a DataFrame for the embedding
    df_embedding = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    df_embedding["SMILES"] = embedding_smiles
    df_embedding["Category"] = colour_categories if colour_categories else None

    if query_smiles:
        df_query = pd.DataFrame(umap_model.transform(query_smiles), columns=["UMAP1", "UMAP2"])
        df_query["SMILES"] = query_smiles


    fig, ax = plt.subplots(figsize=(8, 8))

    if colour_categories:
        groups = df_embedding.groupby("Category")
        for i, (name, group) in enumerate(groups):
            ax.scatter(group["UMAP1"], group["UMAP2"], label=name, alpha=0.5, c=colors[i % len(colors)])
    else:
        ax.scatter(df_embedding["UMAP1"], df_embedding["UMAP2"], alpha=0.5, label="SMILES")

    if query_smiles:
        ax.scatter(df_query["UMAP1"], df_query["UMAP2"], color="red", label="Query SMILES")


    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    return fig, ax
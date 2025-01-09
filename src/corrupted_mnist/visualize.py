import matplotlib.pyplot as plt
import torch
import typer
from model import (
    MyAwesomeModel,  # type: ignore  # Ignoring type check due to potential issues with dynamic imports or missing type hints
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """
    Visualizes the model embeddings for a test dataset using PCA (if needed) and t-SNE, then
    plots the resulting 2D projection by class label and saves the figure as an SVG.
    Args:
        model_checkpoint (str):
            Path to the saved model checkpoint (typically a .pt file) containing the trained model weights.
        figure_name (str, optional):
            Name of the output figure file to save (default is "embeddings.png", saved as SVG).
    Returns:
        None
    Note:
        - This function loads a model, replaces its final fully connected layer with an identity,
          extracts embeddings from test images, optionally reduces the dimensionality with PCA if
          the embedding dimension is larger than 500, then applies t-SNE to obtain a 2D representation.
        - A scatter plot is generated for each class label on the resulting 2D embedding space, and
          the figure is saved in SVG format in the "reports/figures" directory.
    """
    """Visualize model predictions."""
    model: torch.nn.Module = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    model.fc = torch.nn.Identity()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, target = batch
            predictions = model(images)
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}.svg", format="svg", dpi=300)


if __name__ == "__main__":
    typer.run(visualize)

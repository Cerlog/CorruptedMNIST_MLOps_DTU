import typer
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import os
import matplotlib.pyplot as plt
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

app = typer.Typer()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def train_with_hydra(lr: float, batch_size: int, epochs: int) -> None:
    """
    Typer entry point to train the model with CLI-provided arguments.
    Overrides Hydra's default configuration values for training.
    """
    typer.echo(f"Starting training with Typer arguments: lr={lr}, batch_size={batch_size}, epochs={epochs}")

    # Overriding Hydra configuration dynamically
    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def train(cfg: DictConfig) -> None:
        cfg.training.lr = lr
        cfg.training.batch_size = batch_size
        cfg.training.epochs = epochs
        typer.echo(f"Using Hydra configuration: {cfg}")

        # Instantiate the model
        model = instantiate(cfg.model).to(DEVICE)

        # Load dataset
        from data import corrupt_mnist
        train_set, _ = corrupt_mnist()
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.training.batch_size)

        # Define loss and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

        # Training loop
        statistics = {"train_loss": [], "train_accuracy": []}
        for epoch in range(cfg.training.epochs):
            model.train()
            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

                # Record statistics
                statistics["train_loss"].append(loss.item())
                accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
                statistics["train_accuracy"].append(accuracy)

                if i % 100 == 0:
                    typer.echo(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        typer.echo("Training complete")

        # Save model and statistics
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports/figures", exist_ok=True)
        torch.save(model.state_dict(), "models/model.pth")
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(statistics["train_loss"])
        axs[0].set_title("Train loss")
        axs[1].plot(statistics["train_accuracy"])
        axs[1].set_title("Train accuracy")
        fig.savefig("reports/figures/training_statistics.png")

    # Run Hydra-enabled training
    train()


@app.command()
def train(
    lr: float = typer.Option(0.001, help="Learning rate"),
    batch_size: int = typer.Option(32, help="Batch size"),
    epochs: int = typer.Option(10, help="Number of epochs"),
) -> None:
    """
    Typer command to train the model with CLI-provided arguments.
    """
    train_with_hydra(lr=lr, batch_size=batch_size, epochs=epochs)


if __name__ == "__main__":
    app()

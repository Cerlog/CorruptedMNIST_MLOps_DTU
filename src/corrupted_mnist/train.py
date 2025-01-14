import os

import hydra
import matplotlib.pyplot as plt
import torch
import typer
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

import wandb
from corrupted_mnist.data import corrupt_mnist

load_dotenv()  # Load environment variables from .env file


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train_model(cfg: DictConfig) -> None:
    """Core training function that uses Hydra configuration."""
    print("Training day and night")
    print(f"Training config:\n{cfg.training}")

    # Initialize WandB with environment variables
    wandb.init(
        project="new_project_name",
        entity=os.getenv("WANDB_ENTITY", None),
        config={
            "lr": cfg.optimizer.lr,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "model": cfg.model._target_,
        },
    )

    # Model instantiation using Hydra
    model = hydra.utils.instantiate(cfg.model).to(DEVICE)

    # Data loading
    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.training.batch_size)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer instantiation using Hydra
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

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

            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy, "epoch": epoch})

            if i % cfg.training.log_interval == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")

    # Save model using Hydra's working directory
    model_save_path = f"models/model.pth"
    torch.save(model.state_dict(), model_save_path)

    # Create and save training plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

    # Log plot to WandB
    # wandb.log({"training_curves": wandb.Image(os.path.join("models", "training_statistics.png"))})

    # Finish WandB session
    wandb.finish()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def hydra_train(cfg: DictConfig) -> None:
    """Entry point for Hydra-based training."""
    train_model(cfg)


def typer_train(
    lr: float = typer.Option(1e-3, help="Learning rate"),
    batch_size: int = typer.Option(32, help="Batch size"),
    epochs: int = typer.Option(10, help="Number of epochs"),
) -> None:
    """Entry point for Typer-based training."""
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Initialize Hydra manually
    hydra.initialize(version_base=None, config_path="../../config")
    cfg = hydra.compose(config_name="config")

    # Override config with CLI arguments
    cfg.optimizer.lr = lr
    cfg.training.batch_size = batch_size
    cfg.training.epochs = epochs

    # Run the training
    train_model(cfg)


def main():
    """Main function to dynamically select CLI framework."""
    import sys

    # Check if arguments are in Hydra-style (key=value pairs)
    if any("=" in arg for arg in sys.argv[1:]):
        hydra_train()  # Use Hydra CLI
    else:
        typer.run(typer_train)  # Use Typer CLI


if __name__ == "__main__":
    main()

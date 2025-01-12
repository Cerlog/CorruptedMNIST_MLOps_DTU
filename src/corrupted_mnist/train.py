import os
import sys
from pathlib import Path
import typer
import torch
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig

from .data import corrupt_mnist
from .model import MyAwesomeModel

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """
    Example training function using Hydra + Typer.
    """
    # 1) Determine the absolute path to your configs folder
    script_dir = Path(__file__).resolve().parent   # e.g., .../src/corrupted_mnist
    root_dir = script_dir.parent.parent            # go up 2 levels to project root
    configs_path = root_dir / "configs"            # -> .../<project>/configs (absolute)

    # 2) Initialize Hydra for an absolute config dir:
    with initialize_config_dir(config_dir=str(configs_path), version_base=None):
        # 3) Compose your config, optionally overriding fields
        cfg = compose(
            config_name="config",
            overrides=[
                f"training.lr={lr}",
                f"training.batch_size={batch_size}",
                f"training.epochs={epochs}",
            ],
        )

    print("Training with Hydra + Typer")
    print(f"lr={lr}, batch_size={batch_size}, epochs={epochs}")

    model = instantiate(cfg.model).to(DEVICE)
    train_set, _ = corrupt_mnist()
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    stats = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        for i, (img, target) in enumerate(dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            preds = model(img)
            loss = loss_fn(preds, target)
            loss.backward()
            optimizer.step()
            stats["train_loss"].append(loss.item())
            acc = (preds.argmax(dim=1) == target).float().mean().item()
            stats["train_accuracy"].append(acc)
            if i % 100 == 0:
                print(f"Epoch={epoch} batch={i} loss={loss.item():.4f}")

    print("Training complete!")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")

    os.makedirs("reports/figures", exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(stats["train_loss"])
    axs[0].set_title("Train Loss")
    axs[1].plot(stats["train_accuracy"])
    axs[1].set_title("Train Accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    print("Saved model and figures!")


def main():
    typer.run(train)

if __name__ == "__main__":
    main()

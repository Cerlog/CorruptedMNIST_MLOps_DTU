import torch
import typer
from model import MyAwesomeModel

from data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """
    Evaluate a trained model from a checkpoint file, measure performance on a test set,
    and print the resulting accuracy.
    Args:
        model_checkpoint (str): Path to the model checkpoint file to be loaded.
    Returns:
        None
    # Comments:
    # 1. The function loads the specified checkpoint into a PyTorch model instance.
    # 2. It then processes the test dataset to compute performance metrics.
    # 3. Finally, it prints the test accuracy without returning any value.
    """
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)

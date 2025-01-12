import torch
import typer
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate
from .model import MyAwesomeModel
from .data import corrupt_mnist

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)
    
    # Get config path
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent
    configs_path = root_dir / "configs"
    
    # Initialize Hydra config
    with initialize_config_dir(config_dir=str(configs_path), version_base=None):
        cfg = compose(config_name="config")
    
    # Create model using the same configuration as training
    model = instantiate(cfg.model).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():  # Add this for evaluation
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)
    
    print(f"Test accuracy: {correct / total}")

def main():
    typer.run(evaluate)

if __name__ == "__main__":
    main()
import torch
from torch import nn
from omegaconf import DictConfig # type: ignore
from hydra.utils import instantiate # type: ignore
import hydra # type: ignore
import sys
import os 

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

class MyAwesomeModel(nn.Module):
    """A CNN model for classification on MNIST-like data."""

    def __init__(
            self,
            conv1_out_channels: int ,
            conv2_out_channels: int,
            conv3_out_channels: int,
            dropout_rate: float,
            fc1_out_features: int,
        ) -> None:
        """
        Initializes the model with three convolutional layers, a dropout layer,
        and a fully connected output layer.
        """
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, conv1_out_channels, 3, 1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, 3, 1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(conv2_out_channels, conv3_out_channels, 3, 1)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Fully connected output layer
        self.fc1 = nn.Linear(conv3_out_channels, fc1_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:    
            torch.Tensor: The output tensor (logits or predictions).
        """
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # instantiate the model using Hydra
    model = instantiate(cfg.model)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Load training settings
    data_cfg = cfg.training.data
    dummy_input = torch.randn(1, data_cfg.input_channels, data_cfg.input_height, data_cfg.input_width)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
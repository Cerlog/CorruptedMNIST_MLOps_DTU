import torch
from torch import nn
import hydra 
from omegaconf import DictConfig

class MyAwesomeModel(nn.Module):
    def __init__(
        self,
        conv1_out_channels: int,
        conv2_out_channels: int,
        conv3_out_channels: int,
        dropout_rate: float,
        fc1_out_features: int,
        input_channels: int = 1
    ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, conv1_out_channels, 3, 1)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, 3, 1)
        self.conv3 = nn.Conv2d(conv2_out_channels, conv3_out_channels, 3, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(conv3_out_channels, fc1_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

if __name__ == "__main__":

    @hydra.main(config_path="../config", config_name="config")
    def main(cfg: DictConfig) -> None:
        # Hyra will automatically instantiate the model using the config 
        model = hydra.utils.instantiate(cfg.model)
        print(f"Model architecture: {model}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        dummy_input = torch.randn(1, cfg.model.input_channels, 28, 28)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")

    main()
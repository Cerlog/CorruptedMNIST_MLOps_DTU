import pytest
import torch
from corrupted_mnist.model import MyAwesomeModel


def test_model():
    model = MyAwesomeModel(
        conv1_out_channels=16, conv2_out_channels=32, conv3_out_channels=64, dropout_rate=0.5, fc1_out_features=10
    )
    x = torch.rand(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


# Add a test for the model with different batch sizes using pytest.mark.parametrize
@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_2(batch_size: int) -> None:
    model = MyAwesomeModel(
        conv1_out_channels=16, conv2_out_channels=32, conv3_out_channels=64, dropout_rate=0.5, fc1_out_features=10
    )
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)

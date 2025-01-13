from omegaconf import DictConfig
from corrupted_mnist.train import train_model
from corrupted_mnist.data import corrupt_mnist
import os 
import hydra 
from corrupted_mnist.model import MyAwesomeModel
def test_train_model():
    # test whether the model trains

    # load data
    train_set, _ = corrupt_mnist()

    model = MyAwesomeModel(conv1_out_channels=16, conv2_out_channels=32, conv3_out_channels=64, dropout_rate=0.5, fc1_out_features=11)

    hydra_train(DictConfig({"training": {"epochs": 1, "batch_size": 32, "log_interval": 10, "save_dir": "models"}}))
    # check if the model was saved
    assert os.path.exists("models/model.pth")
    # check if the training statistics were saved
    assert os.path.exists("training_statistics.png")

    # clean up
    os.remove("models/model.pth")

if __name__ == "__main__":
    test_train_model()
    print("Everything passed")



@hydra.main(version_base=None, config_path="../../config", config_name="config")
def hydra_train(cfg: DictConfig) -> None:
    """Entry point for Hydra-based training."""
    train_model(cfg)


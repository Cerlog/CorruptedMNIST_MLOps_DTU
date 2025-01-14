import os

from dotenv import load_dotenv

import wandb

load_dotenv()  # Load environment variables from .env file
print(f"WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")
print(f"WANDB_ENTITY: {os.getenv('WANDB_ENTITY')}")
print(f"WANDB_API_KEY: {os.getenv('WANDB_API_KEY')}")

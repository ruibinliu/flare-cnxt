from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Config:
    # Dataset configurations
    RUNTIME_DATA_ROOT: str = os.getenv("RUNTIME_DATA_ROOT", "/tmp/nvflare/data/lung_function")
    MANIFEST_PATH: str = str(os.getenv("MANIFEST_PATH", "/processed/convnext/image_manifest.csv"))

    # Federated Learning configurations
    NUM_ROUNDS: int = int(os.getenv("NUM_ROUNDS", "10"))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "10"))
    MIN_CLIENTS: int = int(os.getenv("MIN_CLIENTS", "3"))
    NUM_CLIENTS: int = int(os.getenv("NUM_CLIENTS", "3"))

    # Training configurations
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    SEED: int = int(os.getenv("SEED", "42"))
    WEIGHT_DECAY: float = float(os.getenv("LEARNING_RATE", "0.0001"))
    EARLY_STOP_PATIENCE: int = int(os.getenv("EARLY_STOP_PATIENCE", "12"))
    GRAD_CLIP: float = float(os.getenv("GRAD_CLIP", "5.0"))

    # PLE Model configurations
    N_BINS = 20
    EMB_DIM = 8


config = Config()

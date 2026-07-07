from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.pt"
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "shakespeare.txt"

# Character vocabulary for the trained checkpoint (vocab_size=65).
VOCAB_CHARS = (
    "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)

from functools import lru_cache
from pathlib import Path

import torch

from dataset import build_tokenizer
from model import GPT
from paths import MODEL_PATH


@lru_cache(maxsize=1)
def load_model(checkpoint_path: str | None = None):
    resolved_path = Path(checkpoint_path) if checkpoint_path else MODEL_PATH
    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Trained model file is missing. Place model.pt in the project root."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        resolved_path,
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    tokenizer = build_tokenizer()

    model = GPT(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
        dropout=0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, tokenizer, device, config


@torch.no_grad()
def generate_text(
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.8,
    checkpoint_path: str | None = None,
):
    model, tokenizer, device, config = load_model(checkpoint_path)
    max_tokens = max(20, min(int(max_tokens), 800))
    temperature = max(0.2, min(float(temperature), 1.5))

    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "text": tokenizer.decode(output_ids[0]),
        "device": device,
        "config": config,
        "tokens": max_tokens,
        "temperature": temperature,
    }

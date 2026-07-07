import urllib.request

import torch

from paths import DATA_DIR, DATA_PATH, VOCAB_CHARS

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/"
    "refs/heads/main/input.txt"
)


def download_shakespeare():
    if DATA_PATH.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)



class CharacterTokenizer:
    def __init__(self, text: str):
        self.characters = sorted(list(set(text)))
        self.vocab_size = len(self.characters)
        self.char_to_id = {char: index for index, char in enumerate(self.characters)}
        self.id_to_char = {index: char for index, char in enumerate(self.characters)}

    def encode(self, text: str) -> list[int]:
        unknown = sorted({char for char in text if char not in self.char_to_id})
        if unknown:
            pretty = ", ".join(repr(char) for char in unknown[:8])
            raise ValueError(f"Model vocabulary does not contain: {pretty}")
        return [self.char_to_id[char] for char in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.id_to_char[int(token_id)] for token_id in ids)


def build_tokenizer() -> CharacterTokenizer:
    return CharacterTokenizer(VOCAB_CHARS)


def load_data(train_split: float = 0.9):
    download_shakespeare()
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = CharacterTokenizer(text)
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)

    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]

    return train_data, test_data, tokenizer


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    max_start = len(data) - block_size - 1
    positions = torch.randint(max_start, (batch_size,))

    x_list = []
    y_list = []
    for pos in positions:
        x_list.append(data[pos : pos + block_size])
        y_list.append(data[pos + 1 : pos + block_size + 1])

    return torch.stack(x_list), torch.stack(y_list)

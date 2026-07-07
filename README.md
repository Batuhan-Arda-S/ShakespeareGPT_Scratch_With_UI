# Shakespeare GPT with Flask UI

A character-level GPT model trained on Shakespeare text, wrapped in a clean Flask web interface for portfolio and CV use.

## Features

- Custom PyTorch transformer architecture
- `model.pt` checkpoint loading
- Flask web app and JSON generation API
- White, responsive HTML/CSS/JS interface
- Automatic Shakespeare dataset download for tokenizer reconstruction

## Project Structure

```text
.
├── app.py
├── dataset.py
├── inference.py
├── model.py
├── model.pt
├── requirements.txt
├── static/
│   ├── css/styles.css
│   └── js/app.js
└── templates/
    └── index.html
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"ROMEO:\",\"max_tokens\":300,\"temperature\":0.8}"
```

## Notes

The app expects `model.pt` in the project root. On first run, it downloads the Shakespeare text used to rebuild the character tokenizer.

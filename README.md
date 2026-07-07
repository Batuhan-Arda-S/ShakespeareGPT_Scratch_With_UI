# GPT Language Model From Scratch
![Model](model3.png)

A decoder-only GPT language model implemented entirely from scratch using PyTorch and trained on the Shakespeare dataset at the character level.

The project recreates the core building blocks of modern autoregressive language models, including tokenization, positional embeddings, multi-head self-attention, transformer blocks, and autoregressive text generation.

## Project Overview

This project was built to gain a deep understanding of how GPT-style language models work internally by implementing the entire architecture from scratch instead of relying on high-level frameworks.

The model is trained using next-token prediction on Shakespeare's complete works and generates Shakespeare-style text from custom prompts.
![Model](model4.png)

## Model Architecture

Custom decoder-only Transformer architecture including:

- Character-level tokenizer
- Token embedding layer
- Learned positional embeddings
- Multi-Head Self-Attention
- Residual connections
- Layer Normalization
- Feed Forward Networks (MLP)
- Stacked Transformer blocks
- Final language modeling head

## Training Pipeline

Implemented modern language model training techniques including:

- Character-level next-token prediction
- Cross Entropy Loss
- AdamW optimizer
- Learning rate warmup
- Gradient clipping
- Train/Test split evaluation
- Periodic validation during training
- CUDA GPU acceleration
- Model checkpointing
- Temperature-controlled autoregressive text generation

## Project Structure

```text
GPT-Language-Model/
│
├── dataset.py          # Dataset loading and tokenizer
├── model.py            # GPT architecture
├── train.py            # Training pipeline
├── inference.py        # Text generation
├── checkpoints/        # Saved model weights
├── data/               # Shakespeare dataset
└── README.md
```

## Results

- Dataset: Shakespeare
- Language Modeling: Character-level
- Vocabulary Size: 65 characters
- Context Length: 256 tokens
- Embedding Dimension: 384
- Transformer Layers: 6
- Attention Heads: 6
- Batch Size: 64
- Training Iterations: 5000
- Final Test Loss: **2.086**

## Technologies

- Python
- PyTorch
- CUDA

## Features

- GPT-style decoder-only Transformer implemented from scratch
- Character-level tokenizer
- Multi-Head Self-Attention
- Learned positional embeddings
- Autoregressive text generation
- Custom training pipeline
- Model checkpointing
- GPU-accelerated training
- Interactive inference script

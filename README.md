# LLMFromScratch
Implementation of modern LLM architecture from scratch

A simple yet fully functional autoregressive Transformer language model built entirely in PyTorch, trained on dialog-style and Shakespeare-style text.
This project demonstrates the full pipeline for building a small LLM: dataset preparation, tokenizer training, Transformer implementation, training loop, and text generation with prompt conditioning.

---

## Features

* Byte-level BPE tokenizer trained from scratch
* Dataset builder for both plain text and custom conversation datasets
* Custom Transformer architecture with

  * Learned token embeddings
  * Learned positional embeddings
  * Multi-head causal self-attention
  * Feed-forward MLP layers
  * Residual connections
  * LayerNorm
* Efficient batching windowing system
* Auto-regressive text generation using multinomial sampling
* Prompt wrapper for simple Q&A formatting using `<user>` and `<bot>` tags

---

## Project Structure

```
project/
│
├── data.csv                       Raw text dataset (dialog + lines)
├── Conversation.csv               Q&A dataset for post-training
│
├── final_training_data.txt        Flattened training corpus
├── final_post_training_data.txt   Flattened Q&A fine-tuning corpus
│
├── output_file.txt                Generated text samples
│
└── main.py                        Full LLM implementation (this script)
```

---

## Pipeline Overview

### 1. Data Preparation

Both datasets are cleaned and flattened.
Shakespeare/dialogue lines are concatenated into `final_training_data.txt`.
Conversation data is wrapped into:

```
<user>question</user>
<bot>answer</bot>
```

and written into `final_post_training_data.txt`.

### 2. Tokenizer Training

A fresh **Byte-Level BPE tokenizer** is trained on the dataset:

```python
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files='final_training_data.txt', vocab_size=32000)
```

This tokenizer is then used for encoding both the base corpus and Q&A corpus.

### 3. Model Configuration

```
context_length = 256
embedding_dim = 512
head_count = 8
feed_forward_dim = embedding_dim * 4
pos_emb_length = 2048
```

The model is a 6-layer Transformer with causal attention.

---

## Transformer Architecture

### Attention Head

Computes Q, K, V projections and performs causal masked attention.

### Multi-Head Attention

Concatenates several heads and applies a final projection.

### Feed-Forward Layer

Two linear layers with ReLU activation.

### Transformer Block

LayerNorm → Attention → Residual
LayerNorm → FeedForward → Residual

### Full Model

Token + positional embeddings → 6 blocks → linear decoder head → logits

Loss is computed with `F.cross_entropy` by flattening `(B * T)` tokens.

---

## Training

Two-stage training is supported:

### Stage 1: Base Modeling

```python
Train(500, src_train, src_test)
```

### Stage 2: Instruction Tuning

Fine-tunes on `<user> ... <bot> ...` format:

```python
Train(1000, src_post_train, src_post_test)
```

Both train and test losses are logged every 50 epochs.

---

## Text Generation

Generation is autoregressive.
Steps:

1. Encode a prompt
2. Feed through Transformer
3. Apply softmax to last token logits
4. Sample a token with `torch.multinomial`
5. Append token and continue

### Example

```python
Prompt("how are you doing?")
```

Outputs a short conversation-style response and writes it to `output_file.txt`.

---

## Usage

### Prompting

```python
Prompt("tell me something about stars")
```

### Direct generation

```python
tokens = torch.tensor([[1, 42, 17]]).to(device)
generate(tokens, max_size=100)
```

---

## Requirements

```
torch
tokenizers
pandas
re
math
```

CUDA is automatically used if available.

---

## Next Steps

* Add dropout
* Add positional encoding alternatives (sinusoidal, rotary)
* Switch to FlashAttention or scaled dot-product APIs
* Export model and tokenizer for inference scripts
* Add top-k and top-p sampling

---

## License

MIT License.

---

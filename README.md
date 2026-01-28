# Small Language Model (SLM) from Scratch

This repository implements a Small Language Model (SLM) with approximately 10-15 million parameters, trained on the TinyStories dataset to generate coherent English text similar to stories for 3-4 year olds.

The model follows a GPT-like architecture using GPT-2 tokenization, multi-head self-attention, and next-token prediction training.

## Features
- **Dataset**: TinyStories from Hugging Face (2M training stories, 20K validation).
- **Tokenizer**: GPT-2 compatible (tiktoken library).
- **Architecture**: 6 transformer layers, 6 attention heads, 384 embedding dim, 128 context length, vocab size 50,257.
- **Training**: AdamW optimizer with warmup + cosine annealing LR schedule, gradient accumulation, mixed precision (bfloat16/float16).
- **Generation**: Sampling with temperature and top-k.

## Prerequisites
- Python 3.8+
- PyTorch (with CUDA for GPU training)
- Required libraries: `datasets`, `tiktoken`, `numpy`, `tqdm`, `matplotlib`

Install dependencies:
```
pip install torch datasets tiktoken numpy tqdm matplotlib
```

## Dataset Preparation
1. Load TinyStories dataset:
   ```
   from datasets import load_dataset
   ds = load_dataset("roneneldan/TinyStories")
   ```
2. Tokenize and save to binary files (`train.bin`, `validation.bin`):
   - Uses GPT-2 tokenizer.
   - Processes text to token IDs, stores sequentially on disk via numpy memmap for efficiency.

Full preprocessing script in the article under "Part 2: Data pre-processing".

## Model Architecture
Defined in `GPT` class (PyTorch nn.Module):
- Token and position embeddings (shared weights with LM head).
- Stack of transformer blocks (attention + FFN).
- Layer norm and linear LM head.
- Custom weight initialization.

Key config:
```
config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1
)
```

## Training
1. Create model: `model = GPT(config).to(device)`
2. Optimizer: AdamW with weight decay 0.1, betas=(0.9, 0.95).
3. LR schedule: Linear warmup (1000 steps) + cosine annealing.
4. Batch size 32, gradient accumulation 32, max iters 20,000.
5. Loss: Cross-entropy on next-token prediction.
6. Evaluates every 500 iters, saves best model by val loss.

Training loop uses `get_batch(split)` for input/output pairs (shifted by 1).

## Inference
Load trained model and generate:
```
model.load_state_dict(torch.load("best_model_params.pt"))
idx = torch.tensor(enc.encode("Once upon a time")).unsqueeze(0).to(device)
generated = model.generate(idx, max_new_tokens=100, temperature=0.8)[page:0]
print(enc.decode(generated[0].tolist()))
```



## Results
- Trains to low validation loss (~1.5-2.0).
- Generates coherent short stories post-training.


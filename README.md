# MLX-LM-LENS
Find the hidden meaning of LLMs

MLX-LM-LENS provides a simple wrapper to inspect hidden states of MLX-based language models.

This package is mainly intended as a research tool, though it can also be used to
create real-world models such as the "abliterated" models. Beyond hidden states
it lets you inspect attention scores and embedding layer outputs. MLX-LM-LENS is
built on top of the MLX-LM framework, so every model supported in MLX-LM works
here as well.

## Installation

```bash
pip install mlx-lm-lens
```

## Quick Start

```python
import mlx.core as mx
from mlx_lm import load
from mlx_lm_lens.lm_lens import MLX_LM_Lens_Wrapper

model, tokenizer = load("Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1")
model_lens = MLX_LM_Lens_Wrapper(model)

tokens = mx.array([[9707]])

lens_data = model_lens(
    tokens,
    return_dict=True
)

embeds = model_lens.get_embeds(tokens)

print(lens_data)
print(embeds)
```

## Examples

The `examples/` directory contains additional scripts illustrating various uses:

- `abliterate.py`
- `visualize_attentions.py`

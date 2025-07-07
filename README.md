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

from mlx_lm_lens.lens import open_lens

model_lens, tokenizer = open_lens("Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1")
# When loading it will trow a debug print like "Identified components: Embeddings=Embedding, Layers=24, Norm=True, LM Head=Embedding, Tied=True"

tokens = mx.array([[9707]]) # <-- "Hello"

lens_data = model_lens(
    tokens,
    return_dict=True
)

embeds = model_lens.get_embeds(tokens)

print(lens_data) # Results in:
# {'logits': array([[[5.71875, 4.78125, 0.542969, ..., -3.3125, -3.3125, -3.3125]]], dtype=bfloat16), 'hidden_states': [array([[[-0.0256348, 0.00537109, -0.010376, ..., 0.00285339, -0.00860596, 0.00369263]]], dtype=bfloat16), array([[[-0.0256348, 0.00537109, -0.010376, ..., 0.00285339, -0.00860596, 0.00369263]]], dtype=bfloat16), array([[[0.0966797, 0.0878906, 0.302734, ..., 0.357422, 0.0361328, 0.105957]]], dtype=bfloat16), array([[[0.0966797, 0.0878906, 0.302734, ..., 0.357422, 0.0361328, 0.105957]]], dtype=bfloat16), array([[[-0.03125, 0.230469, 0.275391, ..., 0.255859, 0.0395508, 0.0361328]]], dtype=bfloat16), array([[[-0.03125, 0.230469, 0.275391, ..., 0.255859, 0.0395508, 0.0361328]]], dtype=bfloat16), array([[[-0.945312, -6.09375, -2.34375, ..., -0.71875, -1.03125, 2.73438]]], dtype=bfloat16), array([[[-0.945312, -6.09375, -2.34375, ..., -0.71875, -1.03125, 2.73438]]], dtype=bfloat16), array([[[-2.54688, -10.0625, -4.9375, ..., -0.078125, -0.546875, 1.42188]]], dtype=bfloat16), array([[[-2.54688, -10.0625, -4.9375, ..., -0.078125, -0.546875, 1.42188]]], dtype=bfloat16), array([[[-2.59375, -10, -5.03125, ..., -0.0366211, -0.558594, 1.4375]]], dtype=bfloat16), array([[[-2.59375, -10, -5.03125, ..., -0.0366211, -0.558594, 1.4375]]], dtype=bfloat16), array([[[-2.64062, -10.1875, -4.9375, ..., 0.0859375, -0.392578, 1.375]]], dtype=bfloat16), array([[[-2.64062, -10.1875, -4.9375, ..., 0.0859375, -0.392578, 1.375]]], dtype=bfloat16), array([[[-2.67188, -10.4375, -4.90625, ..., -0.043457, -0.396484, 1.48438]]], dtype=bfloat16), array([[[-2.67188, -10.4375, -4.90625, ..., -0.043457, -0.396484, 1.48438]]], dtype=bfloat16), array([[[-2.65625, -10.5625, -4.90625, ..., 0.0253906, -0.371094, 1.45312]]], dtype=bfloat16), array([[[-2.65625, -10.5625, -4.90625, ..., 0.0253906, -0.371094, 1.45312]]], dtype=bfloat16), array([[[-2.70312, -10.625, -4.8125, ..., 0.181641, -0.382812, 1.36719]]], dtype=bfloat16), array([[[-2.70312, -10.625, -4.8125, ..., 0.181641, -0.382812, 1.36719]]], dtype=bfloat16), array([[[-2.73438, -10.625, -4.875, ..., 0.345703, -0.449219, 1.22656]]], dtype=bfloat16), array([[[-2.73438, -10.625, -4.875, ..., 0.345703, -0.449219, 1.22656]]], dtype=bfloat16), array([[[-2.6875, -10.5, -4.875, ..., 0.337891, -0.390625, 1.1875]]], dtype=bfloat16), array([[[-2.6875, -10.5, -4.875, ..., 0.337891, -0.390625, 1.1875]]], dtype=bfloat16), array([[[-2.65625, -10.5, -4.71875, ..., 0.476562, -0.332031, 1.21094]]], dtype=bfloat16), array([[[-2.65625, -10.5, -4.71875, ..., 0.476562, -0.332031, 1.21094]]], dtype=bfloat16), array([[[-2.6875, -10.5625, -4.71875, ..., 0.447266, -0.337891, 1.05469]]], dtype=bfloat16), array([[[-2.6875, -10.5625, -4.71875, ..., 0.447266, -0.337891, 1.05469]]], dtype=bfloat16), array([[[-2.73438, -10.625, -4.71875, ..., 0.566406, -0.242188, 0.871094]]], dtype=bfloat16), array([[[-2.73438, -10.625, -4.71875, ..., 0.566406, -0.242188, 0.871094]]], dtype=bfloat16), array([[[-2.75, -10.5625, -4.75, ..., 0.582031, -0.188477, 0.792969]]], dtype=bfloat16), array([[[-2.75, -10.5625, -4.75, ..., 0.582031, -0.188477, 0.792969]]], dtype=bfloat16), array([[[-2.78125, -10.5625, -4.75, ..., 0.570312, -0.152344, 0.796875]]], dtype=bfloat16), array([[[-2.78125, -10.5625, -4.75, ..., 0.570312, -0.152344, 0.796875]]], dtype=bfloat16), array([[[-2.8125, -10.625, -4.71875, ..., 0.707031, -0.296875, 0.75]]], dtype=bfloat16), array([[[-2.8125, -10.625, -4.71875, ..., 0.707031, -0.296875, 0.75]]], dtype=bfloat16), array([[[-2.875, -10.625, -4.75, ..., 0.714844, -0.178711, 0.566406]]], dtype=bfloat16), array([[[-2.875, -10.625, -4.75, ..., 0.714844, -0.178711, 0.566406]]], dtype=bfloat16), array([[[-2.82812, -10.625, -4.65625, ..., 0.710938, -0.234375, 0.566406]]], dtype=bfloat16), array([[[-2.82812, -10.625, -4.65625, ..., 0.710938, -0.234375, 0.566406]]], dtype=bfloat16), array([[[-2.6875, -10.625, -4.71875, ..., 0.828125, -0.208008, 0.0917969]]], dtype=bfloat16), array([[[-2.6875, -10.625, -4.71875, ..., 0.828125, -0.208008, 0.0917969]]], dtype=bfloat16), array([[[-2.65625, -10.625, -4.625, ..., 0.976562, -0.162109, 0.417969]]], dtype=bfloat16), array([[[-2.65625, -10.625, -4.625, ..., 0.976562, -0.162109, 0.417969]]], dtype=bfloat16), array([[[0.078125, -1.125, 0.09375, ..., 0.488281, -0.279297, -0.0859375]]], dtype=bfloat16), array([[[0.078125, -1.125, 0.09375, ..., 0.488281, -0.279297, -0.0859375]]], dtype=bfloat16), array([[[-1.85938, -0.402344, 1.96875, ..., 2.20312, -1.94531, -0.427734]]], dtype=bfloat16), array([[[-1.85938, -0.402344, 1.96875, ..., 2.20312, -1.94531, -0.427734]]], dtype=bfloat16), array([[[0.539062, -0.316406, 2.15625, ..., 1.35156, -1.95312, -1.60938]]], dtype=bfloat16), array([[[1.39062, -0.742188, 5.3125, ..., 3.3125, -4.71875, -5.15625]]], dtype=bfloat16)]}

print(embeds) # array([[[-0.0256348, 0.00537109, -0.010376, ..., 0.00285339, -0.00860596, 0.00369263]]], dtype=bfloat16)
```

## Examples

The `examples/` directory contains additional scripts illustrating various uses:

- `abliterate.py`
- `visualize_attentions.py`

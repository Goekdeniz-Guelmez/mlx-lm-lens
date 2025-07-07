# ✨ MLX-LM-LENS

Find the hidden meaning of LLMs.

MLX-LM-LENS provides a simple wrapper to inspect hidden states, attention scores, and embedding outputs of MLX-based language models.
This package is mainly intended as a research tool, though it can also be used to create real-world models, such as the Josiefied Abliterated models.

MLX-LM-LENS is built on top of the MLX-LM framework, so every model supported by MLX-LM works here as well.

Heavily inspired by the fantastic work on TransformerLens by Neel Nanda and Joseph Bloom, which laid the groundwork for interpretable transformer research.


## 🚀 Installation

```shell
pip install mlx-lm-lens
```

## ⚡ Quick Start

```python
import mlx.core as mx
from mlx_lm_lens.lens import open_lens

model_lens, tokenizer = open_lens("Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1")
# When loading it will print debug info like:
# "Identified components: Embeddings=Embedding, Layers=24, Norm=True, LM Head=Embedding, Tied=True"

tokens = mx.array([[9707]])  # "Hello"

lens_data = model_lens(
    tokens,
    return_dict=True
)

embeds = model_lens.get_embeds(tokens)

print(lens_data)
print(embeds)
```


## 📂 Examples

The examples/ directory contains scripts illustrating various uses:
- abliterate.py
- visualize_attentions.py


## 🙏 Acknowledgements

This project draws inspiration from:
	•	TransformerLens
Neel Nanda, Joseph Bloom, et al.


## Citing MLX-LM-LENS

The MLX-LM-LENS software suite was developed by Gökdeniz Gülmez. If you find MLX-LM-LENS useful in your research and wish to cite it, please use the following BibTex entry:

```
@software{
  MLX-LM-LENS,
  author = {Gökdeniz Gülmez},
  title = {{MLX-LM-LENS}: Find the hidden meaning of LLMs.},
  url = {https://github.com/Goekdeniz-Guelmez/mlx-lm-lens},
  version = {0.0.1},
  year = {2025},
}
```

---

Enjoy exploring the hidden structure of your models, or abliterate them, all localy on Apple Silicon!

✨ – Gökdeniz Gülmez

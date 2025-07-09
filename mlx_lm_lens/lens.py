from typing import Literal

import mlx.nn as nn
from mlx_lm.utils import load
from .lm_lens import MLX_LM_Lens_Wrapper

def open_lens(
        model,
        force_type: Literal["text", "vision", "vlm", "embedding", "audio", None] = "text",
    ):

    if isinstance(model, str):
        this_model, tokenizer = load(model)

        if force_type is not None:
            if force_type == "text":
                this_model = MLX_LM_Lens_Wrapper(this_model)
            else:
                raise NotImplemented("The other models like vlm etc. are not implemented jet, currently only mlx-lm (text based models)!")
        else:
            this_model = MLX_LM_Lens_Wrapper(model)

        return this_model, tokenizer
    elif isinstance(model, nn.Module):
        return MLX_LM_Lens_Wrapper(model), None
    else:
        raise NotImplemented("Model type is not suported, can only be a nn.Module, or path!")
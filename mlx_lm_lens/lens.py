from typing import Literal

from mlx_lm.utils import load
from .lm_lens import MLX_LM_Lens_Wrapper

def open_lens(
        model_name: str,
        force_type: Literal["text", "vision", "vlm", "embedding", "audio", None] = "text",
    ):

    model, tokenizer = load(model_name)

    if force_type is not None:
        if force_type == "text":
            model = MLX_LM_Lens_Wrapper(model)
        else:
            raise NotImplemented("The other models like vlm etc. are not implemented jet, currently only mlx-lm (text based models)!")
    else:
        model = MLX_LM_Lens_Wrapper(model)

    return model, tokenizer
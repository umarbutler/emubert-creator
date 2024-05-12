"""Convert the model from a Better Transformer into a vanilla Transformer."""
import os

import safetensors
import safetensors.torch

from config import model_dir
from transformers import RobertaForMaskedLM
from optimum.bettertransformer import BetterTransformer

def main():
    # Load the model.
    # NOTE Because the model is in the vanilla Transformer format, this will discard the Better Transformer weights and will effectively load a husk.
    model: RobertaForMaskedLM = RobertaForMaskedLM.from_pretrained(model_dir)

    # Convert the model into a Better Transformer.
    # NOTE This will rearchitect the model to be able to load the saved Better Transformer weights.
    model: RobertaForMaskedLM = BetterTransformer.transform(model)

    # Reload the model into the Better Transformer architecture.
    state = safetensors.torch.load_file(f'{model_dir}/model.safetensors', device='cpu')
    state |= {'lm_head.decoder.weight': model.lm_head.decoder.weight, 'lm_head.decoder.bias': model.lm_head.decoder.bias}
    model.load_state_dict(state)

    # Convert the model into a vanilla Transformer.
    model: RobertaForMaskedLM = BetterTransformer.reverse(model)

    # Preserve the Better Transformer weights.
    if os.path.exists(f'{model_dir}/model.bettertransformer.safetensors'):
        os.remove(f'{model_dir}/model.bettertransformer.safetensors')

    os.rename(f'{model_dir}/model.safetensors', f'{model_dir}/model.bettertransformer.safetensors')

    # Save the model.
    model.save_pretrained(model_dir)

if __name__ == '__main__':
    main()
"""Benchmark the model against other popular masked language models."""
from contextlib import nullcontext

import numpy as np
import torch

from tqdm import tqdm
from config import model_dir
from helpers import flatten, print_header, batch_generator
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForMaskedLM

# BEGIN CONFIG #
MODELS = [
    model_dir,
    'roberta-base',
    'bert-base-cased',
    'bert-base-uncased',
    'albert-base-v2',
]
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# END CONFIG #

def compute_pseudo_perplexity(texts: list[str], model: AutoModelForMaskedLM, tokenizer: PreTrainedTokenizer, batch_size: int = 8, device: str = 'cuda') -> list[float]:
    """Compute the provided masked language model's pseudo-perplexity on the given texts.
    
    Adapted from https://stackoverflow.com/a/70482924."""
    
    mask_token = tokenizer.mask_token_id
    autocast: torch.cuda.amp.autocast = torch.cuda.amp.autocast if device == 'cuda' else nullcontext
    losses = []
    
    for text in tqdm(texts):
        input: torch.Tensor = tokenizer(text, return_tensors = 'pt')['input_ids']
        repeated_inputs = input.repeat(input.size(-1) - 2, 1)
        
        mask = torch.ones(input.size(-1) - 1).diag(1)[:-2]
        masked_inputs: torch.Tensor = repeated_inputs.masked_fill(mask == 1, mask_token)
        
        labels = repeated_inputs.masked_fill(masked_inputs != mask_token, -100)
        
        masked_inputs = masked_inputs
        labels = labels
        
        # Compute the text's loss as the weighted average of the losses of batches of its masked instances. Batching is used to prevent memory overflow and weighted averaging is used to prevent the overweighting of losses for final batches less than the batch size.
        with autocast(), torch.inference_mode():
            text_losses = flatten([
                [float(
                    model(torch.stack([inputs_labels[0] for inputs_labels in batch]).to(device), labels = torch.stack([inputs_labels[1] for inputs_labels in batch]).to(device)).loss.item()
                )] * len(batch)
                
                for batch in batch_generator(zip(masked_inputs, labels), batch_size)
            ])
        
        losses.append(sum(text_losses) / len(text_losses))
    
    # Compute the pseudo-perplexity as the exponential of the average loss.
    perplexity = np.exp(sum(losses) / len(losses))
    
    return perplexity


def main():
    # Load the data.
    data = load_dataset('umarbutler/open-australian-legal-qa', split = 'train')['text']
    
    # Benchmark the models.
    print_header('Benchmarking')
    metrics = {model_name: None for model_name in MODELS}
    
    for model_name in MODELS:
        # Log.
        print_header(model_name, '=')
        
        # Load the model and tokenizer.
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(DEVICE)
        model = model.to_bettertransformer()
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Compute the model's pseudo-perplexity.
        perplexity = compute_pseudo_perplexity(data, model, tokenizer, BATCH_SIZE)

        # Log.
        metrics[model_name] = perplexity
        print(f'Pseudo-perplexity: {perplexity:,.2f}'), print()

    # Log the results.
    print_header('Results')
    for model_name, perplexity in metrics.items():
        print(f'{model_name}: {perplexity:,.2f}')
    
if __name__ == '__main__':
    main()
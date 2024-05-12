"""Benchmark the model against other popular masked language models."""
import tqdm
import lmppl
import torch
import msgspec
import semchunk

from config import model_dir, texts_path
from helpers import flatten, print_header
from datasets import load_dataset
from transformers import AutoTokenizer

# BEGIN CONFIG #
MODELS = [
    model_dir,
    'roberta-base',
    'bert-base-cased',
    'albert-base-v2',
    'distilbert-base-cased',
]
BATCH_SIZE = 8
# END CONFIG #

class Text(msgspec.Struct, frozen = True):
    text: str
    split: str

text_decoder = msgspec.json.Decoder(Text)

def main():
    # Load the test datasets.
    datasets = {}

    with open(texts_path, 'rb') as file:
        datasets['Open Australian Legal Corpus'] = [text.text for line in tqdm(file) if (text := text_decoder.decode(line)).split == 'test']

    datasets['Open Australian Legal QA'] = load_dataset('umarbutler/open-australian-legal-qa', split = 'train')['text']
    print()

    # Benchmarks the models against the datasets.
    print_header('Benchmarks')
    metrics = {dataset_name: {model_name: None for model_name in MODELS} for dataset_name in datasets.keys()}

    for dataset_name, data in datasets.items():
        # Log.
        print_header(dataset_name, '=')
        
        for model_name in MODELS:
            # Log.
            print_header(model_name, '-')
            
            # Chunk the data.
            tokeniser = AutoTokenizer.from_pretrained(model_name)
            
            def token_counter(text: str) -> int:
                return len(tokeniser.encode(text, add_special_tokens = False))
            
            context_window = tokeniser.model_max_length - (len(tokeniser.encode('a')) - 1)
            chunks = flatten(semchunk.chunk(text, context_window, token_counter) for text in tqdm(data))
            
            # Compute perplexities for the chunks.
            scorer = lmppl.LM(model_dir)
            scorer.model = scorer.model.to_bettertransformer() # Convert the model into a Better Transformer for faster inference.
            
            with torch.cuda.amp.autocast(): # Utilise automatic 16-bit mixed precision for faster inference.
                perplexities = scorer.get_perplexity(chunks, batch = BATCH_SIZE)
            
            # Average the perplexities.
            perplexity = sum(perplexities) / len(perplexities)
            metrics[dataset_name][model_name] = perplexity
            
            # Log.
            print(f'Pseudo-perplexity: {perplexity:,.2f}')
        
        print()

    # Log all the metrics.
    print('Final metrics:', metrics)

if __name__ == '__main__':
    main()
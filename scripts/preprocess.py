"""Clean documents, split them into training, validation and test sets, filter out short documents from the training set, deduplicate the training set, train a tokeniser and save the resulting data."""
import os
import re
import random

import xxhash
import msgspec
import orjsonl

from tqdm import tqdm
from config import model_dir, texts_path, corpus_path, base_model_name
from helpers import flatten, count_lines, print_header
from transformers import RobertaTokenizerFast

# BEGIN CONFIG #
TEST_SIZE = 10_000
VALIDATION_SIZE = 10_000
# END CONFIG #

class Doc(msgspec.Struct, frozen = True):
    """A document in the Corpus."""
    
    version_id: str
    type: str
    jurisdiction: str
    source: str
    citation: str
    url: str
    text: str

decoder = msgspec.json.Decoder(Doc)

def stream():
    """Stream documents from the Corpus."""
    
    with open(corpus_path, 'rb') as file:
        for json in file:
            yield decoder.decode(json)

def clean(text: str) -> str:
    """Clean text."""
    
    # Replace non-breaking spaces with regular spaces.
    text = text.replace('\xa0', ' ')

    # Replace return carriages followed by newlines with newlines.
    text = text.replace(r'\r\n', '\n')

    # Remove whitespace from lines comprised entirely of whitespace.
    text = re.sub(r'(?<=\n)\s*(?=\n)', '\n', text)

    # If the text begins with a newline or a newline preceded by whitespace, remove it and any preceding whitespace.
    text = re.sub(r'^\s*\n', '', text)

    # If the text ends with a newline or a newline succeeded by whitespace, remove it and any succeeding whitespace.
    text = re.sub(r'\n\s*$', '', text)

    # Remove spaces and tabs from the ends of lines.
    text = re.sub(r'[ \t]+\n', '\n', text)
    
    return text

def main():
    # Cache the number of documents in the Corpus.
    num_texts = count_lines(corpus_path)
    
    # Clean the documents, filtering out documents that, after cleaning, are empty.
    print_header('Cleaning')
    print(f'Original dataset size: {num_texts:,}')
    
    texts = [clean(text) for doc in tqdm(stream(), total = num_texts) if (text:=doc.text).strip()]
    
    print(f'Cleaned dataset size (ie, excluding empty documents post-stripping): {len(texts):,}')
    print(f'Difference: {num_texts - len(texts):,}')
    print()

    # Randomly split the texts into training, validation, and test sets.
    print_header('Splitting')
    
    random.seed(42), random.shuffle(texts)
    test = texts[:TEST_SIZE]
    validation = texts[TEST_SIZE : TEST_SIZE + VALIDATION_SIZE]
    train = texts[TEST_SIZE + VALIDATION_SIZE:]

    print(f'Training set size: {len(train):,}')
    print(f'Validation set size: {len(validation):,}')
    print(f'Test set size: {len(test):,}')
    print()

    del texts
    
    # Filter out extremely short documents.
    print_header('Deshorting')
    
    original_train_size = len(train)
    train = [text for text in train if len(text) >= 128]
    
    print(f'Deshorted training set size: {len(train):,}')
    print(f'Difference: {original_train_size - len(train):,}')
    print()
    
    # Deduplicate the training set.
    print_header('Deduplicating')
    
    original_train_size = len(train)
    train = {xxhash.xxh3_128_hexdigest(text): text for text in train}.values()
    
    print(f'Deduplicated training set size: {len(train):,}')
    print(f'Difference: {original_train_size - len(train):,}')
    print()
    
    # Train a tokeniser on the training set.
    print_header('Training tokeniser')
    tokeniser: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(base_model_name)
    base_model_vocabulary = tokeniser.get_vocab()
    tokeniser = tokeniser.train_new_from_iterator(
        train,
        vocab_size = len(base_model_vocabulary),
        length = len(train),
    )
    tokeniser.save_pretrained(model_dir)
    vocabulary = tokeniser.get_vocab()

    # Log differences between the current and previous vocabularies.
    base_model_vocabulary = set(base_model_vocabulary.keys())
    vocabulary = set(vocabulary.keys())

    print(f'New tokens: {len(vocabulary.difference(base_model_vocabulary)):,} ({len(vocabulary.difference(base_model_vocabulary)) / len(vocabulary) * 100:.2f}%)')
    print(f'Shared tokens: {len(vocabulary.intersection(base_model_vocabulary)):,} ({len(vocabulary.intersection(base_model_vocabulary)) / len(vocabulary) * 100:.2f}%)')
    print()
    
    # Save the training, validation, and test sets.
    print_header('Saving')
    
    os.makedirs(os.path.dirname(texts_path), exist_ok = True)
    orjsonl.save(texts_path, flatten([[{'text': text, 'split': split} for text in texts] for split, texts in {'train': train, 'validation': validation, 'test': test}.items()]))
    
    print('Saved!')

if __name__ == '__main__':
    main()
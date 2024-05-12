"""Shared paths and configurations for the Creator."""
import os

# Directories
root = os.path.dirname(os.path.abspath(__file__))
d = f'{root}/../data'
model_dir = f'{root}/../model'

# Paths
corpus_path = os.environ.get('OALC') or f'{d}/corpus.orjsonl'
texts_path = f'{d}/texts.orjsonl'
blocks_path = f'{d}/blocks.orjsonl'

# Model
base_model_name = 'roberta-base'
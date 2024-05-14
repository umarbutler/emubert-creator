# EmuBert Creator
EmuBert is the largest open-source masked language model for Australian law. This repository preserves the code used to create EmuBert.

If you're looking to download EmuBert, you may do so on [Hugging Face](https://huggingface.co/umarbutler/emubert).

## Setup 🛠️
The EmuBert Creator has only been tested on Python 3.11 but should work for later versions and *may* also work for earlier versions.

To set up the Creator, start by running the following commands:
```bash
git clone https://github.com/umarbutler/emubert-creator.git
cd emubert-creator
pip install -r requirements.txt
```

Next, download the version of the [Open Australian Legal Corpus](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus) you'd like to train EmuBert on by navigating to its [changelog](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus/blob/main/CHANGELOG.md), clicking on the version number you'd like to use, clicking on the file named `corpus.jsonl` and finally hitting 'download'. Any version of the Corpus that begins with the number 4 should be compatible with the Creator. The specific version of the Corpus used to produce EmuBert is 4.2.1 and can be downloaded [here](https://huggingface.co/datasets/umarbutler/open-australian-legal-corpus/blob/fe0cd918dbe0a1fb5afe09cfa682ec3dbc1b94ca/corpus.jsonl).

Finally, you can either place the Corpus in a directory named `data` in the root of the repository, define an environment variable named `OALC` that points to the Corpus or override the `corpus_path` variable in `scripts/config.py`.

## Usage 👩‍💻
To train EmuBert, run the following scripts in the `scripts` directory in order:
1. `preprocess.py`, which cleans documents, splits them into training, validation and test sets, filters out short documents from the training set, deduplicates the training set, trains a tokeniser and finally save the resulting data.
2. `block.py`, which splits texts into block of the same size as EmuBert's context window and saves them.
3. `train.py`, which trains EmuBert and saves it to a directory named `model` (unless the `model_dir` variable in `config.py` is overridden). If training is interrupted at any point, set the script's `RESUME` variable to `True`.
4. `convert.py`, which converts EmuBert from a Better Transformer into a vanilla Transformer.
5. `benchmark.py`, which benchmarks EmuBert against other popular masked language models.

## Licence 📜
The Creator is licensed under the [MIT License](LICENCE).
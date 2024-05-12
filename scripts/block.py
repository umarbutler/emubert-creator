"""Split documents into blocks of the same size as the model's context window and save them."""
import io

import mpire
import orjson
import orjsonl

from tqdm import tqdm
from config import model_dir, texts_path, blocks_path
from helpers import count_lines, batch_generator
from transformers import AutoTokenizer

# BEGIN CONFIG #
BATCH_SIZE = 50_000
TOKENISER = AutoTokenizer.from_pretrained(model_dir)
# END CONFIG #

bos_token_id = TOKENISER.bos_token_id
eos_token_id = TOKENISER.eos_token_id
pad_token_id = TOKENISER.pad_token_id
context_window = TOKENISER.model_max_length # NOTE It is important that the tokeniser has an actual maximum sequence length set (not the extremely large number some tokenisers use).

def write_block(file: io.BufferedWriter, block_split: str, block_tokens: list[int], block_attention_mask: list[int] = None) -> None:
    """Write the given block to the provided file."""
    
    file.write(
        orjson.dumps(
            {
                'split': block_split,
                'input_ids': block_tokens,
                'attention_mask': block_attention_mask if block_attention_mask is not None else [1] * len(block_tokens), # NOTE The only time an attention mask should be provided is for the final blocks of the validation and test datasets which will be padded, thereby requiring masks. The final block of the training dataset is dropped if it is shorter than the context window as Better Transformer does not support padding during training.
            }
        )
    )
    file.write(b'\n')
    
    block_tokens.clear()

def tokenise(text: str, split: str) -> tuple[list[int], str]:
    """Tokenise the provided text, also returning its split."""
    
    return TOKENISER.encode(text), split

def main():
    with mpire.WorkerPool() as pool, \
        open(blocks_path, 'wb') as file, \
        tqdm(total = count_lines(texts_path)) as bar:
            block_tokens = []
            block_split = None
            
            # Tokenise documents in batches of the specified size with multiprocessing.
            for batch in batch_generator(orjsonl.stream(texts_path), BATCH_SIZE):
                for doc_tokens, split in pool.imap(tokenise, batch):
                    # If we have encountered a document belonging to a new split, switch to that split and also write and clear the block if the block is not empty and it is the exact same size as the context window or it belongs to the validation or test split (in which case, padding may be added).
                    if split != block_split:
                        if block_tokens:
                            if len(block_tokens) == context_window:
                                write_block(file, block_split, block_tokens)
                            
                            elif block_split in {'validation', 'test'}:
                                block_attention_mask = [1] * len(block_tokens) + [0] * (context_window - len(block_tokens))
                                block_tokens.extend([pad_token_id] * (context_window - len(block_tokens)))
                                
                                write_block(file, block_split, block_tokens, block_attention_mask)
                            
                            else:
                                block_tokens.clear()
                        
                        block_split = split
                    
                    # Add the tokens to the block and write and clear the block whenever the context window is reached but skip adding end-of-sequence tokens to the start of blocks.
                    for token in doc_tokens:
                        if token == eos_token_id and not block_tokens:
                            continue
                        
                        block_tokens.append(token)

                        if len(block_tokens) == context_window:
                            write_block(file, block_split, block_tokens)
                    
                    bar.update()
            
            # If there are tokens left in the block, write it to the training data if necessary.
            if (block_tokens_len := len(block_tokens)) > 0:
                # If the block is shorter than the context window and it belongs to the validation or test splits, pad the remainder of the block and then write it, otherwise, if the block does not belong to the training split, discard it, unless the block is not shorter than the context window, in which case, write it.
                if (padding := context_window - block_tokens_len) > 0:
                    if block_split in {'validation', 'test'}:
                        block_attention_mask = [1] * block_tokens_len + [0] * padding
                        block_tokens.extend([pad_token_id] * padding)
                    
                        write_block(file, block_split, block_tokens, block_attention_mask)
                
                else:
                    write_block(file, block_split, block_tokens)
    
    print('Done!')

if __name__ == '__main__':
    main()
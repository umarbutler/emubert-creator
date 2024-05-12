"""A fast, memory-efficient dataset loader."""
import numpy as np
import msgspec

from tqdm import tqdm
from block import pad_token_id
from config import blocks_path
from helpers import count_lines
from torch.utils.data import Dataset

UNSIGNED_16_BIT_NUMPY_INTEGER = np.uint16

class JsonBlock(msgspec.Struct, frozen = True):
    split: str
    input_ids: list[int]
    attention_mask: list[int]

json_decoder = msgspec.json.Decoder(JsonBlock)

class BlockDataset(Dataset):
    def __init__(self, data: list[tuple[list[int], list[int]]]) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> dict[str, list[int]]:
        block = self.data[index]
        attention_mask = block != pad_token_id
        
        return {
            'input_ids': block,
            'labels': block,
            'attention_mask': attention_mask,
        }

def load(splits: set[str] = {'train', 'validation', 'test'}) -> dict[str, BlockDataset]:
    splits = set(splits)
    data = {split: [] for split in splits}
    
    with open(blocks_path, 'rb') as file:
        for block in tqdm(file, total = count_lines(blocks_path)):
            block = json_decoder.decode(block)
            
            split = block.split
            
            if split not in splits:
                continue
            
            data[split].append(np.array(block.input_ids, dtype = UNSIGNED_16_BIT_NUMPY_INTEGER))
    
    return {split: BlockDataset(data[split]) for split in splits}
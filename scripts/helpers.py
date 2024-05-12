"""Convenient helper functions."""
import itertools

from typing import Iterable, Generator

def count_lines(path: str) -> int:
    """Count the number of lines in a file."""
    
    with open(path, 'rb') as file:
        return sum(1 for _ in file)

def flatten(l):
    """Flatten a list of lists."""
    
    return [item for sublist in l for item in sublist]

def batch_generator(iterable: Iterable, batch_size: int) -> Generator[list, None, None]:
    """Generate batches of the specified size from the provided iterable."""
    
    iterator = iter(iterable)
    
    for first in iterator:
        yield list(itertools.chain([first], itertools.islice(iterator, batch_size - 1)))

def print_header(text: str, char: str = '#', width: int = 100) -> None:
    """Print a header."""
    
    # Compute the space necessary for the header.
    text_length = len(text) + 2
    
    # Raise an error if the space needed is more than the specified width.
    if text_length > width:
        raise ValueError('The space necessary for the header is larger than the specified width.')
    
    # Calculate the number of characters needed to affix to either size of the text.
    affix_length = (width - text_length) // 2
    
    # Construct the affixed text.
    affixed_text = f"{char * affix_length} {text} {char * affix_length}"
    
    # Add an extra character to the end of the affixed text if the total width is odd.
    if len(affixed_text) < width:
        affixed_text += char
    
    # Print the header.
    print(affixed_text)
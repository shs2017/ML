from typing import Sequence

def argmax(l: Sequence) -> int:
    """Arg-max implementation for a list"""
    best_index = None
    best_value = None

    for i, val in enumerate(l):
        if best_value is None or best_value < val:
            best_value = val
            best_index = i

    return best_index

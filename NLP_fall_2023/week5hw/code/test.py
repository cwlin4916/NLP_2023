
from __future__ import annotations

import logging
import math
import sys
from tqdm import tqdm 
from pathlib import Path

import torch
from torch import nn
from torch import optim
from jaxtyping import Float
from typeguard import typechecked
from typing import Counter
from collections import Counter
from SGD_convergent import ConvergentSGD #for 7d 

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union


lexicon_file_path = Path("./lexicons/chars-10.txt")
embeddings = {}
with open(lexicon_file_path, 'r') as f:
    next(f) #skip first line 
    for line in f:
        parts = line.strip().split()  # split by space
        word = parts[0]
        vector = list(map(float, parts[1:]))
        embeddings[word] = vector


#print firsst element in dictionary
print(f"{list(embeddings.keys())[0]}: {embeddings[list(embeddings.keys())[0]]}")

vocab = ["BOS"] 
my_embeddings = {word: torch.tensor(vector) for word, vector in embeddings.items() if word in vocab} 
if "OOL" in embeddings:
            my_embeddings["OOL"] = torch.tensor(embeddings["OOL"])


if "OOL" in my_embeddings:
    print(f"My OOL {embeddings['OOL']}")

print(f"My embeddings: {my_embeddings}")
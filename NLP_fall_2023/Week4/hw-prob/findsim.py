#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info

# Needed for Python's optional type annotations.
# We've included type annotations and recommend that you do the same, 
# so that mypy (or a similar package) can catch type errors in your code.
from typing import List, Optional

try:
    # PyTorch is your friend. Not *using* it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    import torch as th
    import torch.nn as nn
except ImportError:
    print("\nERROR! You need to install Miniconda, then create and activate the nlp-class environment.  See the INSTRUCTIONS file.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.

# Logging is in general a good practice to check the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# 
# - It prints to standard error (stderr), not standard output (stdout) by
#   default.  So these messages will normally go to your screen, even if
#   you have redirected stdout to a file.  And they will not be seen by
#   the autograder, so the autograder won't be confused by them.
# 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
# 
# In `parse_args`, we provided two command line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'.
#
# More info: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
# 
# In all the starter code for the NLP course, we've elected to create a separate
# logger for each source code file, stored in a variable named log that
# is globally visible throughout the file.  That way, calls like log.info(...)
# will use the logger for the current source code file and thus their output will 
# helpfully show the filename.  You could configure the current file's logger using
# log.basicConfig(...), whereas logging.basicConfig(...) affects all of the loggers.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words(bagpipe)
    """

    def __init__(self) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
      
        
        # FINISH THIS FUNCTION

        # Store your stuff! Both the word-index mapping and the embedding matrix.
        #
        # Do something with this size info?
        # PyTorch's th.Tensor objects rely on fixed-size arrays in memory.
        # One of the worst things you can do for efficiency is
        # append row-by-row, like you would with a Python list.
        
        self.word_to_index={} 
        self.index_to_word={}
        self.embeddings = None # type: Optional[th.Tensor]
        

        # Probably make the entire list all at once, then convert to a th.Tensor.
        # Otherwise, make the th.Tensor and overwrite its contents row-by-row.
        
        
    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        
        # FINISH THIS FUNCTION
        lexicon = Lexicon()#instantiating a Lexicon object
        word_to_index={}
        embeddings_list=[]
        
        #open file and read lines 
        with open(file, 'r') as f:
            first_line = next(f).strip()  # Peel off the special first line.
            
            #parse the remaining lines
            for index, line in enumerate(f):  # All of the other lines are regular.
                parts = line.strip().split()
                embedding=list(map(float, parts[1:])) 
                word=parts[0]
                #let me explain this part: 1. parts[1:] is a list of strings, we need to convert them to float
                #2. map(float, parts[1:]) is a map object, we need to convert it to a list
                #3. list(map(float, parts[1:])) is a list of floats
                
                word_to_index[word] = index 
                embeddings_list.append(embedding) 
                pass  # `pass` is a placeholder. Replace with real code!
            
            
          # Maybe put args here. Maybe follow Builder pattern.
        lexicon.word_to_index=word_to_index
        lexicon.index_to_word={index:word for word, index in word_to_index.items()} #
        
        lexicon.embeddings=th.tensor(embeddings_list)
        return lexicon

    def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""
        # FINISH THIS FUNCTION
        if word not in self.word_to_index:
            return ["Word not in vocabulary."]
        #1. fetch the embedding of the given word 
        word_idx=self.word_to_index[word]
        word_vec=self.embeddings[word_idx]
        
        #2. adjust vector if 'plus' and 'minus' words are provided 
        if plus is not None and minus is not None:
            plus_idx=self.word_to_index[plus] 
            minus_idx=self.word_to_index[minus]
            
            word_vec=word_vec+self.embeddings[plus_idx]-self.embeddings[minus_idx]
        
        #3. compute cosine similarities
        cos_sim=th.nn.functional.cosine_similarity(self.embeddings, word_vec.unsqueeze(0), dim=1)
         #`dim=1` specifies that the cosine similarity is computed along columns. In the context of 2d tensors  #this means that the cosine similarity is computed along rows.
        
        sorted_indices = th.argsort(cos_sim, descending=True)
        #convert the tensor to a list of indices
        sorted_indices_10=sorted_indices[1:11].tolist() #the index [1:11] is used to exclude the first element, which is the index of the given word itself
        
        first_few_itmes=list(self.index_to_word.keys())[:10] #the first 10 items in the index_to_word dictionary 
        print(first_few_itmes) 
        print(sorted_indices[1:11])

        similar_words = [self.index_to_word[idx] for idx in sorted_indices_10[1:11]] 
        # # sort indices by similarity to the given word
        # The star above forces you to use `plus` and `minus` only
        # as named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.
        #
        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")
        # Keep going!
        # Be sure that you use fast, batched computations
        # instead of looping over the rows. If you use a loop or a comprehension
        # in this function, you've probably made a mistake.
        return similar_words


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(" ".join(similar_words))  # print all words on one line, separated by spaces


if __name__ == "__main__":
    main()
    
    



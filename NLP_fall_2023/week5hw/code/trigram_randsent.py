
#this is a modified code to sample from my trigram models
import argparse
import logging
import math
from pathlib import Path
import random
from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

#this is a modified code to sample from my trigram models
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    
    # #is this code necessary? 
    # parser.add_argument(
    #     "test_files",
    #     type=Path,
    #     nargs="*"
    # )
    
    
    #this is to add the number of sentences 'k' to generate
    parser.add_argument(
        "--num_sentences",
        type=int,
        default=1,
        help="number of sentences to generate"
    )
    
    #maximum length 
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="maximum length of sentence to generate"
    )
    
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

    return parser.parse_args()

#write a method that samples sentences from the language model. 
# The sampling method should accept the language model 
# as an argument and return a sentence as a string or list of tokens.

def sample_next_token(lm: LanguageModel, x:str, y:str) -> str: 
    
    """
    Sample the next token z given a bigram context (x, y) using a provided language model.
    
    Args:
    language_model: an instance of a language model class with a .prob() method
    x, y: the bigram context
    
    Returns:
    The next token z
    """
    #compute probabilities for each possible next word z 
    prob_distribution={} 
    for z in lm.vocab: 
        prob_distribution[z] = lm.prob(x, y, z) #computing this prob is specific to each model
    
    words = list(prob_distribution.keys())
    probs = [prob_distribution[word] for word in words]
    return random.choices(words, weights=probs)[0] #this is to sample from the distribution
        
        
    
def sample_sentences(lm: LanguageModel, num_sentences: int, max_length: int) -> str: 
    sentence =[]
    current_context=("BOS", "BOS") #we are using a trigram model
    for _ in range(max_length): 
        next_token = sample_next_token(lm, current_context[0], current_context[1])
#this function needs to be implemented 
        sentence.append(next_token)
        if next_token == "EOS":
            break
        current_context = (current_context[1], next_token)
    return " ".join(sentence) 


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    log.info("Testing...")
    lm = LanguageModel.load(args.model)
    
    for _ in range(args.num_sentences):
        sentence = sample_sentences(lm, args.num_sentences, args.max_length)
        print(sentence)

if __name__ == "__main__":
    main()

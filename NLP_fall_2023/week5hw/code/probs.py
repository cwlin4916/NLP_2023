#!/usr/bin/env python3

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


log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

##### TYPE DEFINITIONS (USED FOR TYPE ANNOTATIONS)
from typing import Iterable, List, Optional, Set, Tuple, Union

Wordtype = str  # if you decide to integerize the word types, then change this to int
Vocab    = Set[Wordtype]
Zerogram = Tuple[()]
Unigram  = Tuple[Wordtype]
Bigram   = Tuple[Wordtype, Wordtype]
Trigram  = Tuple[Wordtype, Wordtype, Wordtype]
Ngram    = Union[Zerogram, Unigram, Bigram, Trigram]
Vector   = List[float]
TorchScalar = Float[torch.Tensor, ""] # a torch.Tensor with no dimensions, i.e., a scalar


##### CONSTANTS
BOS: Wordtype = "BOS"  # special word type for context at Beginning Of Sequence
EOS: Wordtype = "EOS"  # special word type for observed token at End Of Sequence
OOV: Wordtype = "OOV"  # special word type for all Out-Of-Vocabulary words
OOL: Wordtype = "OOL"  # special word type whose embedding is used for OOV and all other Out-Of-Lexicon words


##### UTILITY FUNCTIONS FOR CORPUS TOKENIZATION

def read_tokens(file: Path, vocab: Optional[Vocab] = None) -> Iterable[Wordtype]:
    """Iterator over the tokens in file.  Tokens are whitespace-delimited.
    If vocab is given, then tokens that are not in vocab are replaced with OOV."""

    with open(file) as f:
        for line in f:
            for token in line.split():
                if vocab is None or token in vocab:
                    yield token
                else:
                    yield OOV  # replace this out-of-vocabulary word with OOV
            yield EOS  # Every line in the file implicitly ends with EOS.


def num_tokens(file: Path) -> int:
    """Give the number of tokens in file, including EOS."""
    return sum(1 for _ in read_tokens(file))


def read_trigrams(file: Path, vocab: Vocab) -> Iterable[Trigram]:
    """Iterator over the trigrams in file.  Each triple (x,y,z) is a token z
    (possibly EOS) with a left context (x,y)."""
    x, y = BOS, BOS
    for z in read_tokens(file, vocab):
        yield (x, y, z)
        if z == EOS:
            x, y = BOS, BOS  # reset for the next sequence in the file (if any)
        else:
            x, y = y, z  # shift over by one position.


def draw_trigrams_forever(file: Path, 
                          vocab: Vocab, 
                          randomize: bool = False) -> Iterable[Trigram]:
    """Infinite iterator over trigrams drawn from file.  We iterate over
    all the trigrams, then do it again ad infinitum.  This is useful for 
    SGD training.  
    
    If randomize is True, then randomize the order of the trigrams each time.  
    This is more in the spirit of SGD, but the randomness makes the code harder to debug, 
    and forces us to keep all the trigrams in memory at once.
    """
    trigrams = read_trigrams(file, vocab)
    if not randomize:
        import itertools
        return itertools.cycle(trigrams)  # repeat forever
    else:
        import random
        pool = tuple(trigrams)   
        while True:
            for trigram in random.sample(pool, len(pool)):
                yield trigram

##### READ IN A VOCABULARY (e.g., from a file created by build_vocab.py)

def read_vocab(vocab_file: Path) -> Vocab:
    vocab: Vocab = set()
    with open(vocab_file, "rt") as f:
        for line in f:
            word = line.strip()
            vocab.add(word)
    log.info(f"Read vocab of size {len(vocab)} from {vocab_file}")
    return vocab

##### LANGUAGE MODEL PARENT CLASS

class LanguageModel:

    def __init__(self, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        self.progress = 0   # To print progress.

        self.event_count:   Counter[Ngram] = Counter()  # numerator c(...) function.
        self.context_count: Counter[Ngram] = Counter()  # denominator c(...) function.
        # In this program, the argument to the counter should be an Ngram, 
        # which is always a tuple of Wordtypes, never a single Wordtype:
        # Zerogram: context_count[()]
        # Bigram:   context_count[(x,y)]   or equivalently context_count[x,y]
        # Unigram:  context_count[(y,)]    or equivalently context_count[y,]
        # but not:  context_count[(y)]     or equivalently context_count[y]  
        #             which incorrectly looks up a Wordtype instead of a 1-tuple

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None
        return len(self.vocab)

    def count_trigram_events(self, trigram: Trigram) -> None:
        """Record one token of the trigram and also of its suffixes (for backoff)."""
        (x, y, z) = trigram
        self.event_count[(x, y, z )] += 1
        self.event_count[   (y, z )] += 1
        self.event_count[      (z,)] += 1  # the comma is necessary to make this a tuple
        self.event_count[        ()] += 1

    def count_trigram_contexts(self, trigram: Trigram) -> None:
        """Record one token of the trigram's CONTEXT portion, 
        and also the suffixes of that context (for backoff)."""
        (x, y, _) = trigram    # we don't care about z
        self.context_count[(x, y )] += 1
        self.context_count[   (y,)] += 1
        self.context_count[     ()] += 1

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes an estimate of the trigram log probability log p(z | x,y)
        according to the language model.  The log_prob is what we need to compute
        cross-entropy and to train the model.  It is also unlikely to underflow,
        in contrast to prob.  In many models, we can compute the log_prob directly, 
        rather than first computing the prob and then calling math.log."""
        class_name = type(self).__name__
        if class_name == LanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling log_prob on an instance of LanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.log_prob is not implemented yet (you should override LanguageModel.log_prob)"
        )

    @classmethod
    def load(cls, source: Path) -> "LanguageModel":
        import pickle  # for loading/saving Python objects
        log.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            log.info(f"Loaded model from {source}")
            return pickle.load(f)

    def save(self, destination: Path) -> None:
        import pickle
        log.info(f"Saving model to {destination}")
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info(f"Saved model to {destination}")

    def train(self, file: Path) -> None:
        """Create vocabulary and store n-gram counts.  In subclasses, we might
        override this with a method that computes parameters instead of counts."""

        log.info(f"Training from corpus {file}")

        # Clear out any previous training.
        self.event_count   = Counter()
        self.context_count = Counter()

        for trigram in read_trigrams(file, self.vocab):
            self.count_trigram_events(trigram)
            self.count_trigram_contexts(trigram)
            self.show_progress()

        sys.stderr.write("\n")  # done printing progress dots "...."
        log.info(f"Finished counting {self.event_count[()]} tokens")

    def show_progress(self, freq: int = 5000) -> None:
        """Print a dot to stderr every 5000 calls (frequency can be changed)."""
        self.progress += 1
        if self.progress % freq == 1:
            sys.stderr.write(".")


##### SPECIFIC FAMILIES OF LANGUAGE MODELS

class CountBasedLanguageModel(LanguageModel):

    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        # For count-based language models, it is usually convenient
        # to compute the probability first (by dividing counts) and
        # then taking the log.
        prob = self.prob(x, y, z)
        if prob == 0.0:
            return -math.inf
        return math.log(prob)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Computes a smoothed estimate of the trigram probability p(z | x,y)
        according to the language model.
        """
        class_name = type(self).__name__
        if class_name == CountBasedLanguageModel.__name__:
            raise NotImplementedError("You shouldn't be calling prob on an instance of CountBasedLanguageModel, but on an instance of one of its subclasses.")
        raise NotImplementedError(
            f"{class_name}.prob is not implemented yet (you should override CountBasedLanguageModel.prob)"
        )

class UniformLanguageModel(CountBasedLanguageModel):
    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        return 1 / self.vocab_size


class AddLambdaLanguageModel(CountBasedLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab)
        if lambda_ < 0.0:
            raise ValueError(f"Negative lambda argument of {lambda_} could result in negative smoothed probs")
        self.lambda_ = lambda_

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        assert self.event_count[x, y, z] <= self.context_count[x, y]
        return ((self.event_count[x, y, z] + self.lambda_) /
                (self.context_count[x, y] + self.lambda_ * self.vocab_size))

        # Notice that summing the numerator over all values of typeZ
        # will give the denominator.  Therefore, summing up the quotient
        # over all values of typeZ will give 1, so sum_z p(z | ...) = 1
        # as is required for any probability function.

class BackoffAddLambdaLanguageModel(AddLambdaLanguageModel):
    def __init__(self, vocab: Vocab, lambda_: float) -> None:
        super().__init__(vocab, lambda_)

    def prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
    #     # TODO: Reimplement me so that I do backoff
    #     #vocab size  
         assert self.event_count[x, y, z] <= self.context_count[x, y] #c(xyz) <= c(xy) this is the case for all trigrams
         
         #zerogramprobability
         V=self.vocab_size
         #unigram
         unigram_prob=(self.event_count[(z,)] + self.lambda_) / (self.context_count[()] + self.lambda_ * V)
         #bigram
         bigram_prob=(self.event_count[(y,z)] + self.lambda_ * V * unigram_prob) / (self.context_count[(y,)] + self.lambda_ * V)
         #trigram
         return (self.event_count[(x,y,z)] + self.lambda_ * V * bigram_prob) / (self.context_count[(x,y)] + self.lambda_ * V)


def load_lexicon(lexicon_file_path: str) -> dict:
    embeddings = {}
    with open(lexicon_file_path, 'r') as f:
        next(f) # skip the first line
        for line in f:
            parts = line.strip().split()  # split by space
            word = parts[0]
            vector = list(map(float, parts[1:]))
            embeddings[word] = vector
    return embeddings


class EmbeddingLogLinearLanguageModel(LanguageModel, nn.Module):
    # Note the use of multiple inheritance: we are both a LanguageModel and a torch.nn.Module.
    def __init__(self, vocab: Vocab, lexicon_file: Path, l2: float, epochs: int) -> None:
        super().__init__(vocab)
        if l2 < 0:
            raise ValueError("Negative regularization strength {l2}")
        self.l2: float = l2
        self.epochs: int = epochs
        self.vocab_dict = {word: i for i, word in enumerate(self.vocab)}

        self.embeddings = {word: torch.tensor(vector) for word, vector in load_lexicon(str(lexicon_file)).items() if word in vocab} 
        
        self.embeddings["OOL"] = torch.tensor(load_lexicon(str(lexicon_file))["OOL"])
        self.vocab_values_stack = torch.stack([self.get_embedding(word) for word in self.vocab_dict])  # New line: store the embeddings for all "z"

         # Add the "OOL" vector if it exists in all_embeddings
        
        some_word = list(self.embeddings.keys())[1] #next__iter__ returns the next item from the iterator
    
        self.dim = self.embeddings[some_word].size(0) #dimension of parameter matrices
        self.X = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
        self.Y = nn.Parameter(torch.zeros((self.dim, self.dim)), requires_grad=True)
#We will add a new method 

    def get_embedding(self, word: Wordtype) -> torch.Tensor:
        """Return the embedding for word, or the OOL embedding if word is not in the vocabulary."""
        return self.embeddings.get(word, self.embeddings["OOL"]) 
    def logits(self, x: Wordtype, y: Wordtype) -> torch.Tensor:
        """Calculate logits for all vocabulary given context words x and y"""
        # Initialize vector to store logits
        logit_vec = torch.zeros(len(self.vocab))

        # Get embeddings for x and y
        x_emb = self.get_embedding(x)
        y_emb = self.get_embedding(y)
        #maybe can create this as a 
        # Create a stack of all z word embeddings from vocab
        
        # Calculate logits
        z_vecs=self.vocab_values_stack
        logit_vec = (x_emb @ self.X @ z_vecs.T) + (y_emb @ self.Y @ z_vecs.T)
        
        return logit_vec
    
    
    #normalization
    def compute_Z(self, x_emb: torch.Tensor, y_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute the normalization constant Z(xy) using vectorized operations for speedup.
        x_emb and y_emb are the embeddings for words x and y.
        """

        E = self.vocab_values_stack # Stack all word vectors to create the matrix E
        part1 = x_emb @ self.X @ E.T  # Compute x^T X E using matrix-matrix multiplication
        part2 = y_emb @ self.Y @ E.T  # Compute y^T Y E using matrix-matrix multiplication
        
        Z_xy = torch.exp(part1 + part2).sum()  # Take the exponent and sum up
        return Z_xy
        
    def log_prob(self, x: Wordtype, y: Wordtype, z: Wordtype) -> float:
        """Return log p(z | xy) according to this language model."""
        x_emb= self.get_embedding(x)
        y_emb= self.get_embedding(y)
        z_emb= self.get_embedding(z)
        numerator = torch.exp(x_emb @self.X @y_emb + y_emb @self.Y@z_emb)
        #now compute Z(xy) the partition function 
        Z_xy =self.compute_Z(x_emb, y_emb) 
        log_prob = torch.log(numerator/Z_xy) 
        return log_prob.item() #item() returns the value of this tensor as a standard Python number. This only works for tensors with one element.

    @typechecked
    def log_prob_tensor(self, x: Wordtype, y: Wordtype, z: Wordtype) -> TorchScalar:
        """Return the same value as log_prob, but stored as a tensor."""        
        logit_vec = self.logits(x,y) #compute the logits
        log_pxyz = logit_vec[self.vocab_dict.get(z)]
        x_emb= self.get_embedding(x)
        y_emb= self.get_embedding(y)
        log_z=torch.logsumexp(logit_vec, dim=0)  #normalize we need to consider log of the partition function
        return log_pxyz-log_z #return the log of the probability of the trigram

    def train(self, file: Path):    # type: ignore
         ### it overrides not only `LanguageModel.train` (as desired) but also `nn.Module.train` (which has a different type). 
         ### The `type: ignore` comment above tells the type checker to ignore this inconsistency.
        gamma0 = 1e-2  # initial learning rate, this is from 7b) 
        optimizer = optim.SGD(self.parameters(), lr=gamma0)
        nn.init.zeros_(self.X)   # type: ignore
        nn.init.zeros_(self.Y)   # type: ignore
        N = num_tokens(file)
        log.info("Start optimizing on {N} training tokens...")
        t=0 #initialize the number of updates so far. 
        for e in range(self.epochs):
            #loop over traigrams in the training data
            F_epoch = 0.0
            for i, trigram in enumerate(tqdm(read_trigrams(file, self.vocab), total=N)): # To get the training examples, you can use the `read_trigrams` function
                gamma = gamma0/(1+gamma0*2*self.l2*t/N) #current step size
                x,y,z=trigram
                #compute the forward 
                Fi = self.log_prob_tensor(x,y,z)  #need tensor for bookkeeping instead of float 
                Fi += - self.l2*(self.X.norm()+self.Y.norm())#For each successive training example i, compute the stochastic
                 # we want to maximize Fi_theta, but SGD minimizes, so we negate it
                #compuate thegradients by back propagation 
                (-Fi).backward()  
                
                optimizer.step()  #, update the parameters in the direction of the gradient via step method
                optimizer.zero_grad() 
                
                t+=1 #increment the number of updates so far 
                F_epoch += Fi.item()
            log.info("done optimizing.")
            print(f"epoch {e+1}: F = {F_epoch/N}") #print the value of F after each epoch
            # print("Model parameters:", self.X, self.Y)
            print(f"Finished training on {N} tokens")


            
      
        #####################

        log.info("done optimizing.")


class ImprovedLogLinearLanguageModel(EmbeddingLogLinearLanguageModel):
    # TODO: IMPLEMENT ME!
    
    # This is where you get to come up with some features of your own, as
    # described in the reading handout.  This class inherits from
    # EmbeddingLogLinearLanguageModel and you can override anything, such as
    # `log_prob`.

    # OTHER OPTIONAL IMPROVEMENTS: You could override the `train` method.
    # Instead of using 10 epochs, try "improving the SGD training loop" as
    # described in the reading handout.  Some possibilities:
    #
    # * You can use the `draw_trigrams_forever` function that we
    #   provided to shuffle the trigrams on each epoch.
    #
    # * You can choose to compute F_i using a mini-batch of trigrams
    #   instead of a single trigram, and try to vectorize the computation
    #   over the mini-batch.
    #
    # * Instead of running for exactly 10*N trigrams, you can implement
    #   early stopping by giving the `train` method access to dev data.
    #   This will run for as long as continued training is helpful,
    #   so it might run for more or fewer than 10*N trigrams.
    #
    # * You could use a different optimization algorithm instead of SGD, such
    #   as `torch.optim.Adam` (https://pytorch.org/docs/stable/optim.html).
    #
    pass



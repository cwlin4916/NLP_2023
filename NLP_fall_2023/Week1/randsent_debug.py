#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()


class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Parse the input grammar file
        self.rules ={} # dictionary mapping lhs to list of (rhs, weight) tuples
        self._load_rules_from_file(grammar_file)

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """
        
        with open(grammar_file, 'r') as f: 
            for line in f:
                # print(f'Reading line: {line}')  # Debugging line
                if line.strip() == '' or line.strip().startswith('#'):
                    continue
                parts=line.strip().split('\t')
                # print(f"Debugging: parts = {parts}")
                # print(ord(parts[0][-1]))
                # print(f"Debugging: parts = {parts}, length = {len(parts)}")  # Debugging line


                weight, lhs, rhs = parts 
                if lhs not in self.rules: 
                    self.rules[lhs] = [] 
                self.rules[lhs].append((rhs, float(weight))) 
        
        # print(self.rules)    
        # raise NotImplementedError
    def sample(self, derivation_tree, max_expansions, start_symbol, expansions_so_far=None):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from
            
            expansions_so_far (list): list containing the number of expansions so far

        Returns:
            str: the random sentence or its derivation tree
        """
        # Initialize expansions_so_far if it's None
        if expansions_so_far is None:
            expansions_so_far = [max_expansions]

        # print(f"Debugging: Current max_expansions = {expansions_so_far[0]}")

        # If max_expansions is 0, return "..."
        if expansions_so_far[0] <= 0:
            return "..."

        # Decrement the value in the list
        expansions_so_far[0] -= 1

        if start_symbol not in self.rules:
            return start_symbol  # if the start symbol is a terminal, return it

        possible_expansions = self.rules[start_symbol]  # list of (rhs, weight) tuples
        weights = [float(weight) for _, weight in possible_expansions]  # list of weights
        total_weight = sum(weights)  # sum of weights
        normalized_weights = [w / total_weight for w in weights]  # list of normalized weights

        # Choose a random expansion
        rhs, _ = random.choices(possible_expansions, weights=normalized_weights, k=1)[0]

        # print(f"Debugging: rhs = {rhs}")

        expanded = []
        for symbol in rhs.split():
            expanded.append(self.sample(derivation_tree, max_expansions, symbol, expansions_so_far))
            # print(f"Debugging: Expanding symbol = {symbol}")

        # Increment it back before the function ends
        # expansions_so_far[0] += 1

        if derivation_tree:  # if we want a derivation tree
            return f"({start_symbol} {' '.join(expanded)})"  # return the derivation tree
        else:
            return ' '.join(expanded)  # return the sentence


####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
    



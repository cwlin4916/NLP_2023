#!/usr/bin/env python3
"""
#Modify fileprob to obtain a new program textcat that does text categorization via Bayes’ Theorem.
The two programs both use the same probs module to get the language model probabilities. 
"""
import matplotlib.pyplot as plt
import argparse
import logging
import math
import re
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

  # Key: file length, Value: total number of files with this length

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the trained model for the first category",
    )
    
    parser.add_argument(
        "model2",
		type=Path,
		help="path to the trained model for the second category",
	)
    
    parser.add_argument(
		"prior_prob",
		type=float,
		help="prior probability of the first category"
	)
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
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


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


def main():
    success_count = {}  # Key: file length, Value: number of successful classifications
    total_count = {}
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    
    log.info("Loading languagem models...")
    lm1 = LanguageModel.load(args.model1)
    lm2 = LanguageModel.load(args.model2)
    
    #initialize counters for each model
    count1 =0 
    count2 =0
    log.info("Performing text categorization...")
    for file in args.test_files:
        
        log_prob1: float = file_log_prob(file, lm1)
        log_prob2: float = file_log_prob(file, lm2)
        #caluclate the final based on baye's theorem
        post_prob1= (log_prob1 + math.log(args.prior_prob))
        # print(post_prob1)
        post_prob2= (log_prob2 + math.log(1-args.prior_prob))
        # print(post_prob2)
        category = "gen.model" if post_prob1 > post_prob2 else "spam.model"
        # print(f"{category}\t{file}")
        
        # 1. Extract file length (assuming the length is the first part of the filename)
        file_length = int(re.search(r"\d+", file.name).group())
            # 2. Update total_count
        total_count[file_length] = total_count.get(file_length, 0) + 1
		#extracting the type of the file (assuming its the first three letters)
        file_type = file.name.split('.')[0] 
        if (category == "gen.model" and file_type == "gen") or (category == "spam.model" and file_type == "spam"):
            success_count[file_length] = success_count.get(file_length, 0) + 1
        
        if category == "gen.model":
            count1 +=1
        else:
            count2 +=1
    accuracy = {}
    for length in total_count:
        accuracy[length] = success_count.get(length, 0) / total_count[length]
    total_files=len(args.test_files)
    
    
 
    # print(f"{count1} files were more probably gen.model ({count1 / total_files * 100:.2f}%)")
    # print(f"{count2} files were more probably spam.model ({count2 / total_files * 100:.2f}%)")

  #Ensuring compatilbity of the language models with the test files
    if lm1.vocab != lm2.vocab:
        raise ValueError("The language models are incompatible.")
    lengths = list(accuracy.keys())
    filtered_lengths = [length for length in accuracy.keys() if 0<= length <= 150] 
    accuracies = [accuracy[length] for length in lengths]
    filtered_accuracies= [accuracy[length] for length in filtered_lengths]
    
    plt.scatter(lengths, accuracies)
    plt.xlabel('File Length')
    plt.ylabel('Classification Accuracy')
    plt.title('File Length vs Classification Accuracy')
    plt.show()


    plt.scatter(filtered_lengths, filtered_accuracies)
    plt.xlabel('File Length')
    plt.ylabel('Classification Accuracy')
    plt.title('File Length vs Classification Accuracy (Filtered to 0-150)')
    plt.show()


    if lm1.vocab != lm2.vocab:
        raise ValueError("The language models are incompatible.")

# Extract lengths and corresponding accuracies




if __name__ == "__main__":
    main()

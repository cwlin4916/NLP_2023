# from collections import defalutdict 

from pathlib import Path
import re

# def extract_file_lengths(directory_path):
#     lengths={} 
#     for filepath in Path(directory_path).glob("*.txt"): #the .glob() method returns a list of paths matching a pathname pattern
#         # Assuming filenames start with length e.g., "120_filename.txt"
#         match = re.search(r"\d+", filepath.name) 
#         #we add match.group() to extract the length
#         if match: 
#             length=int(match.group()) 
#             lengths[filepath]=length
#     return lengths

# gen_lengths = extract_file_lengths('./data/gen_spam/dev/gen/')
# spam_lengths = extract_file_lengths('./data/gen_spam/dev/spam/')
# print(gen_lengths)
# print(spam_lengths)


def number_of_files(directory_path: str) -> int:
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print("Invalid directory path.")
        return 0

    number_of_files = 0
    for file in directory.iterdir():
        if file.is_file():
            number_of_files += 1
            
    return number_of_files

# Test
print(number_of_files('./data/gen_spam/dev/gen/'))
print(number_of_files('./data/gen_spam/dev/spam/'))
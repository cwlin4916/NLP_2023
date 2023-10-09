#To get the total number of tokens across all the development files, 
# #you would typically read each file in the development set
# (both 'gen' and 'spam'), tokenize the content,
# and count the number of tokens.
from fileprob import num_tokens 
from pathlib import Path

#the following code counts the number of tokens in a single file
def count_tokens_in_directory(directory_path):
    total_tokens =0 
    for filepath in Path(directory_path).glob("*.txt"):
        total_tokens +=num_tokens(filepath)
    return total_tokens

gen_tokens = count_tokens_in_directory('./data/gen_spam/dev/gen/')
print(gen_tokens)
spam_tokens = count_tokens_in_directory('./data/gen_spam/dev/spam/')

print(spam_tokens)
total_tokens= gen_tokens + spam_tokens
print(total_tokens)

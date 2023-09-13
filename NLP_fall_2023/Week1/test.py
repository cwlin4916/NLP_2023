from randsent import Grammar

def test_grammar(): 
    g=Grammar('grammar.gr')
    # print(g.rules) 
    
    print("\nSample sentences:")
    
    print("Sample 1:", g.sample(False, 30, 'ROOT'))
    
     # Test with different start symbol, max_expansions=5, and derivation_tree=False
    # print("Sample 2:", g.sample(False, 5, 'S'))
    
    
    
if __name__ == '__main__': # ensures that test_grammar() is only called when this file is run directly
    test_grammar()



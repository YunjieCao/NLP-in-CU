"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 2 - Parsing with Context Free Grammars 
Yassine Benajiba
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
from math import log as log

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar
        self.rhs_to_rules = self.grammar.rhs_to_rules
        self.lhs_to_rules = self.grammar.lhs_to_rules
        self.startsymbol = self.grammar.startsymbol

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        n = len(tokens)
        parse_table, prob_table = self.parse_with_backpointers(tokens)
        if self.startsymbol in parse_table[(0,n)]:
            return True
        else:
            return False
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        n = len(tokens)
        parse_table = defaultdict(dict)
        prob_table = defaultdict(dict)
        # for terminals
        for i in range(n):
            if (tokens[i],) in self.rhs_to_rules:
                vals = self.rhs_to_rules[(tokens[i],)]
                for v in vals:
                    left, right, prob = v
                    if left in parse_table[(i, i + 1)]:
                        continue
                    else:
                        parse_table[(i, i + 1)][left] = right[0] # for terminal, just string
                        prob_table[(i, i + 1)][left] = log(prob)
            # print(self.parse_table[(i,i+1)])
        for l in range(2, n + 1):
            for i in range(0, n - l + 1):
                j = i + l
                for k in range(i + 1, j):
                    parse_left = parse_table[(i, k)]
                    prob_left = prob_table[(i, k)]
                    parse_right = parse_table[(k, j)]
                    prob_right = prob_table[(k, j)]
                    for k1, v1 in parse_left.items():
                        for k2, v2 in parse_right.items():
                            if (k1, k2) in self.rhs_to_rules:  # k1, k2 are child nodes which have been parsed
                                for v in self.rhs_to_rules[(k1, k2)]:
                                    left, right, prob = v  # left: cur root node
                                    prob_span = log(prob) + prob_left[k1] + prob_right[k2]
                                    if left in parse_table[(i, j)]:
                                        if prob_span > prob_table[(i, j)][left]:
                                            parse_table[(i, j)][left] = ((k1, i, k), (k2, k, j))
                                            prob_table[(i, j)][left] = prob_span
                                    else:
                                        parse_table[(i, j)][left] = ((k1, i, k), (k2, k, j))
                                        prob_table[(i, j)][left] = prob_span
        return parse_table, prob_table


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    children = chart[(i,j)][nt]
    if not isinstance(children, tuple):
        return (nt, children)
    else:
        # print(children)
        c1, c2 = children
        return (nt, get_tree(chart, c1[1], c1[2], c1[0]), get_tree(chart, c2[1], c2[2], c2[0]))

 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        toks2 = ['miami', 'flights','cleveland', 'from', 'to','.']
        print(parser.is_in_language(toks))
        print(parser.is_in_language(toks2))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        print(probs[(0,3)]['NP'])
        print(get_tree(table, 0, len(toks), grammar.startsymbol))

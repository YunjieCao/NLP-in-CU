# -*- coding: utf-8 -*-
# @Time    : 2019/9/25 9:12
# @Author  : Yunjie Cao
# @FileName: trigram_model.py
# @Software: PyCharm
# @Email   ：YunjieCao@hotmail.com


import sys
from collections import defaultdict
from collections import Counter
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Fall 2019
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""


def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if n <= 2:
        sequence = ["START"] + sequence + ["STOP"]
    else:
        sequence = ["START"] * (n-1) + sequence + ["STOP"]
    ret = []
    for i in range(len(sequence)-n+1):
        ret.append(tuple(sequence[i:i+n]))
    return ret


class TrigramModel(object):

    def __init__(self, corpusfile):
        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here
        unigram = []
        bigram = []
        trigram = []
        for sentence in corpus:
            unigram += get_ngrams(sentence, 1)
            bigram += get_ngrams(sentence, 2)
            trigram += get_ngrams(sentence, 3)
        self.unigramcounts = Counter(unigram)
        self.bigramcounts = Counter(bigram)
        self.trigramcounts = Counter(trigram)

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if not hasattr(self, 'trigram_cnt'):
            self.trigram_cnt = defaultdict(int)
            for k, v in self.trigramcounts.items():
                w1, w2, w3 = k
                self.trigram_cnt[(w1,w2)] += v
        if trigram not in self.trigramcounts:
            return 0
        else:
            return (float)(self.trigramcounts[trigram]) / float(self.trigram_cnt[(trigram[0], trigram[1])])

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if not hasattr(self, 'bigram_cnt'):
            self.bigram_cnt = defaultdict(int)
            for k, v in self.bigramcounts.items():
                w1, w2 = k
                self.bigram_cnt[w1] += v
        if bigram not in self.bigramcounts:
            return 0
        else:
            return (float)(self.bigramcounts[bigram]) / (float)(self.bigram_cnt[bigram[0]])

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if not hasattr(self, 'unigram_cnt'):
            self.unigram_cnt = 0
            for k,v in self.unigramcounts.items():
                self.unigram_cnt += v
        if unigram not in self.unigramcounts:
            return 0
        else:
            return (float)(self.unigramcounts[unigram]) / self.unigram_cnt

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.


    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0
        u, v, w = trigram
        p_w_uv = self.raw_trigram_probability(trigram)
        p_w_v = self.raw_bigram_probability(tuple([v,w]))
        p_w = self.raw_unigram_probability(tuple([w]))
        p = lambda1 * p_w_uv + lambda2 * p_w_v + lambda3 * p_w
        return p

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, n=3)
        res = 0.0
        for tri in trigrams:
            res += math.log2(self.smoothed_trigram_probability(tri))
        return res

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        M = 0
        log_sum = 0.0
        for sentence in corpus:
            M += len(sentence)
            log_sum += self.sentence_logprob(sentence)
        res = log_sum / float(M)
        return math.pow(2, -res)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        total += 1
        pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp_1 <= pp_2:
            correct += 1


    for f in os.listdir(testdir2):
        total += 1
        pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        pp_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        if pp_2 <= pp_1:
            correct += 1
    return float(correct) / float(total)


if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])
    # model = TrigramModel("./hw1_data/brown_train.txt")

    # test part1
    # test get_ngrams method
    # res = get_ngrams(["natural", "language", "processing"], 3)
    # print(res)

    # test part2
    # print(model.trigramcounts[('START','START','the')])
    # print(model.bigramcounts[('START','the')])
    # print(model.unigramcounts[('the',)])

    # test part3
    # print(model.raw_unigram_probability(('START',)))
    # print(model.unigram_cnt)
    # print(model.raw_bigram_probability(('START','the')))
    # print(model.bigram_cnt)
    # print(model.raw_trigram_probability(('START', 'START', 'the')))
    # print(model.trigram_cnt)

    # test part4
    # print(model.smoothed_trigram_probability(('START', 'START', 'the')))

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.

    # test part5 & part6
    # Testing perplexity:
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # # dev_corpus = corpus_reader("./hw1_data/brown_test.txt", model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)
    #
    # # Essay scoring experiment:
    # acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    # print(acc)


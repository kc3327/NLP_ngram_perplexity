import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
"""
COMS W4705 - Natural Language Processing - Fall B 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
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
    temp=["START"]*(n-1)+sequence+["STOP"]
    if n==1:
      temp=["START"]+sequence+["STOP"] 
    result=[]
    for i in range(len(temp)-n+1):
        result.append(tuple(temp[i:i+n]))
    return result



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
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        n=0
        ##Your code here
        for ch in corpus:
            n+=1
            tri=get_ngrams(ch, 3)
            bi=get_ngrams(ch, 2)
            uni=get_ngrams(ch, 1)
            for item in tri:
                if item in self.trigramcounts:
                    self.trigramcounts[item]+=1
                else:
                    self.trigramcounts[item]=1
            for item in bi:
                if item in self.bigramcounts:
                    self.bigramcounts[item]+=1
                else:
                    self.bigramcounts[item]=1
            for item in uni:
                if item in self.unigramcounts:
                    self.unigramcounts[item]+=1
                else:
                    self.unigramcounts[item]=1    
        self.sentence_count=n
        temp_dict=self.unigramcounts.copy()
        del temp_dict[("START",)]
        del temp_dict[("STOP",)]
        self.word_count=sum(temp_dict.values())
                

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram not in self.trigramcounts:
            return 0
        if trigram[:2]==('START', 'START'):
            return self.trigramcounts[trigram]/self.sentence_count
        else:
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]
        

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram not in self.bigramcounts:
            return 0
        if bigram[:1]==('START',):
            return self.bigramcounts[bigram]/self.sentence_count
        else:
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram]/self.word_count

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result=[]
        tempkey=('START', 'START')
        n=0
        while n<=20:
            temp=[x for x in self.trigramcounts.keys() if x[:2]==tempkey]
            temp_prob=[self.raw_trigram_probability(x) for x in temp]
            temp_index=np.argmax(np.random.multinomial(100,temp_prob))
            result.append(temp[temp_index][-1])
            if result[-1]=="STOP":
                break
            tempkey=temp[temp_index][1:]
            n+=1      
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        result=lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(trigram[1:])+lambda3*self.raw_unigram_probability(trigram[2:])
        return result
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        tri=get_ngrams(sentence, 3)
        tri_prob=[math.log2(self.smoothed_trigram_probability(x)) for x in tri]
        result=sum(tri_prob)
        
        return result

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total=0
        n=0
        for ch in corpus:
            n+=len(ch)
            total+=self.sentence_logprob(ch)
        return 2**(-total/n)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        for f in os.listdir(testdir1):
            total+=1
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp2:
                correct+=1
    
        for f in os.listdir(testdir2):
            total+=1
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            # .. 
            if pp < pp2:
                correct+=1
            

        return correct/total

if __name__ == "__main__":
    path_train="C:/Users/Aaron/Desktop/Columbia University/NLP/hw1/hw1/hw1_data/brown_train.txt"
    path_test="C:/Users/Aaron/Desktop/Columbia University/NLP/hw1/hw1/hw1_data/brown_test.txt"
    model = TrigramModel(path_train)
    corpus = corpus_reader(path_train, model.lexicon)
    print(model.perplexity(corpus))
    model1 = TrigramModel(path_train)
    corpus1 = corpus_reader(path_test, model.lexicon)
    print(model1.perplexity(corpus1))
    
    # model.count_ngrams()
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    


    # Essay scoring experiment: 
    train_high="C:/Users/Aaron/Desktop/Columbia University/NLP/hw1/hw1/hw1_data/ets_toefl_data/train_high.txt"
    train_low="C:/Users/Aaron/Desktop/Columbia University/NLP/hw1/hw1/hw1_data/ets_toefl_data/train_low.txt"
    test_high="C:/Users/Aaron/Desktop/Columbia University/NLP/hw1/hw1/hw1_data/ets_toefl_data/test_high"
    test_low="C:/Users/Aaron/Desktop/Columbia University/NLP/hw1/hw1/hw1_data/ets_toefl_data/test_low"
    acc = essay_scoring_experiment(train_high, train_low, test_high, test_low)
    print(acc)
    
    model = TrigramModel(train_high)
    corpus = corpus_reader(train_high, model.lexicon)
    print(model.perplexity(corpus))


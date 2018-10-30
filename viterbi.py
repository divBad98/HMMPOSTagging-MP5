# viterbi.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

'''
TODO: implement the baseline algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences, each sentence is a list of (word,tag) pairs. 
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''

import math

def baseline(train, test):
    wordTagCountDict = {} #how many times each word occurs with each tag in training
    predicts = []
    for pair in train:
        if pair in wordTagCountDict:
            wordTagCountDict[pair] += 1 #counts each tuple's frequency
        else:
            wordTagCountDict[pair] = 1 #counts each tuple's frequency
    for word in test:
        oneWordDict = {} #how many times a single word occurs with different tags
        for wt_combo in wordTagCountDict:
            if (wt_combo[0] == word):
                if word in oneWordDict:
                    oneWordDict[wt_combo] += 1
                else:
                    oneWordDict[wt_combo] = 1
        for wt_count_pair in oneWordDict:
            freqTag = ""
            freqTagCount = 0
            if (oneWordDict[wt_count_pair] > freqTagCount):
                freqTagCount = oneWordDict[wt_count_pair]
                freqTag = wt_count_pair[1]
            newPair = (wt_count_pair[0], freqTag)
            predicts.append(newPair)
    return predicts

'''
TODO: implement the Viterbi algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences with tags on the words
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def viterbi(train, test):
    wordTagCountDict = {}
    tagCountDict = {}
    tagTransitionCountDict = {}
    tagTransitionProbDict = {}
    predicts = []
    prevPair = ()
    for pair in train:
        curr_transition = (prevPair, pair)
        if pair in wordTagCountDict:
            wordTagCountDict[pair] += 1 #counts each tuple's frequency
        else:
            wordTagCountDict[pair] = 1 #counts each tuple's frequency
        if pair in tagCountDict:
            tagCountDict[pair[1]] += 1 #counts each tuple's frequency
        else:
            tagCountDict[pair[1]] = 1 #counts each tuple's frequency
        if curr_transition in tagTransitionCountDict:
            tagTransitionCountDict[curr_transition] += 1
        else:
            tagTransitionCountDict[curr_transition] = 1
        for key, value in tagTransitionCountDict:
            tagTransitionProbDict[key] = math.log(value/len(tagTransitionCountDict.keys()), 10)
        prevPair = pair
        predicts.append(prevPair)
    return predicts

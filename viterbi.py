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
    totalTagCountDict = {} #count the total occurrences of each tag and use most seen for the unknown words
    predicts = []

    #Count instances of (word, tag)
    for sentence in train:
        for pair in sentence:
            if pair[1] in totalTagCountDict:
                totalTagCountDict[pair[1]] += 1
            else:
                totalTagCountDict[pair[1]] = 1

            found_tag = False
            if pair[0] in wordTagCountDict:
                for tag_count in wordTagCountDict[pair[0]]:
                    if tag_count[1] == pair[1]:
                        found_tag = True
                        tag_count[0] += 1
                        break
                if not found_tag: 
                    wordTagCountDict[pair[0]].append([1, pair[1]])
            else:
                wordTagCountDict[pair[0]] = [ [1, pair[1]] ]

    #Map each word to its most used POS tag
    for key in wordTagCountDict:
        max_tag_count = None
        for tag_count in wordTagCountDict[key]:
            if max_tag_count == None:
                max_tag_count = tag_count
            elif max_tag_count[0] < tag_count[0]:
                max_tag_count = tag_count
        wordTagCountDict[key] = max_tag_count[1]

    #Find the most used tag for unknowns
    max_tag_count_unk = None
    for tag, count in totalTagCountDict.items():
        if max_tag_count_unk == None or max_tag_count_unk[1] < count:
            max_tag_count_unk = (tag, count)

    max_tag = max_tag_count_unk[0]

    #create dev set
    for i, sentence in enumerate(test):
        predicts.append([])
        for word in sentence:
            if word in wordTagCountDict:
                predicts[i].append( (word, wordTagCountDict[word]) )
            else:
                predicts[i].append( (word, max_tag) )

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


def print_top(to_print, n):
    i = 0
    for key, val in to_print.items():
        print( str(key) + ", " + str(val))
        i += 1
        if(i == n):
            break

def laplace_smooth(data_set, N, alpha, V):
    for key, value in data_set.items():
        data_set[key] = (value + alpha) / (N + alpha * V)

    #return unknown
    return alpha / (N + alpha * V)

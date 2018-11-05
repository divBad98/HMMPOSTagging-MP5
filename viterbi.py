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
import numpy as np

def baseline(train, test):
    #how many times each word occurs with each tag in training
    #keys will be words
    #values will be list of 2-lists of format [tag_count, tag]
    wordTagCountDict = {}     
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
    #wordTagCountDict will now have key->word and value->tag_string
    for key in wordTagCountDict:
        max_tag_count = None
        for tag_count in wordTagCountDict[key]:
            if max_tag_count == None or max_tag_count[0] < tag_count[0]:
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

    #smoothing parameters
    init_smooth = 0.01
    tran_smooth = 0.01
    emis_smooth = 0.01

    #key = tag
    #val = index for tag (mapped in order of appearance)
    tag_ind_dict = {}
    ind_tag_dict = {}
    word_ind_dict = {}

    #Map each tag to an index so we can use arrays to store probabilities
    #also the other way around for tags (only figured out i'd need this at the very end)
    num_tags = 0
    num_words = 0
    for sentence in train:
        for pair in sentence:
            if pair[1] not in tag_ind_dict:
                tag_ind_dict[pair[1]] = num_tags
                ind_tag_dict[num_tags] = pair[1]
                num_tags += 1
            if pair[0] not in word_ind_dict:
                word_ind_dict[pair[0]] = num_words
                num_words += 1
    
    #1D array where each tag is an index according to tag_ind_dict
    #and the value in the index is its probability of occurrence at the beginning of a sentence
    init_prob = np.zeros(shape=(num_tags,))
    init_prob_unk = 0

    #calculate init_prob
    for sentence in train:
        init_prob[ tag_ind_dict[sentence[0][1]] ] += 1

    #smoothing applied because there is a chance a tag never begins a sentence
    laplace_smooth(init_prob, len(train), init_smooth, num_tags)

    #take logs of all probabilities
    one_d_log(init_prob)

    #2D array where each tag is an index according to tag_ind_dict
    #the value at a pair of indexes (i, j) is the probability of tag j following tag i
    trans_prob = np.zeros(shape=(num_tags, num_tags))

    #calculate trans_prob
    count_t = np.zeros(shape=(num_tags,)) #count of times tag t is preceding tag

    for sentence in train:
        for i in range(1, len(sentence)):
            this_tag = sentence[i][1]
            last_tag = sentence[i-1][1]
            
            count_t[ tag_ind_dict[last_tag] ] += 1
            trans_prob[ tag_ind_dict[last_tag] , tag_ind_dict[this_tag] ] += 1

    for tp_ind in range(len(trans_prob)):
        #number of trials = count_t[tp_ind]
        #vocab = num_tags
        laplace_smooth(trans_prob[tp_ind], count_t[tp_ind], tran_smooth, num_tags)

    #take logs of all probabilities
    two_d_log(trans_prob)

    #2d array where (i, j) is the probability of tag at index i (via tag_ind_dict) produces
    #word at index j (via word_ind_dict)
    emis_prob = np.zeros(shape=(num_tags, num_words))

    #1d array for keeping unkown probabilities for each tag
    emis_prob_unk = np.zeros(shape=(num_tags,))

    #initialize emis_prob
    count_tags = np.zeros(shape=(num_tags,)) #count of how many times each tag appears
    for sentence in train:
        for pair in sentence:
            word = pair[0]
            tag = pair[1]

            count_tags[ tag_ind_dict[tag] ] += 1
            emis_prob[ tag_ind_dict[tag] , word_ind_dict[word] ] += 1

    for tag_ind in range(len(emis_prob)):
        #number of times a word appeared with a given tag divided by total number of times tag appeared
        #number of trials = count_tags[tag_ind] ie times a tag appeared
        #vocab = num_words
        emis_prob_unk[tag_ind] = laplace_smooth(emis_prob[tag_ind], count_tags[tag_ind], emis_smooth, num_words)

    #take logs of probabilities
    two_d_log(emis_prob)
    one_d_log(emis_prob_unk)

    #Now that we have the appropriate emission, transition, and initial probabilities we run
    #the viterbi algorithm over each sentence
    predictions = []
    for sentence in test:

        if not sentence:
            predictions.append([])
            continue

        l = len(sentence)
        
        #viterbi trellis
        trellis = np.zeros(shape=(num_tags, l))
        #the previous max
        backpointers = np.zeros(shape=(num_tags, l))

        #initialize trellis
        for tag_ind in range(num_tags):
            if sentence[0] in word_ind_dict:
                trellis[tag_ind, 0] = init_prob[tag_ind] + emis_prob[tag_ind, word_ind_dict[sentence[0]]]
            else:
                trellis[tag_ind, 0] = init_prob[tag_ind] + emis_prob_unk[tag_ind]

        for w_i in range(1, l):
            for t_i in range(num_tags):
                prev_max = -1
                arg_prev_max = -1
                #find the previous max and its corresponding argmax
                for t_j in range(num_tags):
                    prev_val = 0
                    if sentence[w_i] in word_ind_dict:
                        prev_val = trellis[t_j, w_i-1] + trans_prob[t_j, t_i] + emis_prob[t_i, word_ind_dict[sentence[w_i]]]
                    else:
                        prev_val = trellis[t_j, w_i-1] + trans_prob[t_j, t_i] + emis_prob_unk[t_i]

                    if prev_max == -1 or prev_max < prev_val:
                        prev_max = prev_val
                        arg_prev_max = t_j

                trellis[t_i, w_i] = prev_max
                backpointers[t_i, w_i] = arg_prev_max

        #get best path
        max_path = -1
        for t_i in range(num_tags):
            if max_path == -1 or max_path < trellis[t_i, l-1]:
                max_path = t_i

        sentence_prediction = []
        for w_i in range(l-1, -1, -1):
            sentence_prediction.append( (sentence[w_i], ind_tag_dict[max_path]) )
            max_path = int(backpointers[max_path, w_i])

        sentence_prediction.reverse()

        predictions.append(sentence_prediction)

    return predictions


def print_top(to_print, n):
    i = 0
    for key, val in to_print.items():
        print( str(key) + ", " + str(val))
        i += 1
        if(i == n):
            break

#data_set will be 1d numpy array
def laplace_smooth(data_set, N, alpha, V): 

    for i in range(len(data_set)):
        data_set[i] = (data_set[i] + alpha) / (N + alpha * V)

    #return unknown
    return alpha / (N + alpha * V)

def two_d_log(to_log):
    for n in range(len(to_log)):
        one_d_log(to_log[n])

def one_d_log(to_log):
    for n in range(len(to_log)):
        to_log[n] = math.log(to_log[n])

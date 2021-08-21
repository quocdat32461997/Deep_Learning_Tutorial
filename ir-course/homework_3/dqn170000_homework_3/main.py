"""
Author: Dat Quoc Ngo
NET-ID: dqn170000
Date: April 19, 2021
File: main.py - execution in homework 2
"""

# import dependencies
import os
import re
import sys
import nltk
import argparse
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

import index
import utils
import query

def main(args):
    """
    Implement Single-Pass In-Memory Indexing
    """

    # retrieve list of text files
    files = os.listdir(args.path)
    files = [os.path.join(args.path, x) for x in files]

    # get stop words
    with open('./stop_words.txt') as file:
        stop_words = file.read().split()

    # construct index
    print('Index Construction')

    # build index with lemmas
    lem_docs = index.Documents()
    for file, f_idx in zip(files, range(1, len(files) + 1)):
        # read text
        with open(file) as f:
            text = f.read()

        # parse
        doc = index.Document(doc_idx = f_idx)
        doc.parse(text, stop_words, proc_fn = utils.lem)
        lem_docs.update(doc)

    ## merge into big index
    lem_index = index.Index()
    lem_index.build(lem_docs)
    lem_postings, lem_dict = lem_index.invert()

    print('Finish index construction')

    # read queries
    with open('./hw3.queries') as file:
        queries = file.read().split('\n\n')

    # initialize Query object
    querier = query.Query(
            docs = lem_docs,
            index = lem_index,
            postings = lem_postings,
            vocabs = lem_dict)

    # parse query and retrieve documents
    w1_docs = defaultdict()
    w2_docs = defaultdict()
    for q in queries:
        # parse query
        text = querier.parse(q)

        # retrieve documents
        w1_docs[q] = querier.retrieve(freq = text, mode = 'w1')
        w2_docs[q] = querier.retrieve(freq = text, mode = 'w2')

    # display results
    print("\n\nFor weighting 1 function")
    for q, doc in w1_docs.items():
        print("\nQuery {} has vector = {}".format(q, doc[1]))

        # print ranked documents
        docs = doc[0]
        doc_vectors = doc[-1]
        print("\nTop 5 documents is:")
        for x in range(5):
            print('\nDocument {} with rank {} has cosine similarity score = {} and has doc-vector = {}'.format(docs[x][0], x + 1, docs[x][-1], doc_vectors[docs[x][0]]))
            print("Document {} has the headline: {}".format(\
                    docs[x][0], lem_docs._dict[docs[x][0]]._headline))

    print('***********************')

    print("\n\nFor weighting 2 function")
    for q, doc in w2_docs.items():
        print("\nQuery {} has vector = {}".format(q, doc[1]))

        # print ranked documents
        docs = doc[0]
        doc_vectors = doc[-1]
        print("\nTop 5 documents is:")
        for x in range(5):
            print('\nDocument {} with rank {} has cosine similarity score = {} and has doc-vector = {}'.format(docs[x][0], x + 1, docs[x][-1], doc_vectors[docs[x][0]]))
            print("Document {} has the headline: {}".format(\
                    docs[x][0], lem_docs._dict[docs[x][0]]._headline))

if __name__ == '__main__':
    # initialize argument parser
    parser = argparse.ArgumentParser(description = 'Argument parser for homework 1')

    # add arguments
    parser.add_argument('--path', default = './Cranfield', type = str)

    # execute main
    main(parser.parse_args())

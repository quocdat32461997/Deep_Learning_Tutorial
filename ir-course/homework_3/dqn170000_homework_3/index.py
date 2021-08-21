"""
File: index.py
Time: April 21, 2021
Author: Dat Quoc Ngo
"""

# import dependencies
from collections import defaultdict

import utils

class Index(object):
    def __init__(self):
        self._postings = defaultdict(list)
        self._doc_count = defaultdict(int)

    def build(self, docs):
        """
        Args:
            docs : list of Document(s)
        Returns: None
        """

        # loop over possting
        for doc_idx, doc in docs._dict.items():
            # loop over tookens
            for token in doc._dict.keys():
                self._postings[token].append(doc_idx)
                self._doc_count[token] += 1

    def invert(self):
        # sort keys alphabetical ordering
        temp_dict = sorted(self._postings.items(), key = lambda x:x[0])
        temp_doc_count = sorted(self._doc_count.items(), key = lambda x:x[0])
        self._postings = defaultdict(list)
        self._doc_count = defaultdict(int)
        for idx in range(len(temp_dict)):
            # get token
            token = temp_dict[idx][0]

            # save dict
            self._postings[token] = temp_dict[idx][-1]

            # save doc count
            self._doc_count[token] = temp_doc_count[idx][-1]

        # split into dictionary and posting list
        postings = Posting(self._postings.copy())
        dictionary = Dictionary(self._doc_count.copy())

        return postings, dictionary

class Posting(object):
    def __init__(self, postings):
        self._postings = list(postings.values())

    def _gamma(self,inp):
        # to binary
        binary = str(bin(inp))[2:] # discard 0b at front

        # get offset
        offset = binary[1:]

        # get length
        offset_len = len(offset)
        unary = ''.join(['1'] * offset_len) + '0'

        # concat to gamma
        gamma = unary + offset

        return gamma

    def _delta(self, inp):
        # to binary
        binary = str(bin(inp))[2:]

        # get offset
        offset = binary[1:]

        # binary length
        bin_len = len(binary)

        # gamma of binary length
        gamma = self._gamma(bin_len)

        # concat to delta
        delta = gamma + offset

        return delta

    def compress(self, method):
        # initialize compressed postings
        self._compressed_postings = []
        self._postings.copy()

        # get encoding function
        encode_fn = self._gamma if method == 'gamma' else self._delta

        for i in range(len(self._postings)):
            prev = 0
            self._compressed_postings.append([])
            # compute gap and encode into gamma
            for j in range(len(self._postings[i])):
                # get gap
                id = self._postings[i][j]

                # encode
                self._compressed_postings[-1].append(encode_fn(id - prev))

                # update previous gap
                prev = id

class Dictionary(object):
    def __init__(self, doc_count):
        # indexing by 0
        self._doc_count = doc_count
        self._postings = {tok:v for tok, v in zip(doc_count.keys(), range(len(doc_count)))}
        self._dict_str = None
        self._dict = list(doc_count.keys())

    def compress(self, k, method):
        if method == 'blocking':
            pointers = [0]
            idx = 0
            self._dict_str = ''
            while idx < len(self._dict):
                # get terms and update idx
                terms = self._dict[idx : min(idx + k, len(self._dict))]
                idx += 4

                # update dict_str
                self._dict_str += ''.join([str(len(term)) + term for term in terms])
                pointers.append(len(self._dict_str))
            # update term pointer
            self._dict = pointers[:-1]
        elif method == 'front-coding':
            pointers = [0]
            idx = 0
            self._dict_str = ''
            while idx < len(self._dict):
                # get common prefix
                prefix = self._dict[idx][:k]

                # get first term
                if k > len(prefix):
                    self._dict_str += str(len(prefix) - 1) + prefix[:-1] + '*' +  prefix[-1]
                else:
                    self._dict_str += str(k + 1) + prefix + '*' + self._dict[idx][k:]
                # update index
                idx += 1

                # add suffixes
                while idx < len(self._dict) and self._dict[idx].startswith(prefix):
                    suffix = self._dict[idx][len(prefix):]
                    self._dict_str += str(len(suffix)) + '^' + suffix
                    idx += 1

                # update pointer
                pointers.append(len(self._dict_str))
            # update term pointer
            self._dict = pointers[:-1]

class Documents(object):
    """
    Postings - class to map doc index to word frequency
    """
    def __init__(self):
        self._dict = defaultdict()
    def update(self, doc):
        self._dict[doc._doc_idx] = doc

class Document(object):
    """
    Document - an implementation of Poting list in Infomration Retrieval system
    """
    def __init__(self, doc_idx):
        super().__init__() # inherit from Index class
        self._max_freq = 0
        self._doclen = 0
        self._doc_idx = doc_idx
        self._dict = defaultdict(int)
        self._text = None
    
    def parse(self,
            text,
            stop_words,
            proc_fn = None):
        """
        Parse and generate posting list
        Args:
            text : raw text
            stop_words : list
                List of stopwords
            proc_fn : callable function
                text-proc functions such as None, lemmatize, or stem
        Returns: None
        """
        self._headline = utils.get_headline(text)

        # proceess text
        text = utils.process(text)

        # tokenize
        text = utils.tokenize(text)

        # get doclen
        self._doclen = len(text)

        # justify proc_fn
        proc_fn = proc_fn if proc_fn else lambda inp: inp

        # compute word frequency
        for token in text:
            # apply text_proc
            token = proc_fn(token)

            # remove stopwords
            if token in stop_words:
                continue

            # count valid word frequenccy
            self._dict[token] += 1

            # find frequency of the most frequnt term/stem in doc
            self._max_freq = self._dict[token] if self._dict[token] > self._max_freq\
                    else self._max_freq


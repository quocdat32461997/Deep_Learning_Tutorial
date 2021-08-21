# import dependencies
import math
from collections import defaultdict

import utils

class Query(object):
    def __init__(self, docs = None, index = None,
            postings = None, vocabs = None):
        self._docs = docs
        self._index = index
        self._postings = postings
        self._vocabs = vocabs

        # compute average document length
        self.avg_doclen = sum([doc._doclen for doc in self._docs._dict.values()]) / len(self._docs._dict)
        self.collection_size = len(self._docs._dict)

    def _cosine(self, a, b):
        _a = math.sqrt(sum([x**2 for x in a]))
        _b = math.sqrt(sum([x**2 for x in b]))
        ab = sum([x * y for x, y in zip(a, b)])

        return ab / (_a * _b)

    def _w1(self, token, tf, **kwargs):

        df = self._index._doc_count[token] + 1
        tf += 1

        return (0.4 +\
                    0.6 * math.log(tf + 0.5)\
                        / math.log(kwargs['maxtf'] + 1.0))\
                * (math.log(self.collection_size / df)\
                    / math.log(self.collection_size))

    def _w2(self, token, tf, **kwargs):

        df = self._index._doc_count[token] + 1
        tf += 1

        return (0.4 + 0.6\
                    * (tf / (tf + 0.5 + 1.5 * (kwargs['doclen'] / self.avg_doclen))))\
                * (math.log(self.collection_size / df)\
                    / math.log(self.collection_size))

    def parse(self, text):
        """
        Parse and process text
        Args:
            - text : str
                Input text
        Returns:
            - freq : dct
                Dict: token - frequency
        """
        # remove Qx
        text = text.split('\n')
        text = ' '.join(text[1:])

        # process text
        text = utils.process(text)

        # tokenize 
        tokens = utils.tokenize(text)

        # text-processing
        tokens = [utils.lem(token) for token in tokens]

        # count frequency
        freq = defaultdict(int)
        for tok in tokens:
            freq[tok] += 1

        return freq

    def _compute_score(self, doc, q_vector, freq, weight_fn):
        """
        Computer cosine similarity score
        Args:
            - doc : Document object
            - freq : dic
                Dictionary of token and frequency count
            - weight_fn : callable object
                Weighting function
        Returns:
            - score : float
                Cosine similarity score
            - doc_vector : list of float
        """

        doc_vector = []
        for token in freq.keys():
            # doc vector
            doc_vector.append(
                    weight_fn(
                        token = token,
                        tf = doc._dict[token],
                        doclen = doc._doclen,
                        maxtf = doc._max_freq))

        # compute cosine similarity
        score = self._cosine(q_vector, doc_vector)

        return score, doc_vector

    def retrieve(self, freq, mode):
        """
        Args:
            - freq : dict
                Dictionary of token and its frequency
            - mode : str
                Either 'w1' or 'w2'
        Returns:
            - docs : list of tuple
                List of tuple: doc_id and cosine score
            - q_vectors : dict
            - doc_vectors : dict
        """
        assert mode in ['w1', 'w2'], "Mode must be either w1 or w2"

        # retrieve weghting function
        weight_fn = self._w1 if mode == 'w1' else self._w2
        # compute query vector
        text_len = sum(list(freq.values()))
        maxtf = sorted(freq.items(), key = lambda x: x[-1], reverse = True)[0][-1]

        q_vector = [weight_fn(
                token = token,
                tf = count,
                doclen = text_len,
                maxtf = maxtf) for token, count in freq.items()] 
        # compute doc vector and cosine-score
        docs = defaultdict(float)
        doc_vectors = defaultdict(list)
        for doc_idx, doc in self._docs._dict.items():
            docs[doc_idx], doc_vectors[doc_idx] = self._compute_score(doc = doc, q_vector = q_vector, freq = freq, weight_fn = weight_fn)

        # rank docs
        docs = sorted(docs.items(), key = lambda x: x[-1], reverse = True)

        return docs, q_vector, doc_vectors

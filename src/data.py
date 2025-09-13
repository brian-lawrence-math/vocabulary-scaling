import numpy as np
import torch
import torch.nn as nn
import random

# Hack: the START, STOP, PAD tokens are characters that will not be used in text.
START = chr(0)
STOP = chr(1)
PAD = chr(2)

class Data:
    """
    This class loads vocabulary from a file into memory.
    It offers helper methods to select random vocabulary items
    and for encoding and decoding.

    Usage:
        data = Data()
        data.load(num_words=100)

    To sample and tokenize training data:
        es, en = data.random_items(batch_size)
        x, y = data.encode_list(es), data.encode_list(en)
    """
    def __init__(self):
        """
        Initializes the list of characters (tokens)
        but does not load vocabulary from file.
        """
        self.data = None
        self.maxlen = None
        self.get_chars()

    def load(self, fname = "../data/vocab.txt", num_words = None):
        """
        Loads up to num_words vocabulary items from file.
        """
        with open(fname, "r") as file:
            data = []
            for line in file:
                items = line.split("\t")
                if len(items) == 2:
                    data.append((items[0].strip().lower(), items[1].strip().lower()))
        if num_words is not None:
            data = data[:num_words]
        self.data = data
        maxlen0 = max([len(x[0]) for x in data])
        maxlen1 = max([len(x[1]) for x in data])
        self.maxlen = max(maxlen0, maxlen1) + 2
        return data

    def random_item(self):
        n = len(self.data)
        i = random.randint(0, n-1)
        return self.data[i]

    def random_items(self, k):
        n = len(self.data)
        es_list, en_list = [], []
        for _ in range(k):
            i = random.randint(0, n-1)
            es, en = self.data[i]
            es_list.append(es)
            en_list.append(en)
        return es_list, en_list

    def get_chars(self):
        """
        Create dictionaries for converting back and forth between characters and token idxs
        """
        chars = [c for c in "abcdefghijklmnopqrstuvwxyz"] + [START, STOP, PAD]
        
        ctoi = {}
        itoc = {}
        for i, c in enumerate(chars):
            itoc[i] = c
            ctoi[c] = i
        
        self.chars = chars
        self.ctoi = ctoi
        self.itoc = itoc
        return self.chars

    @property
    def n_tokens(self):
        if self.chars is None:
            self.get_chars()
        return len(self.chars)

    def encode(self, s):
        s = START + s + STOP
        if len(s) < self.maxlen:
            s += PAD * (self.maxlen - len(s))
        return torch.tensor([self.ctoi[c] for c in s])

    def decode(self, tens):
        return "".join([self.itoc[i.item()] for i in tens])

    def encode_list(self, ss):
        out = torch.zeros((len(ss), self.maxlen), dtype=torch.int64)
        for i, s in enumerate(ss):
            s = START + s + STOP
            if len(s) < self.maxlen:
                s += PAD * (self.maxlen - len(s))
            out[i] = torch.tensor([self.ctoi[c] for c in s])
        return out

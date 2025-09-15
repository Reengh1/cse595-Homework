from task1 import tokenize, better_tokenize
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from scipy import sparse
import pickle
import matplotlib.pyplot as plt

class task2:
    def __init__(self, documents):
        self.documents = documents
        self.min_word_freq = 250
        self.vocab = {}

    def build_dictionary(self):
        for document in self.documents["generation"]:
            tokens = tokenize(document)
            for token in tokens:
                self.vocab[token] = self.vocab.get(token, 0) + 1

        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1], reverse=True)
        self.vocab.clear()
        id = 0
        for word, count in sorted_vocab:
            if count > self.min_word_freq:
                self.vocab[word] = id
                id += 1
        print(len(self.vocab))
        #Save the address of the vocabulary list.
        with open('/Users/baconjack/Documents/研究生/课程/cse595/vocab_bad.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)
    def creat_sparse_matrix(self):
        row, col, data = [], [], []
        document_id = 0
        for document in self.documents["generation"]:
            tokens = tokenize(document)
            for token in tokens:
                if token in self.vocab:
                    id = self.vocab[token]
                    row.append(document_id)
                    col.append(id)
                    data.append(1)
            document_id += 1

        Matrix = coo_matrix((data, (row, col)), shape=(document_id, len(self.vocab)))
        Matrix = Matrix.tocsr()
        Matrix.data[Matrix.data > 1] = 1
        #Storage Address of Sparse Matrix
        sparse.save_npz('/Users/baconjack/Documents/研究生/课程/cse595/matrix_bad.npz', Matrix)
if __name__ == '__main__':
    #The location of the training files on your own computer
    csv_root = "/Users/baconjack/Documents/研究生/课程/cse595/train.csv"
    documents = pd.read_csv(csv_root)
    documents = documents[["generation"]]
    task2 = task2(documents)
    task2.build_dictionary()
    task2.creat_sparse_matrix()




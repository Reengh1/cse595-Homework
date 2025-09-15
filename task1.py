import numpy as np
import pandas as pd
import re
def tokenize(text):
    tokens = text.split()
    return tokens

def better_tokenize(text):
    text = text.lower()
    text = re.sub(r"[\u200b-\u200d\uFEFF]", "", text)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    #text = re.sub(r'\b(?:[a-zA-Z]\s){1,}[a-zA-Z]\b', lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    tokens = text.split()
    return tokens

if __name__ == '__main__':
    csv_root = "/Users/baconjack/Documents/研究生/课程/cse595/test.csv"
    documents = pd.read_csv(csv_root, header=None)
    for index, row in documents.iterrows():
        tokens = better_tokenize(row[0])
        print(tokens)


    #tokens = better_tokenize(text)
    #print(tokens)

import pickle
from collections import Counter, defaultdict
from glob import glob
import numpy as np
from tqdm import tqdm

from utils import stop_words, flatten, load_docs

doc_collections = glob('../data/data/wiki-pages/*.jsonl')
    
with open("../pickle_jar/idf.pckl", 'rb') as f:
    idf = pickle.load(f)

with open("../pickle_jar/claims_vocab.pckl", 'rb') as f:
    claims_vocab = pickle.load(f)
    
for doc_collection in tqdm(doc_collections):
    tfs = {}
    for document in load_docs([doc_collection]):
        tf = defaultdict(float)
        doc_words = flatten(document.sentences)
        num_words = 0
        for word in doc_words:
            if word in claims_vocab:
                num_words += 1
                tf[word] = tf[word] + 1
        for word in tf:
            tf[word] /= num_words
        tfs[document.id] = (tf, num_words)
    with open(doc_collection + "_tf.pckl", 'wb') as f:
        pickle.dump(tfs, f)

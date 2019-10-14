import pickle
import numpy as np
from glob import glob
from collections import Counter

from utils import load_docs, stop_words, flatten

with open("./pickle_jar/claims_vocab.pckl", 'rb') as f:
    claims_vocab = pickle.load(f)
    
doc_collections = glob('./data/data/wiki-pages/*.jsonl')


frequencies = Counter()

number_of_documents = 0
for document in load_docs(doc_collections):
    number_of_documents += 1
    doc_words = set(flatten(document.sentences)) - stop_words
    frequencies.update(doc_words)
        
idf = {}
for word in frequencies:
    idf[word] = np.log(number_of_documents / frequencies[word])

with open("./pickle_jar/idf.pckl", 'wb') as f:
    pickle.dump(idf, f)
    
from utils import preprocess
from collections import Counter
from glob import glob
from tqdm import tqdm
import json
import pickle

from utils import Document, preprocess

doc_collections = glob('../data/data/wiki-pages/*.jsonl')

#add mapping of document id to document in collection and position in file 

number_of_documents = 0
all_words = Counter()

doc_lookup = {}

for file in tqdm(doc_collections):
    documents = {}
    pickle_file = file + ".pckl"
    with open(file,'r') as f: 
        for i,line in enumerate(f): 
            doc = json.loads(line)
            doc_id = doc['id']
            sentences = doc['lines'].split('\n')
            sentence_lines = []
            for sentence in sentences: 
                try:
                    split_sentence = preprocess(sentence.split('\t')[1])
                    all_words.update(split_sentence)
                    sentence_lines.append(split_sentence)
                except IndexError:
                    pass
                
            document = Document(sentence_lines, doc_id)
            documents[doc_id] = document
            
            
            number_of_documents += 1
            doc_lookup[doc_id] = pickle_file

    
    with open(pickle_file, 'wb') as f:
        pickle.dump(documents, f)

with open("../pickle_jar/all_words.pckl", 'wb') as f:
    pickle.dump(all_words, f)
    
with open("../pickle_jar/doc_lookup.pckl", 'wb') as f:
    pickle.dump(doc_lookup, f)

with open("../pickle_jar/number_of_documents.pckl", 'wb') as f:
    pickle.dump(number_of_documents, f)
    
    

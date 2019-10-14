import operator 
import json
from collections import deque, Counter, defaultdict
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import glob, os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm_notebook, tqdm
import pickle
#from utils import tokenize, preprocess

#matplotlib inline

lemmatizer = WordNetLemmatizer()

#flattening nested lists
flatten = lambda l: [item for sublist in l for item in sublist]

regex_str = [
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z'\-]+)", # words with - and '
    r'-LRB-',
    r'-RRB-',
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)


punctuation = list(string.punctuation)
stop_words = stopwords.words('english') + punctuation
stop_words.append('also')
stop_words.append('rrb') #remove right round bracket
stop_words.append('lrb') #remove left round bracket
stop_words = set(stop_words)

def tokenize(s):
    """ This function tokenizes the input and returns a set wi"""
    return tokens_re.findall(s)

def preprocess(s, lowercase=True, lemmatize = False):
    """ This function tokenizes the input and returns a set of """
    tokens = tokenize(s)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    if lemmatize: 
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens
        

class Sentence:
    def __init__(self, string, string_id, source):
        self.string = string
        self.string_id = string_id
        self.embedding = {}
        self.source = source

class Claim:
    def __init__(self, sentence, evidence, truthfulness, id):
        self.sentence = sentence
        self.evidence = evidence
        self.truthfulness = truthfulness
        self.id = id
    
    def words(self,claim_words):
        words = Counter()
        for term in preprocess(self.sentence, lowercase=True):
            if term in claim_words:
                words[term] += 1
        return words

class Document:
    def __init__(self, sentences, id):
        self.sentences = sentences
        self.embedding = {}
        self.id = id

    def words(self,claim_words):
        words = Counter()
        for term in preprocess(self.sentences, lowercase=True):
            if term in claim_words:
                words[term] += 1
        return words
   
class CoxedFour:
    def __init__(self, limit):
        self.ids = np.array([], dtype="object")
        self.sims = np.array([], dtype="float")
        self.limit = limit

    def sort(self):
        indices = np.argsort(self.sims)
        self.sims = self.sims[indices]
        self.ids = self.ids[indices]

    def push(self,doc_id,sim):
        if len(self.sims) < self.limit: 
          self.ids = np.append(self.ids, doc_id)
          self.sims = np.append(self.sims, sim)
          self.sort()

        elif sim > self.sims[0]:
          self.sims[0] = sim
          self.ids[0] = doc_id
          self.sort()
    
    
    

                
def load_claims(claims_path, claim_ids):
  claims = []
  with open(claims_path, 'r') as f:
      for line in f:
          claim = json.loads(line)
          if claim_ids is None or claim['id'] in claim_ids:
            #tokenise claims
            sentence = claim['claim']
            if claim['verifiable'] == "VERIFIABLE":
              if claim['label'] == "SUPPORTS":
                truthfulness = "true"
              else:
                truthfulness = "false"
            else:
              truthfulness = "perhaps"
            evidence = []
            for page in claim['evidence']:
              for clue in page: 
                _,_,doc_id,sentence_id = clue
                evidence.append(Sentence(None, sentence_id, doc_id))
            id = claim['id']
            claims.append(Claim(preprocess(sentence), evidence, truthfulness, id))
  return claims




def load_test_claims(claims_path, claim_ids):
    claims = []
    with open(claims_path, 'r') as f:
        for line in f:
            claim = json.loads(line)
            if claim_ids is None or claim['id'] in claim_ids:
                #tokenise claims
                sentence = claim['claim']
                id = claim['id']
                claims.append(Claim(preprocess(sentence), None, None, id))
    return claims

    
def load_docs(doc_collections):
    for doc_collection in tqdm(doc_collections):
        with open(doc_collection + ".pckl",'rb') as f:
            documents = pickle.load(f)
            for doc_id in documents:
                yield documents[doc_id]
                
                
                
                
def get_doc(doc_id,doc_collections,doc_lookup):
    doc_collection = doc_lookup[doc_id]
    with open(doc_collection,'rb') as f:
        documents = pickle.load(f)
        return documents[doc_id]
    
                   
    

import operator 
import json
from collections import deque, Counter, defaultdict
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import re, os, math
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm_notebook, tqdm
import pickle

from utils import tokenize, preprocess, Sentence, Claim, Document, load_docs, load_claims, get_doc, stop_words





def load_glove(glove_path,dimensions):
    '''This function loads in the glove embeddings and returns an embedding dictionary with words as keys and embedding as value. It takes as input: 
    - glove_path: the file path where the embeddings are saved 
    - dimensions: the dimensions for the glove embeddings. Choose from 50, 100, 200, 300'''
    glove_vocab = []
    glove_embed = []
    embedding_dict = {}
    glovefile = glove_path + 'glove.6B.{}d.txt'.format(dimensions)
    file = open(glovefile,'r',encoding = 'UTF-8')
    for line in tqdm(file.readlines()): 
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]] # convert to list of float
        embedding_dict[vocab_word]=embed_vector
        glove_embed.append(embed_vector)
    print('Loaded GLOVE')
    file.close()
    return embedding_dict



def extract_relevant_docs(top_five, relevant_claim_ids):
    '''top_twenty as dictionary of ({claim:coxed fours})'''
    relevant_docs = defaultdict(list)
    all_relevant_docs = []
    for claim_id, coxed_four in zip(relevant_claim_ids, top_five): 
        relevant_docs[claim_id] = coxed_four.ids
        all_relevant_docs.extend(coxed_four.ids)
    return relevant_docs, all_relevant_docs

def embed_claim(claims_path, claim_ids, embedding_dict, dimensions): 
    '''This function embeds the claims given, taking out the stop words. For each claim, it adds the embedded word from the embedding dictionary. The function returns a dictionary with keys as the claim id and the values as the embedded claim.'''
    embedding_claim = {}
    for claim in load_claims(claims_path,claim_ids):
        claim_tokens = claim.sentence
        embedding_claim[claim.id] = np.zeros(dimensions)
        for word in claim_tokens: 
            if word in embedding_dict:
                embedding_claim[claim.id] += np.array(embedding_dict[word])
    return embedding_claim


def embed_relevant_docs(doc_collections, doc_ids, embedding_dict,doc_lookup, dimensions): 
    '''This function embeds the specified relevant documents. It returns a dictionary with each key as a document id and the value as a tuple of (sentence id, sentence embedding).''' 
    embedding_docs = defaultdict(dict)
    for doc_id in doc_ids:
        doc = get_doc(doc_id,doc_collections,doc_lookup)
        sentences = doc.sentences
        i = 0
        for sentence in sentences:
            sentence_embedding = np.zeros(dimensions)
            for word in sentence:
                if word in embedding_dict:
                    sentence_embedding += np.array(embedding_dict[word])
            if len(sentence) != 0:
                embedding_docs[doc.id].update({i:sentence_embedding/len(sentence)})
            i += 1          
    return embedding_docs


def extract_evidence(claims_path,claim_ids): 
    claim_evidence = defaultdict(dict)
    for claim in load_claims(claims_path,claim_ids):
        for sentence in claim.evidence:
            claim_evidence[claim.id].update({sentence.source:sentence.string_id})
            
    return claim_evidence

def get_relevant_sentences(relevant_docs,embedding_docs,evidence):
    claim_relevant_sentences = defaultdict(list)
    number_of_sentences = 0 

    for claim in relevant_docs: 
        for doc_id in relevant_docs[claim]:
            for sentence_id in embedding_docs[doc_id]:
                sentence_embedding = embedding_docs[doc_id][sentence_id]

                is_relevant = 0
                if doc_id in evidence[claim].keys():
                    if evidence[claim][doc_id] == sentence_id:
                        is_relevant = 1

                claim_relevant_sentences[claim].append((sentence_embedding,is_relevant))   
                number_of_sentences += 1
   
    return claim_relevant_sentences, number_of_sentences

def get_variables(dimensions,number_of_sentences,claim_relevant_sentences,embedding_claim):

    X = np.zeros((number_of_sentences,2*dimensions))
    Y = np.zeros((number_of_sentences,1))
    i = 0

    for claim in claim_relevant_sentences: 
        for item in claim_relevant_sentences[claim]: 
            X[i] = np.concatenate((embedding_claim[claim],item[0]))
            if np.any(np.isnan(X[i])):
                import pdb; pdb.set_trace()
            Y[i] = item[1]

            i += 1
    
    return X, Y

def embed_and_pickle(claim_ids, claim_path, top_five, dev_train, embedding_dict, dimensions):
    relevant_docs, all_relevant_docs = extract_relevant_docs(top_five, claim_ids)
    print('Relevant files extracted')
  
    embedding_claim = embed_claim(claim_path, claim_ids, embedding_dict, dimensions)
    print('Relevant claims embedded')

    embedding_docs = embed_relevant_docs(doc_collections, all_relevant_docs,embedding_dict, doc_lookup, dimensions)
    print('Relevant docs embedded')


    claim_evidence = extract_evidence(claim_path,claim_ids)
    claim_relevant_sentences, number_of_sentences = get_relevant_sentences(relevant_docs,embedding_docs,claim_evidence)
    X, Y = get_variables(dimensions,number_of_sentences,claim_relevant_sentences,embedding_claim)

    with open(f"../pickle_jar/X_{dev_train}.pckl", "wb") as f:
        pickle.dump(X, f)
    with open(f"../pickle_jar/Y_{dev_train}.pckl", "wb") as f:
        pickle.dump(Y, f)

# Define paths
train_path = '../data/fever-data/train.jsonl'
dev_path = '../data/fever-data/dev.jsonl'
test_path = '../data/fever-data/test.jsonl'
doc_collections = glob('../data/data/wiki-pages/*.jsonl')
glove_path = '../data/glove/'

train_ids = [75397,150448,214861,156709,129629,33078,6744,226034,40190,76253]
dev_ids=[137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708]

# Load in inputs from question 2
with open('../pickle_jar/jelinekq3_top_five_train.pckl','rb') as f: 
    top_five_train = pickle.load(f)
    
with open('../pickle_jar/jelinekq3_top_five_dev.pckl','rb') as f: 
    top_five_dev = pickle.load(f)
    
with open('../pickle_jar/doc_lookup.pckl','rb') as f: 
    doc_lookup = pickle.load(f)
# Load in Glove embeddings
dimensions = 50
embedding_dict = load_glove(glove_path,dimensions)

embed_and_pickle(dev_ids, dev_path, top_five_dev, "dev", embedding_dict, dimensions)
embed_and_pickle(train_ids, train_path, top_five_train, "train", embedding_dict, dimensions)

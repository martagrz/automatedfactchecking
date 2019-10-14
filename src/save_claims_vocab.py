import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_claims, load_test_claims, stop_words

train_path = '../data/fever-data/train.jsonl'
dev_path = '../data/fever-data/dev.jsonl'
test_path = '../data/fever-data/test.jsonl'

paths = [train_path, dev_path]
claims_vocab = set()
for path in paths:
    claims = load_claims(path, None)
    for claim in tqdm(claims):
        claims_vocab.update(set(claim.sentence))

claims = load_test_claims(dev_path, None)
for claim in tqdm(claims):
    claims_vocab.update(set(claim.sentence))

claims_vocab -= stop_words
with open("../pickle_jar/claims_vocab.pckl", 'wb') as f:
    pickle.dump(claims_vocab, f)

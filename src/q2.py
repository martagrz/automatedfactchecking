import pickle
from tqdm import tqdm
from glob import glob
import sys
import numpy as np

from utils import load_claims, stop_words, CoxedFour, flatten, preprocess, load_test_claims

doc_collections = glob('../data/data/wiki-pages/*.jsonl')

with open("../pickle_jar/claims_vocab.pckl", 'rb') as f:
    claims_vocab = pickle.load(f)

with open("../pickle_jar/idf.pckl", 'rb') as f:
    idf = pickle.load(f)

    
def get_norm(tf, idf):
    norm = 0
    for word in tf:
        norm += np.square(tf[word] * idf[word])
    return np.sqrt(norm)

top_cosine_sims = {}

def cosine_sim(claims, doc_collections, idf):
    top_fives = [CoxedFour(5) for claim in claims]
    for doc_collection in tqdm(doc_collections):
        with open(doc_collection + "_tf.pckl", 'rb') as f:
            document_tf = pickle.load(f)
        for doc_id in document_tf:
            title_words = preprocess(doc_id)
            if '-lrb-disambiguation-rrb-' in title_words:
                continue
            tf,_ = document_tf[doc_id]
            norm = get_norm(tf, idf)
            for i, claim in enumerate(claims):
                sim = 0
                claim_words = claim.sentence
                continue_searching = False
                for word in title_words:
                    if word in claim_words:
                        continue_searching = True
                if not continue_searching:
                    continue
                for claim_word in claim_words:
                    try:
                        idf_word = idf[claim_word]
                        sim += idf_word * idf_word * tf[claim_word]
                    except KeyError:
                        pass
                sim /= norm * len(claim_words)
                top_fives[i].push(doc_id, sim)
    return top_fives

if len(sys.argv) == 1:
    # Train
    train_path = '../data/fever-data/train.jsonl'
    relevant_claim_ids = [75397,150448,214861,156709,129629,33078,6744,226034,40190,76253]
    claims = load_claims(train_path, relevant_claim_ids)
    top_five_train = cosine_sim(claims, doc_collections, idf)

    with open("../pickle_jar/top_five_train.pckl", 'wb') as f:
        pickle.dump(top_five_train, f)
    
    predicted_doc_ids = flatten([coxed_four.ids for coxed_four in top_five_train])
    actual_doc_ids = flatten([[evidence.source for evidence in claim.evidence] for claim in claims])
    recall = sum([1 for doc_id in actual_doc_ids if doc_id in predicted_doc_ids]) / len(actual_doc_ids)
    print(recall)


    # Dev    
    dev_path = '../data/fever-data/dev.jsonl'
    relevant_claim_ids = [137334, 111897, 89891, 181634, 219028, 108281, 204361, 54168, 105095, 18708]
    claims = load_claims(dev_path, relevant_claim_ids)
    top_five_dev = cosine_sim(claims, doc_collections, idf)

    with open("../pickle_jar/top_five_dev.pckl", 'wb') as f:
        pickle.dump(top_five_dev, f)

    predicted_doc_ids = flatten([coxed_four.ids for coxed_four in top_five_dev])
    actual_doc_ids = flatten([[evidence.source for evidence in claim.evidence] for claim in claims])
    recall = sum([1 for doc_id in actual_doc_ids if doc_id in predicted_doc_ids]) / len(actual_doc_ids)
    print(recall)

else:
    # Test
    batch_size = 20
    i = int(sys.argv[1])
    test_path = '../data/fever-data/test.jsonl'
    relevant_claim_ids = None
    claims = load_test_claims(test_path, relevant_claim_ids)
    top_five_test = cosine_sim(claims[i * batch_size:(i + 1) * batch_size], doc_collections, idf)

    with open(f"../pickle_jar/top_five_test_{i}.pckl", 'wb') as f:
        pickle.dump(top_five_test, f)

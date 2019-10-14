import pickle

from utils import CoxedFour

with open("../pickle_jar/top_five_train.pckl", 'rb') as f:
    q2_top_five_train = pickle.load(f)

with open("../pickle_jar/jelinekq3_top_five_train.pckl", 'rb') as f:
    q3_top_five_train = pickle.load(f)
    
for top_five_train, output_file in zip([q2_top_five_train, q3_top_five_train], ["tfidf_top_five_documents", "jelinek_top_five_documents"]):
    relevant_claim_ids = [75397,150448,214861,156709,129629,33078,6744,226034,40190,76253]
    with open(f"../output_files/{output_file}.csv", 'w') as f:
        f.write("claim_id\tdoc_id1\tdoc_id2\tdoc_id3\tdoc_id4\tdoc_id5\n")
        for i,claim_id in enumerate(relevant_claim_ids):
            f.write(str(claim_id) + "\t")
            top_five = top_five_train[i]
            [f.write(str(doc_id) + "\t") for doc_id in top_five.ids]
            f.write("\n")

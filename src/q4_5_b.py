import pickle
import numpy as np
import matplotlib.pyplot as plt


def merit_function(Y, h):
    return Y.T @ np.log(h) + (1-Y.T) @ np.log(1-h)
  
def sigmoid(a):
    return 1/(1+np.exp(-a))

def compute_h(X, w):
    a = np.dot(X,w).reshape(-1)

    # h gives you the probability that the sentence is relevant (label = 1)
    h = sigmoid(a)
    return h

np.random.seed(15)
def logistic_regression(X, Y, max_iter=10000):
    #initialise weights
    w = 0.01 * np.random.uniform(size=X.shape[1]).reshape((-1, 1))

    merits = []

    for i in range(max_iter): 
        if i % 100 == 0:
            print(i)
        if np.any(np.isnan(w)):
            import pdb; pdb.set_trace()
        w_old = w


        a = np.dot(X,w).reshape(-1)

        # h gives you the probability that the sentence is relevant (label = 1)
        h = sigmoid(a)

        #calculate merit function

        # Add results of each iteration in while loop to existing merits vector
        merits.append(merit_function(Y,h))

        # Calculate gradient 
        gradient = - (X.T @ Y) + X.T @ sigmoid(X @ w)

        # Calculate Hessian  
        R = np.diag(sigmoid(a)*(1-sigmoid(a)))
        Hessian = X.T @ R @ X 

        # Update weights
        try:
            w = w - np.linalg.solve(Hessian,gradient)
        except:
            break

        # If statement to break while loop if change in weights is too small 
        if np.linalg.norm(w - w_old) < 0.001 * np.linalg.norm(w):
            break
    return merits, w

with open("../pickle_jar/X_train.pckl", "rb") as f:
    X_train = pickle.load(f)
with open("../pickle_jar/Y_train.pckl", "rb") as f:
    Y_train = pickle.load(f)

merits, w = logistic_regression(X_train, Y_train)

fig, ax = plt.subplots()
ax.plot(merits)
plt.show()

"""Q5 performance metrics

We compute the recall, precision and subsequently the F1 metric for the logistic regression trained on the training data, and tested on the development data.
"""

with open("../pickle_jar/X_dev.pckl", "rb") as f:
    X_dev = pickle.load(f)
with open("../pickle_jar/Y_dev.pckl", "rb") as f:
    Y_dev = pickle.load(f)

#obtain predictions for labels on development set 
h_dev = compute_h(X_dev, w)
dev_true_labels = Y_dev
dev_pred_labels = np.zeros((len(Y_dev),1))
i = 0
threshold = 0.5
for element in h_dev: 
  if element < threshold: 
    dev_pred_labels[i] = 0
  else: 
    dev_pred_labels[i] = 1
  i += 1


true_positives = 0 # TP predicted to be 1 and are also 1
false_negatives = 0 # FN predicted to be 0 but correct label is 1
false_positives = 0 # FP predicted to be 1 but correct label is 0

for i in range(len(Y_dev)): 
    if Y_dev[i] == 1: 
        if dev_pred_labels[i] == 1: 
            true_positives += 1
        if dev_pred_labels[i] == 0:
            false_negatives += 1
    if Y_dev[i] == 0:
        if dev_pred_labels[i] == 1:
            false_positives += 1

recall = true_positives / (true_positives + false_negatives)

precision = true_positives / (true_positives + false_positives)

f1_score = 2* (precision*recall)/(precision + recall)

print('Recall:', recall) 
print('Precision:',precision)
print('F1 score:',f1_score)

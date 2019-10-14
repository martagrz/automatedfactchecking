import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("../pickle_jar/all_words.pckl", 'rb') as f:
    all_words = pickle.load(f)

key_values = []
for val in all_words.values():
      key_values.append(val)
key_values = np.array(key_values) 

key_values = -np.sort(-key_values)

n_words = len(key_values)
rank = range(1,n_words+1)

key_values_log = np.log(key_values)
rank_log = np.log(rank)

n = np.size(rank_log)
mean_x, mean_y = np.mean(rank_log), np.mean(key_values_log)
SS_xy = np.sum(key_values_log@rank_log - n*mean_y*mean_x)
SS_xx = np.sum(rank_log@rank_log - n*mean_x*mean_x)
b_1 = SS_xy/SS_xx
b_0 = mean_y - b_1*mean_x

y_pred = b_0 + b_1*rank_log

prob = key_values/np.sum(key_values)
prob = -np.sort(-prob)
rank = np.array(rank)

plt.figure(figsize=(15, 4))
plt.subplot(1,2,1)

plt.plot(rank,key_values)
plt.xlim([0,1000])
plt.xlabel('Rank')
plt.ylabel('Frequency of word')
plt.title('Frequency by Rank')

plt.subplot(1, 2, 2)
plt.plot(rank_log,key_values_log)
plt.plot(rank_log,y_pred)
plt.xlabel('Log Rank')
plt.ylabel('Log Frequency')
plt.title('Log Frequency by Log Rank')

plt.tight_layout()
plt.savefig('./zipf.png')
plt.show()

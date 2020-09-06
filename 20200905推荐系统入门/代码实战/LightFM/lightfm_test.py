"""
lightfm包官方实例：
"""
import numpy as np
from lightfm.datasets import fetch_movielens
movielens = fetch_movielens()
"""
This is a classic small recommender dataset, consisting of around 950 users, 1700 movies, 
and 100,000 ratings. The ratings are on a scale from 1 to 5, but we’ll all treat them as 
implicit positive feedback in this example.
"""

for key, value in movielens.items():
    print(key, type(value), value.shape)

train = movielens['train']
test = movielens['test']
# Each row represents a user, and each column an item. Entries are ratings from 1 to 5.

"""
BPR: Bayesian Personalised Ranking pairwise loss. Maximises the prediction difference
between a positive example and a randomly chosen negative example. Useful when only 
positive interactions are present and optimising ROC AUC is desired.
"""

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM(learning_rate=0.05, loss='bpr')
model.fit(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

"""
- WARP: Weighted Approximate-Rank Pairwise [2]_ loss. Maximises
    the rank of positive examples by repeatedly sampling negative
    examples until rank violating one is found. Useful when only
    positive interactions are present and optimising the top of
    the recommendation list (precision@k) is desired.
"""

model = LightFM(learning_rate=0.05, loss='warp')

model.fit_partial(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
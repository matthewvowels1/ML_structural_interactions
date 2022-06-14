# used pgmpy conda env



import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pc_alg2 import *

# Graph adapted from Peters, Janzing, Scholkopf 2017
N = 5000
C = np.random.randn(N, 1)  # exogenous cause of X
A = 0.8 * np.random.randn(N, 1)  # backdoor adjustment variable (also cause of X)
K = A + 0.1 * np.random.randn(N, 1)  # backdoor adjustment variable (cause of A and Y)
X = C - 2 * A + 0.2 * np.random.randn(N, 1)  # treatment variable
F = 3 * X + 0.8 * np.random.randn(N, 1)  # descendent of treatment variable
D = -2 * X + 0.5 * np.random.randn(N, 1)  # mediator between x and y
G = D + 0.5 * np.random.randn(N, 1)  # descendent of mediator
Y = 2 * K - D + 0.2 * np.random.randn(N, 1)  # outcome variable
H = 0.5 * Y + 0.1 * np.random.randn(N, 1)  # effect of Y

cols = ['C', 'A', 'K', 'X', 'F', 'D', 'G', 'H']
predictors = np.concatenate((C, A, K, X, F, D, G, H), 1)
outcome = Y[:, 0]
df = pd.DataFrame(predictors)
df.columns = cols
df['Y'] = outcome  # <---- dataframe
cols.append('Y')

alpha = 0.01  # false positive rate
knn = 500  # number of nearest neighbours for MI based tests (in general recommend knn /approx (N/10))

# using the modified PC alg originally from pgmpy which takes the additional MI based ci tests
# note that mi based tests will take quite some time to run!
''' ci_test = {'pearsonr', 'gcit', 'chi_square', 'mixed_mi', 'cont_mi', 'independence_match'}
'''

# gcit uses https://github.com/alexisbellot/GCIT/blob/master/Tutorial.ipynb
c = PC_adapted(df)
model = c.estimate(ci_test="gcit", return_type="cpdag", significance_level=alpha, knn=knn)
nx.draw(model, with_labels=True, node_color='white', edge_color='k',
        node_size=500, font_size=25, arrowsize=20, )
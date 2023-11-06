import pdb

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from sklearn.manifold import TSNE
import pickle

embeddings_dict = {}

with open("model/glove.twitter.27B.50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        if len(vector) == 50:
            embeddings_dict[word] = vector
dic_data_cache = "model/glove.pikle"
with open(dic_data_cache, "wb") as d:
    pickle.dump(embeddings_dict, d)

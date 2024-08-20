import pickle
import os
from datasketch import MinHashLSHEnsemble, MinHash
import gc

if __name__ == "__main__":
    index_path = "/data2/csy/web_large"
    # index_path = "LSH Ensemble"
    for filename in os.listdir(index_path):
        if "pickle" in filename and "7" not in filename and "2" not in filename:
            with open(os.path.join(index_path,filename), "rb") as d:
                index_data = pickle.load(d)
                del index_data
                gc.collect()
            print("{} is ok".format(filename))
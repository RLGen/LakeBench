import numpy as np
import random
import pickle
import time
import hnswlib

from munkres import Munkres, make_cost_matrix, DISALLOWED
from numpy.linalg import norm


class HNSWSearcher(object):
    def __init__(self,table_path,index_path,scale):
        tfile = open(table_path,"rb")
        tables = pickle.load(tfile)
        # For scalability experiments: load a percentage of tables
        self.tables = random.sample(tables, int(scale*len(tables)))
        print("From %d total data-lake tables, scale down to %d tables" % (len(tables), len(self.tables)))
        tfile.close()
        self.vec_dim = len(self.tables[1][1][0])     

        index_start_time = time.time()
        self.index = hnswlib.Index(space='cosine', dim=self.vec_dim)
        self.all_columns, self.col_table_ids = self._preprocess_table_hnsw()
        # if not os.path.exists(index_path):
        # build index from scratch
        # self.index.init_index(max_elements=len(self.all_columns), ef_construction=100, M=16)
        self.index.init_index(max_elements=len(self.all_columns), ef_construction=100, M=32)

        self.index.set_ef(10)
        self.index.add_items(self.all_columns)
        # self.index.save_index(index_path)
        print("--- Indexing Time: %s seconds ---" % (time.time() - index_start_time))
        # else:
        #     # load index
        #     self.index.load_index(index_path, max_elements = len(self.all_columns))
    
    def topk(self, enc, query, K, N=5, threshold=0.6):
        # Note: N is the number of columns retrieved from the index
        # query是什么
        query_cols = []
        for col in query[1]:
            query_cols.append(col)
        candidates = self._find_candidates(query_cols, N)
        if enc == 'sato':
            scores = []
            querySherlock = query[1][:, :1187]
            querySato = query[1][0, 1187:]
            for table in candidates:
                sherlock = table[1][:, :1187]
                sato = table[1][0, 1187:]
                sScore = self._verify(querySherlock, sherlock, threshold)
                sherlockScore = (1/min(len(querySherlock), len(sherlock))) * sScore
                satoScore = self._cosine_sim(querySato, sato)
                score = sherlockScore + satoScore
                scores.append((score, table[0]))
        else: # encoder is sherlock
            scores = [(self._verify(query[1], table[1], threshold)[0], self._verify(query[1], table[1], threshold)[1], table[0]) for table in candidates]
        scores.sort(reverse=True)
        scoreLength = len(scores)
        return scores[:K], scoreLength
    
    def _preprocess_table_hnsw(self):
        all_columns = []
        col_table_ids = []
        for idx,table in enumerate(self.tables):
            for col in table[1]:
                all_columns.append(col)
                col_table_ids.append(idx)
        return all_columns, col_table_ids
    
    def _find_candidates(self,query_cols, N):
        table_subs = set()
        labels, _ = self.index.knn_query(query_cols, k=N)
        for result in labels:
            # result: list of subscriptions of column vector
            for idx in result:
                table_subs.add(self.col_table_ids[idx])
        candidates = []
        for tid in table_subs:
            candidates.append(self.tables[tid])
        return candidates
    
    def _cosine_sim(self, vec1, vec2):
        assert vec1.ndim == vec2.ndim
        return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

    def _verify(self, table1, table2, threshold):
            score = 0.0
            nrow = len(table1)
            ncol = len(table2)
            union_column = []
            graph = np.zeros(shape=(nrow,ncol),dtype=float)
            for i in range(nrow):
                for j in range(ncol):
                    sim = self._cosine_sim(table1[i],table2[j])
                    if sim > threshold:
                        graph[i,j] = sim
                        union_column.append((i, j, sim))

            max_graph = make_cost_matrix(graph, lambda cost: (graph.max() - cost) if (cost != DISALLOWED) else DISALLOWED)
            m = Munkres()
            indexes = m.compute(max_graph)
            for row,col in indexes:
                score += graph[row,col]
            return score, union_column
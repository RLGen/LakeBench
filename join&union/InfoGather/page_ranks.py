"""
The code in this file computes the persionalized PageRanks using the Monte Carlo algorithm.

"""
import collections
from functools import partial
from itertools import islice
import pickle

import networkx as nx
import random
from numpy import cumsum, array
from tqdm import tqdm


class IncrementalPersonalizedPageRank3(object):


    def __init__(self, graph,number_of_random_walks, reset_probability,docnum):
        self.graph = graph
        self.number_of_random_walks = number_of_random_walks
        self.reset_probability = reset_probability
        self.docnum = docnum

        self.ppr_vectors = collections.defaultdict(partial(collections.defaultdict, float))

    def initial_random_walks(self):
        #for node in islice(self.graph.nodes(), 5):
        for node in tqdm(range(self.docnum)):
            for _ in range(self.number_of_random_walks):
                self.regular_random_walk(node)
        return

    def regular_random_walk(self, node):
        random_walk = node
        self.ppr_vectors[node][node] += 1
        c = random.uniform(0, 1)
        while c > self.reset_probability:
            if len(list(self.graph.neighbors(random_walk))) > 0:
                current_node = random_walk
                current_neighbors = list(self.graph.neighbors(current_node))
                current_edge_weights = array(
                    [self.graph[current_node][neighbor]['weight'] for neighbor in current_neighbors])
                cumulated_edge_weights = cumsum(current_edge_weights)
                if cumulated_edge_weights[-1] == 0:
                    break
                random_id = list(
                    cumulated_edge_weights < (random.uniform(0, 1) * cumulated_edge_weights[-1])).index(
                    False)
                next_node = current_neighbors[random_id]
                random_walk = next_node
                self.ppr_vectors[node][next_node] += 1
                c = random.uniform(0, 1)
            else:
                break
        return

    def compute_personalized_page_ranks(self):
        #for node in islice(self.graph.nodes(), 5):
        for node in range(self.docnum):
            sumup = sum(self.ppr_vectors[node].values())
            for visited_node in self.ppr_vectors[node].keys():
                try:
                    self.ppr_vectors[node][visited_node] /= sumup
                except ZeroDivisionError:
                    print("List of visit times is empty...")
        return self.ppr_vectors
    

class IncrementalPersonalizedPageRank1(object):

    ppr_vectors = collections.defaultdict(partial(collections.defaultdict, float))

    @classmethod
    def initialize_ppr_vectors(cls,dataset = "webtables"):
        if len(cls.ppr_vectors) == 0:
            if dataset == "webtables":
                with open("/data/lijiajun/infogather/webtables/index/uninon_ppr.pkl","rb") as f:
                    tmp = pickle.load(f)
                cls.ppr_vectors = tmp
                print("webtables ppr load")
            else:
                with open("/data/lijiajun/infogather/opendata/index/ppr_matrix.pkl","rb") as f:
                    tmp = pickle.load(f)
                cls.ppr_vectors = tmp
        else:
            print("yes,,")
            print(len(cls.ppr_vectors))


    def __init__(self, graph,number_of_random_walks, reset_probability,docnum_list):
        self.graph = graph
        self.number_of_random_walks = number_of_random_walks
        self.reset_probability = reset_probability
        self.docnum = len(docnum_list)
        self.docnum_list = docnum_list

        # self.ppr_vectors = collections.defaultdict(partial(collections.defaultdict, float))

    def initial_random_walks(self):
        #for node in islice(self.graph.nodes(), 5):
        for node in self.docnum_list:
            if node in self.ppr_vectors:
                continue
            for _ in range(self.number_of_random_walks):
                self.regular_random_walk(node)
        return

    def regular_random_walk(self, node):
        random_walk = node
        self.ppr_vectors[node][node] += 1
        c = random.uniform(0, 1)
        while c > self.reset_probability:
            if len(list(self.graph.neighbors(random_walk))) > 0:
                current_node = random_walk
                current_neighbors = list(self.graph.neighbors(current_node))
                current_edge_weights = array(
                    [self.graph[current_node][neighbor]['weight'] for neighbor in current_neighbors])
                cumulated_edge_weights = cumsum(current_edge_weights)
                if cumulated_edge_weights[-1] == 0:
                    break
                random_id = list(
                    cumulated_edge_weights < (random.uniform(0, 1) * cumulated_edge_weights[-1])).index(
                    False)
                next_node = current_neighbors[random_id]
                random_walk = next_node
                self.ppr_vectors[node][next_node] += 1
                c = random.uniform(0, 1)
            else:
                break
        return

    def compute_personalized_page_ranks(self):
        for node in self.docnum_list:
            sumup = sum(self.ppr_vectors[node].values())
            for visited_node in self.ppr_vectors[node].keys():
                try:
                    self.ppr_vectors[node][visited_node] /= sumup
                except ZeroDivisionError:
                    print("List of visit times is empty...")
        return self.ppr_vectors

def test_IncrementalPersonalizedPageRank1():
    graph = nx.Graph()
    data=[(0, 1), (1, 2), (1, 3), (2, 4)]
    for ele in data:
        x,y = ele[0],ele[1]
        graph.add_edge(x, y,weight=0.5)
    pr = IncrementalPersonalizedPageRank1(graph, 300, 0.3,[0])
    pr.initial_random_walks()
    page_ranks3 = pr.compute_personalized_page_ranks()
    print(len(page_ranks3))
    # pr = IncrementalPersonalizedPageRank1(graph, 300, 0.3,[3])
    # pr.initial_random_walks()
    # page_ranks3 = pr.compute_personalized_page_ranks()
    # print(len(page_ranks3))
# test_IncrementalPersonalizedPageRank1()


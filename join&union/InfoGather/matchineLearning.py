## 改变判断方式，通过直接判断是否有文件包含 doc1 和doc2 的交集
import itertools
import pickle
import time
from multiprocessing import Pool
import numpy as np

from infogether.myLogger import MyLogger


class MachineLearnSample:
    def __init__(self,index):
        self.KIV = index.KIV
        self.inversed_index = index.inversed_index
        self.doc_number = index.doc_number

    def computer_pair_docs_label_company(self,company):
        doc1, doc2 = company
        term_doc1 = self.KIV[doc1]
        term_doc2 = self.KIV[doc2]
        doc1_doc2_term_union = term_doc1 | term_doc2
        keyterm = ""
        min_docset_len = 2000000000
        for term in doc1_doc2_term_union:
            set_len = len(self.inversed_index[term])
            if set_len < min_docset_len:
                keyterm = term
                min_docset_len = set_len
            if 0 < min_docset_len < 20:
                break

        min_docset = self.inversed_index[keyterm] - {doc1, doc2}
        for docnum in min_docset:
            if (doc1_doc2_term_union <= self.KIV[docnum]):
                return (doc1, doc2, docnum)
        return (doc1, doc2, -1)

    def getlabled(self):
        startime = time.time()
        matix_len = np.array([i for i in range(self.doc_number)])

        with Pool(2) as p:
            data_infos = map(self.computer_pair_docs_label_company, itertools.combinations(matix_len, 2))

        result = [tup for tup in list(data_infos) if tup[2] != -1]
        endtime = time.time()
        time_consum = endtime - startime
        print("计算full labled的时间为：%s" % time_consum)
        print("changdu:%s" % len(result))

        with open("D:\ljj\data\similiarmatrix.pickle", 'wb') as f:
            pickle.dump(result, f)
        print("full labled 已经存储")




import collections
import os

import util


class QuaryOnline():
    def __init__(self,querytablefile,querycolumn,index,ppr_matrix):
        self.querytablefile = querytablefile
        querytablenamepath,querytablename  = os.path.split(self.querytablefile)
        self.querytablename = querytablename
        self.querycolumn = querycolumn
        self.index = index
        self.querytableDict = {}
        self.querytableColumns = []
        #self.file_ppr_list = {}
        self.ppr_matrix = ppr_matrix

        self.querytableDict = util.get_unique_values(self.querytablefile,self.querycolumn)
        self.queryfileColumns = self.querytableDict.keys()

    # get querytable all column ppr list {column : ppr_vecter_dict}
    def getRelationtablesSore(self):
        file_ppr_list = collections.defaultdict(dict)
        for column,column_content in self.querytableDict.items():
            covert_column = util.convertArrtibute(column)
            # get KIA index
            KIA_docs = self.index.KIA[covert_column]

            # get KIV index
            KIV_docs = self.index.getKIV(column_content)

            # get inter between KIA and KIV
            KIA_KIV_docs_set = KIA_docs.intersection(KIV_docs)

            # get score {realtional table : score}
            sore_dict = collections.defaultdict(float)
            for doc in KIA_KIV_docs_set:
                terms = self.index.KIV[doc]
                fenmu = min(len(terms), len(column_content))
                fenzi = len(terms.intersection(column_content))
                sore_dict[doc] = fenzi / fenmu

            # normalnized
            values_sum = sum(sore_dict.values())
            normalized_sore_dict = {key: value / values_sum for key, value in sore_dict.items()}

            # sum all the ralational tables PRR
            total_ppr = collections.defaultdict(float)
            for doc, s in normalized_sore_dict.items():
                ppr_doc = self.ppr_matrix[doc]
                for key, value in ppr_doc.items():
                    total_ppr[key] += value * s

            # get all the colums ppr
            file_ppr_list[column] = total_ppr

            # order dic
            for key,value_dic in file_ppr_list.items():
                file_ppr_list[key]  = { k : v for k ,v in sorted(value_dic.items(),key= lambda x:x[1],reverse=True)}

        #return file_ppr_list
        self.file_ppr_list = file_ppr_list

    def getTransformatTableInfo(self,docNo):
        for k ,v in self.file_ppr_list.items():
            self.file_ppr_list[k] = [docNo[key] for key in v.keys()]
        return self.file_ppr_list

    """ 

    def getUnionTable(self):
        #  for every table to do BinaryGraphMaxPowerMatch
        BGM = BinaryGraphMatch(file_ppr_list, index.docNo, index.doc_number)
        BGM.getMatchTable()
        print(BGM.finaldic)
    """
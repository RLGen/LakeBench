import collections

import networkx as nx

class BinaryGraphMatch:
    def __init__(self,columnsPPRdict,docNo,docNum):
        self.columns = columnsPPRdict.keys()
        self.columnsPPRdict = columnsPPRdict
        self.docNo = docNo
        self.docNum = docNum
        self.finaldic = collections.defaultdict(list)

    def getMatchTable(self):
        doc = 0
        while doc < self.docNum:
            # scan every doc
            flag = False
            # judge relational table
            for column in self.columns:
                if doc in self.columnsPPRdict[column]:
                    flag = True
                    break
            if flag:
                matchdocName = self.docNo[doc]
                # get all the colums of matchName
                matchtablename = matchdocName.split("#")[0]
                matchtabl_colums = [doc]
                doc = doc + 1
                while matchtablename in self.docNo[doc]:
                    matchtabl_colums.append(doc)
                    doc = doc + 1
                # get binaryGraph the right nodes in matchtabl_colums
                self.biarymatch(matchtabl_colums,matchtablename)
            else:
                doc = doc + 1

    def biarymatch(self, rightNodes,matchTableName):
        # 创建二分图
        G = nx.Graph()
        # 添加左侧节点（代理）
        G.add_nodes_from(self.columns)
        # 添加右侧节点（接收者）
        G.add_nodes_from(rightNodes)
        # 添加边和权重
        for leftnode in self.columns:
            for rightnode in rightNodes:
                if rightnode in self.columnsPPRdict[leftnode]:
                    G.add_edge(leftnode, rightnode, weight=self.columnsPPRdict[leftnode][rightnode])

        # 使用最大权匹配算法
        matching = nx.max_weight_matching(G, weight='weight', maxcardinality=True)
        max_weight = 0
        tableColumsMatchList = []
        print("最大权匹配结果：")
        for agent, receiver in matching:
            weight = G[agent][receiver]['weight']
            matchColumn = self.docNo[receiver].split("#")[1]
            print(f" {agent} 和 {matchColumn} 匹配，权重为 {weight}")
            tableColumsMatchList.append(f"{agent} 和 {matchColumn} 匹配，权重为 {weight}")
            max_weight += weight
        print("最大权重：", max_weight)
        key = "%s#%s"%(matchTableName,max_weight)

        self.finaldic.update({key:tableColumsMatchList})


# 创建二分图
G = nx.Graph()

# 添加左侧节点（代理）
G.add_nodes_from(['Agent1', 'Agent2', 'Agent3'])

# 添加右侧节点（接收者）
G.add_nodes_from(['Receiver1', 'Receiver2', 'Receiver3','test'])

# 添加边和权重
G.add_edge('Agent1', 'Receiver1', weight=3)
G.add_edge('Agent1', 'Receiver3', weight=2)
G.add_edge('Agent2', 'Receiver2', weight=2)
G.add_edge('Agent2', 'Receiver4', weight=1)
G.add_edge('Agent3', 'Receiver1', weight=1)
G.add_edge('Agent3', 'Receiver3', weight=4)
G.add_edge('Agent3', 'Receiver4', weight=2)

# 使用最大权匹配算法
matching = nx.max_weight_matching(G, weight='weight', maxcardinality=True)

max_weight = 0

print("最大权匹配结果：")
for agent, receiver in matching:
    weight = G[agent][receiver]['weight']
    print(f"节点 {agent} 和节点 {receiver} 匹配，权重为 {weight}")
    max_weight += weight

print("最大权重：", max_weight)


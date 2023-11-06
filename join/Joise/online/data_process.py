import pickle

import numpy as np
import pandas as pd
import os
import time
import hashlib

# directory_path = "data/benchmark/"
cpath = "../DataFilter/csvdata"
#cpath = "data/benchmark"
# cpath = "../DataFilter/querytables"

#返回query的setID
def readQueryID(setMap,table_name,column_name):

    for key, value in setMap.items():
        if value['table_name']==table_name and value['column_name'] == column_name:
            queryID=key-1
        else:queryID=-1
    return queryID


def processQuery(query,rawDict):
    tids=[]
    gids=[]
    freqs=[]
    for token in query:
        if token in rawDict.keys():
            tid=rawDict[token][0]
            gid=rawDict[token][1]
            freq=rawDict[token][2]
            tids.append(tid)
            gids.append(gid)
            freqs.append(freq)
    return tids,gids,freqs




class ListEntry:
    def __init__(self, ID, MatchPosition, Size):
        self.ID = ID
        self.Size = Size
        self.MatchPosition = MatchPosition

def getEntries(token,inIndex):
    entries=[]
    token=str(token)
    # token=int(token)
    # print("token",str(type(token)))
    PLs=inIndex[token]
    # print("PLs长度",len(PLs))
    # PLs = inIndex.get(token, 'no')
    for PL in PLs:
        # print(PL)
        entry=ListEntry(PL[0],PL[1],PL[2])
        entries.append(entry)
    return entries

def setTokensSuffix(integerSet,setID,startPo):
    setID=str(setID)
    # setID=int(setID)
    tokens=integerSet[setID][startPo:]
    return tokens





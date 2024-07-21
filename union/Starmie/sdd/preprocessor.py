import pandas as pd
import math
import collections
import string
from pandas.api.types import infer_dtype

def computeTfIdf(tableDf):
    """ Compute tfIdf of each column independently
        Called by _tokenize() method in dataset.py
    Args:
        table (DataFrame): input table
    Return: tfIdf dict containing tfIdf scores for all columns
    """
    # tfidf that considers each column (document) independently
    def computeTf(wordDict, doc):
        # input doc is a list
        tfDict = {}
        docCount = len(doc)
        for word, count in wordDict.items():
            tfDict[word] = count / float(docCount)
        return tfDict
    def computeIdf(docList):
        idfDict = {}
        N = len(docList)
        idfDict = dict.fromkeys(docList.keys(), 0)
        for word, val in docList.items():
            if val > 0:
                idfDict[word] += 1
        for word, val in idfDict.items():
            idfDict[word] = math.log10(N / float(val))
        return idfDict
    idf = {}
    for column in tableDf.columns:
        colVals = [val for entity in tableDf[column] for val in str(entity).split(' ')]
        wordSet = set(colVals)
        wordDict = dict.fromkeys(wordSet, 0)
        for val in colVals:
            wordDict[str(val)] += 1
        idf.update(computeIdf(wordDict))
    return idf


def pmiSample(val_counts, table, colIdxs, currIdx, max_tokens):
    """ Compute PMI of pairs of columns (one of which is the topic column)
        Used in pmi sampling
    Args:
        val_counts (dict): stores the count of each (topic value, property value), topic value, and property value
        table (DataFrame): input table
        colIdxs (list): list of column indexes using column headers
        currIdx: current column index
        max_tokens: maximum tokens from pretrain arguments
    Return: list of sampled tokens for this column
    """
    tokens = []
    valPairs = []
    topicCol = table[colIdxs[0]]
    PMIs = {}
    for i in range(topicCol.shape[0]):
        topicVal = topicCol[i]
        propVal = table.at[i, currIdx]
        if (topicVal, propVal) in val_counts and topicVal in val_counts and propVal in val_counts:
            pair_pmi = val_counts[(topicVal, propVal)] / (val_counts[topicVal] * val_counts[propVal])
            PMIs[(topicVal, propVal)] = pair_pmi
    PMIs = {k: v for k, v in sorted(PMIs.items(), key=lambda item: item[1], reverse=True)}
    if colIdxs.index(currIdx) == 0:
        valPairs = [k[0] for k in PMIs.keys()]
    else:
        valPairs = [k[1] for k in PMIs.keys()]
    for val in valPairs:
        for v in str(val).split(' '):
            if v not in tokens:
                tokens.append(v)
        if len(tokens) >= max_tokens:
            break
    return tokens


def constantSample(colVals, max_tokens):
    '''Helper for preprocess() for constant sampling: take nth elements of column
        For sampling method 'constant'
    Args:
        colVals: list of tokens in each entity in the column
        max_tokens: maximum tokens specified in pretrain argument
    Return:
        list of tokens, such that list is of length max_tokens
    '''
    step = math.ceil(len(colVals) / max_tokens)
    tokens = colVals[::step]
    while len(tokens) > max_tokens:
        step += 1
        tokens = colVals[::step]
    return tokens

def frequentSample(colVals, max_tokens):
    '''Frequent sampling: Take most frequently occuring tokens
        For sampling method 'frequent'
    Args:
        colVals: list of tokens in each entity in the column
        max_tokens: maximum tokens specified in pretrain argument
    Return list of tokens
    '''
    tokens, tokenFreq = [], {}
    tokenFreq = collections.Counter(colVals)
    tokenFreq = {k: v for k, v in sorted(tokenFreq.items(), key=lambda item: item[1], reverse=True)[:max_tokens]}
    for t in colVals:
        if t in tokenFreq and t not in tokens:
            tokens.append(t)
    return tokens

def tfidfSample(column, tfidfDict, method, max_tokens):
    '''TFIDF sampling: Take tokens with highest idf scores
        For sampling methods 'tfidf_token', 'tfidf_entity'
    Args:
        column (pandas Series): current column from input table DataFrame
        tfidfDict (dict): dict with tfidf scores for each column, created in _tokenize()
        method (str): sampling method ('tfidf_token', 'tfidf_entity')
        max_tokens: maximum tokens specified in pretrain argument
    Return list of tokens
    '''
    tokens, tokenList, tokenFreq = [], [], {}
    if method == "tfidf_token":
        # token level
        for colVal in column.unique():
            for val in str(colVal).split(' '):
                idf = tfidfDict[val]
                tokenFreq[val] = idf
                tokenList.append(val)
        tokenFreq = {k: v for k, v in sorted(tokenFreq.items(), key=lambda item: item[1], reverse=True)[:max_tokens]}
        for t in tokenList:
            if t in tokenFreq and t not in tokens:
                tokens.append(t)
                
    elif method == "tfidf_entity":
        # entity level
        for colVal in column.unique():
            valIdfs = []
            for val in str(colVal).split(' '):
                valIdfs.append(tfidfDict[val])
            idf = sum(valIdfs)/len(valIdfs)
            tokenFreq[colVal] = idf
            tokenList.append(colVal)
        tokenFreq = {k: v for k, v in sorted(tokenFreq.items(), key=lambda item: item[1], reverse=True)}
        valCount, N = 0, 0
        for entity in tokenFreq:
            valCount += len(str(entity).split(' '))
            if valCount < max_tokens: N += 1
        tokenFreq = {k: tokenFreq[k] for k in list(tokenFreq)[:N]}
        for t in tokenList:
            if t in tokenFreq and t not in tokens:
                tokens += str(t).split(' ')
    return tokens


def tfidfRowSample(table, tfidfDict, max_tokens):
    '''TFIDF sampling: Take rows with tokens that have highest idf scores
        For sampling method 'tfidf_row'
        Called in _tokenize() method in dataset.py
    Args:
        table (DataFrame): input table
        tfidfDict (dict): dict with tfidf scores for each column, created in _tokenize()
        max_tokens: maximum tokens specified in pretrain argument
    Return table with sampled rows using tfidf
    '''
    tokenFreq = {}
    sortedRowInds = []
    for row in table.itertuples():
        index = row.Index
        valIdfs = []
        rowVals = [val for entity in list(row[1:]) for val in str(entity).split(' ')]
        for val in rowVals:
            valIdfs.append(tfidfDict[val])
        idf = sum(valIdfs)/len(valIdfs)
        tokenFreq[index] = idf
        tokenFreq = {k: v for k, v in sorted(tokenFreq.items(), key=lambda item: item[1], reverse=True)}
        sortedRowInds = list(tokenFreq.keys())[:max_tokens]
    table = table.reindex(sortedRowInds)
    return table

def preprocess(column: pd.Series, tfidfDict: dict, max_tokens: int, method: str): 
    '''Preprocess a column into list of max_tokens number of tokens 
       Possible methods = "head", "alphaHead", "random", "constant", "frequent", "tfidf_token", "tfidf_entity", "tfidf_row"
    Args:
        column (pandas Series): current column from input table DataFrame
        tfidfDict (dict): dict with tfidf scores for each column, created in _tokenize()
        max_tokens: maximum tokens specified in pretrain argument
        method (str): sampling method from list of possible methods
    Returns list of sampled tokens
    '''
    tokens = []
    colVals = [val for entity in column for val in str(entity).split(' ')]
    if method == "head" or method == "tfidf_row":
        for val in colVals:
            if val not in tokens:
                tokens.append(val)
                if len(tokens) >= max_tokens:
                    break
    elif method == "alphaHead":
        if 'mixed' in infer_dtype(column):
            column = column.astype(str)
        sortedCol = column.sort_values()
        sortedColVals = [str(val).lower() for entity in sortedCol for val in str(entity).split(' ')]
        for val in sortedColVals:
            if val not in tokens:
                tokens.append(val)
                if len(tokens) >= max_tokens:
                    break
    elif method == "random":
        tokens = pd.Series(colVals).sample(min(len(colVals),max_tokens)).sort_index().tolist()
    elif method == "constant":
        tokens = constantSample(colVals, max_tokens)
    elif method == "frequent":
        tokens = frequentSample(colVals, max_tokens) 
    elif "tfidf" in method and method != "tfidf_row":
        tokens = tfidfSample(column, tfidfDict, method, max_tokens)
    return tokens
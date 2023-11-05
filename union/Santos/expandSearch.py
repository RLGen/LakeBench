# -*- coding: utf-8 -*-


import string
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
import re

def puncRemove(stringList):
    output = []
    for item in stringList:
        st = ""
        for sym in item:
            if sym not in string.punctuation:
                st = st + sym
            else:
                st = st + " "
        output.append(st)
    return output

def removeStringPunctuations(string):
    string =  re.sub(r'[^\w]', ' ', string)
    string = string.replace("nbsp",'') #whiteSpace
    string =" ".join(string.split())
    return (string)

def checkIfNullString(string):
    nullList = ['nan','-','unknown','other (unknown)','null','na', "", " "]
    if str(string).lower() not in nullList:
        return 1
    else:
        return 0

def preprocessListValues(valueList):
    valueList = [x.lower() for x in valueList if checkIfNullString(x) !=0]
    valueList = [re.sub(r'[^\w]', ' ', string) for string in valueList]
    valueList = [x.replace('nbsp','') for x in valueList ] #remove html whitespace
    valueList = [" ".join(x.split()) for x in valueList]
    valueList = [x for x in valueList if x != np.nan]
    valueList = list(set(valueList))
    return valueList

def cleanBracesContents(stringList):
    output = []
    for item in stringList:
     output.append(re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", item))
    return output
def cleanBracesinString(string):
     return (re.sub("([\(\{\[]).*?([\)\]\}])", "\g<1>\g<2>", string))
     
#print(cleanBracesinString("My name is {khan}"))
def extractNouns(stringList):
    sentence = ' '.join(item for item in stringList)
    nouns = [token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('N')]
    return nouns

def expandQuery(stringList):
    stringList = [item for item in stringList if type(item) == str]
    sentence = ' '.join(item for item in stringList)
    stringList = cleanBracesContents(stringList)
    puncCleanedList = preprocessListValues(stringList)
    nounList = extractNouns(puncCleanedList)
    expandedQueryList = [words for segments in nounList for words in segments.split()]
    # handle phrase queries
    removeNouns = []
    for entity in puncCleanedList:
        entityList = entity.split(" ")
        if entityList.count('') > 0 and entityList.count('') <= 2:
            entityList.remove('')
        index = 0
        while index <= len(entityList) - 1:
            word = entityList[index]
            if word in nounList:
                if index + 1 < len(entityList):
                    nextWord = entityList[index + 1]
                    if entityList[index + 1] in nounList:
                        removeNouns.append(word)
                        removeNouns.append(entityList[index + 1])
                        expandedQueryList.append(word + " " + entityList[index + 1])
                        index += 1
            index += 1
                    
    finalNouns = [noun for noun in expandedQueryList if noun not in removeNouns]
    stopWordsRemovedList= [word for word in finalNouns if word.lower() not in stopwords.words('english')]
    return (list(set(stopWordsRemovedList)))


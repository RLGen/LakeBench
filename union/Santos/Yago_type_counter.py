"""
This Python file is used to count appearances for each Yago type.
The input is yago-wd-full-types.nt, yago-wd-simple-types.nt and yago-wd-facts.nt.
"""

dict = {}

full_types = open("../yago/yago_original/yago-wd-full-types.nt", "r", encoding='utf-8')
for line in full_types.readlines():
    triple = line.split()
    predicate = triple[1]
    if predicate.find("type") != -1:
        t = triple[2][1:-1].split('/')[-1]
        if dict.get(t):
            dict[t] = dict.get(t) + 1
        else:
            dict[t] = 1
full_types.close()

simple_types = open("../yago/yago_original/yago-wd-simple-types.nt", "r", encoding='utf-8')
for line in simple_types.readlines():
    triple = line.split()
    predicate = triple[1]
    if predicate.find("type") != -1:
        t = triple[2][1:-1].split('/')[-1]
        if dict.get(t):
            dict[t] = dict.get(t) + 1
        else:
            dict[t] = 1
simple_types.close()

facts = open("../yago/yago_original/yago-wd-facts.nt", "r", encoding='utf-8')
for line in facts.readlines():
    triple = line.split()
    predicate = triple[1]
    if predicate.find("type") != -1:
        t = triple[2][1:-1].split('/')[-1]
        if dict.get(t):
            dict[t] = dict.get(t) + 1
        else:
            dict[t] = 1
facts.close()

typeDict = open("yago-type", 'w', encoding='utf-8')
for i in dict.items():
    typeDict.write(i[0] + ', '+ str(i[1]) + '\n')
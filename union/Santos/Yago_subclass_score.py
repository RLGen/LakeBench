"""
This Python file is used to generate scores for each set of subclass relationship.
"""
import math
import pickle

relationship = open("yago-subclass", "r", encoding='utf-8')
types = open("yago-type", "r", encoding='utf-8')

type_count = {}
type_subclasses = {}

for line in types.readlines():
    split = line.split(', ')
    type_count[split[0]] = split[1].rstrip('\n')

for line in relationship.readlines():
    split = line.split(',[')
    type_subclasses[split[0]] = split[1].rstrip('\n').replace(']','').replace('\'','').replace(' ','').split(',')

type_score = {}
type_level = {}

for i in type_count:
    if type_count[i] == '1':
        score = 1.0
    else:
        score = min(1.0 , 1 / math.log10(float(type_count[i])))
    type_score[i] = score

def count_level(type, level):
    if type_subclasses.get(type):
        level += 1
        for c in type_subclasses[type]:
            type_level[c] = level
            count_level(c, level)

type_level["Thing"] = 0
count_level("Thing", 0)

depth = 0
for i in type_level:
    depth = max(depth, type_level[i])

deleted = []
for i in type_score:
    if i not in type_level:
        deleted.append(i)
for i in deleted:
    del type_score[i]

while depth > 0:
    depth -= 1
    for i in type_score:
        if type_level[i] == depth and type_subclasses.get(i):
            for c in type_subclasses[i]:
                if type_score.get(c):
                    type_score[i] = min(type_score[i], type_score[c])

score = open("../yago/yago_pickle/yago-type-score.pickle", 'wb')
final_dict = {}
for i in type_score:
	final_dict[i.lower()] = (type_score[i], type_level[i])

pickle.dump(final_dict,score, protocol=pickle.HIGHEST_PROTOCOL)
score.close()
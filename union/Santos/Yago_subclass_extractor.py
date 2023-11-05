"""
This Python file is used to extract the subclass relationships between Yago types. 
The input is yago-wd-class.nt and yago-wd-schema.nt.
"""
classes = open("../yago/yago_original/yago-wd-class.nt", "r", encoding='utf-8')
schema = open("../yago/yago_original/yago-wd-schema.nt", "r", encoding='utf-8')

dict = {}

for line in classes.readlines():
    triple = line.split()
    predicate = triple[1]
    if predicate.find("subClassOf") != -1:
        subject = triple[0][1:-1].split('/')[-1]
        object = triple[2][1:-1].split('/')[-1]
        if subject != object:
            if dict.get(object):
                if subject not in dict.get(object):
                    list = dict.get(object).append(subject)
            else:
                list = [subject,]
                dict[object] = list

for line in schema.readlines():
    triple = line.split()
    predicate = triple[1]
    if predicate.find("subClassOf") != -1:
        subject = triple[0][1:-1].split('/')[-1]
        object = triple[2][1:-1].split('/')[-1]
        if subject != object:
            if dict.get(object):
                if subject not in dict.get(object):
                    list = dict.get(object).append(subject)
            else:
                list = [subject,]
                dict[object] = list

relationship = open("yago-subclass", 'w', encoding='utf-8')
for i in dict.items():
    relationship.write(i[0] + ','+ str(i[1]) + '\n')
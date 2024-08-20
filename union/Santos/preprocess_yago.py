# -*- coding: utf-8 -*-

#YAGO preprocessing file.
#The label file, type file, subclass file and the fact file are converted into python dictionaries.

import pickle
import re
from collections import defaultdict

def getParentAndChild(line):
    triple = line.replace(",","temp_placeholder")
    triple = triple.lower().split('\t',maxsplit = 4)
    child = triple[0].replace("temp_placeholder",",")
    child = child.strip()
    tempList = child.rsplit("/",1)
    child = tempList[-1][:-1]
    parent = triple[2].replace("temp_placeholder",",")
    parent = parent.strip()
    tempList = parent.rsplit("/",1)
    parent = tempList[-1][:-1]
    return (parent, child)

def findSubClasses(path):
    i = 1
    j = 1
    dictionary = {}
    with open(path,'r', encoding = "UTF-8") as infile:
        for line in infile:
            line = line.lower()
            #sample line: <http://yago-knowledge.org/resource/fishing_tackle>	<http://www.w3.org/2000/01/rdf-schema#subclassof>	<http://schema.org/thing>	.
            j+=1
            if "rdf-schema#subclassof" in line and "http://schema.org/thing" in line:
                parent, child = getParentAndChild(line)
                dictionary[child] =[]
                i+=1
                if i % 1000 == 0:
                    print("Written lines:",i)
                    print("Processed lines:",j)
    print("Class file written in class dictionary.")
    return dictionary

def findAllEdges(path):
    i = 1
    j = 1
    edges = []
    with open(path,'r', encoding = "UTF-8") as infile:
        for line in infile:
            line = line.lower()
            #sample line: <http://yago-knowledge.org/resource/fishing_tackle>	<http://www.w3.org/2000/01/rdf-schema#subclassof>	<http://schema.org/thing>	.
            j+=1
            if "rdf-schema#subclassof" in line and "http://schema.org/thing" not in line:
                parent, child = getParentAndChild(line)
                edges.append((parent,child))
                i+=1
                if i % 1000 == 0:
                    print("Written lines:",i)
                    print("Processed lines:",j)
    print("Class file written in class dictionary.")
    return edges

#using bfs to create the inheritance dictionary:
class Hierarchy:
    # class Constructor
    def __init__(self):
        #define dictionary of list. This will store the hierarchy
        self.hierarchy = defaultdict(list)
 
    def addEdge(self,u,v):
        self.hierarchy[u].append(v)
 
    # Function to compute BFS
    def BreadthFirstSearch(self, top_type):
        visited = {}
        for key in self.hierarchy:
            visited[key] = False
        
        # BFS waiting buffer
        buffer = []
        buffer.append(top_type)
        visited[top_type] = True
        while buffer:
            top_type = buffer.pop(0)
            for i in self.hierarchy[top_type]:
                if visited.get(i, "None") == False:
                    buffer.append(i)
                    visited[i] = True
                else:
                    visited[i] = True
        return visited


SCHEMA_FILE_PATH = r"../yago/yago_original/yago-wd-schema.nt" #contains labels and some types too

#ENTITY DICTIONARY
LABEL_INPUT_FILE_PATH = r"../yago/yago_original/yago-wd-labels.nt" 
LABEL_OUTPUT_FILE_PATH = r"../yago/yago_pickle/yago-wd-labels_dict.pickle" 

#TYPE DICTIONARY
FULL_TYPES_INPUT_FILE_PATH = r"../yago/yago_original/yago-wd-full-types.nt"
SIMPLE_TYPES_INPUT_FILE_PATH = r"../yago/yago_original/yago-wd-simple-types.nt"
TYPES_OUTPUT_FILE_PATH = r"../yago/yago_pickle/yago-wd-full-types_dict.pickle" 

#INHERITANCE DICTIONARY
CLASS_INPUT_FILE_PATH = r"../yago/yago_original/yago-wd-class.nt"
CLASS_OUTPUT_FILE_PATH = r"../yago/yago_pickle/yago-wd-class_dict.pickle" 

#RELATIONSHIP DICTIONARY
FACTS_INPUT_FILE_PATH = r"../yago/yago_original/yago-wd-facts.nt"
FACTS_OUTPUT_FILE_PATH = r"../yago/yago_pickle/yago-wd-facts_dict.pickle" 

#entity dictionary
#1. Convert label file  into pickle file. The dictionary stores label as key and entity URI as value
i = 1
j = 1
dictionary = {}
earlierEntity = []
with open(SCHEMA_FILE_PATH,'r', encoding = "UTF-8") as infile:
    for line in infile:
        #sample line: 
            #<http://schema.org/performinggroup>	<http://www.w3.org/2000/01/rdf-schema#label>	"performing group"	.
        #expected output: 
            #{performing group: [performinggroup]} i.e. label : [entity]
        line = line.lower()
        if "rdf-schema#label" in line:
            triple = line.replace(",","temp_placeholder")
            triple = triple.lower().split('\t',maxsplit = 4)
            label = triple[2].replace("temp_placeholder", "")
            label = label.replace("@en", "")
            label = label.replace('"', '')
            label =  re.sub(r'[^\w]', ' ', label)
            label = " ".join(label.split())
            entity = triple[0].replace("temp_placeholder",",")
            entity = entity.strip()
            entityList = entity.rsplit("/",1)
            entity = entityList[-1][:-1]
            if label in dictionary:
                earlierEntity = dictionary[label]
                newEntity = earlierEntity + [entity]
                dictionary[label] = list(set(newEntity))
            else:
                dictionary[label] = [entity]
print("Schema label file written in entity dictionary.")
#go through each lines of input file and write them to output file after conversion
with open(LABEL_INPUT_FILE_PATH,'r', encoding = "UTF-8") as infile:
    for line in infile:
        #Sample line: <http://yago-knowledge.org/resource/ecclesiastes>	<http://schema.org/alternatename>	"book of ecclesiastes"@en	.
        #expected output: {book of ecclesiastes: [ecclesiastes]} i.e. label : [entity]
        line = line.lower()
        j+=1
        if "@en" in line and "rdf-schema#comment" not in line:
            triple = line.replace(",","temp_placeholder")
            triple = triple.lower().split('\t',maxsplit = 4)
            label = triple[2].replace("temp_placeholder", "")
            label = label.replace("@en", "")
            label = label.replace('"', '')
            label =  re.sub(r'[^\w]', ' ', label)
            label = " ".join(label.split())
            entity = triple[0].replace("temp_placeholder",",")
            entity = entity.strip()
            entityList = entity.rsplit("/",1)
            entity = entityList[-1][:-1]
            if label in dictionary:
                earlierEntity = dictionary[label]
                newEntity = earlierEntity + [entity]
                dictionary[label] = list(set(newEntity))
            else:
                dictionary[label] = [entity]
            i+=1
            if i % 1000000 == 0:
                print("Written lines:",i)
                print("Processed lines:",j)
print("Label file written in entity dictionary.")
file_pointer=open(LABEL_OUTPUT_FILE_PATH, 'wb')
pickle.dump(dictionary,file_pointer, protocol=pickle.HIGHEST_PROTOCOL)

#type dictionary
#2. convert type file into dictionary. This stores the entity URIs as key and types as values.
dictionary = {}
earlierType = []
with open(SCHEMA_FILE_PATH,'r', encoding = "UTF-8") as infile:
    for line in infile:
        line = line.lower()
        #sample line:
            # <http://schema.org/performinggroup>	<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>	<http://www.w3.org/2002/07/owl#class>	.
        #expected output: 
            # {performinggroup: [performinggroup]}  i.e. entity: type
        if "22-rdf-syntax-ns#type" in line and "2002/07/owl#class" in line:
            triple = line.replace(",","temp_placeholder")
            triple = triple.lower().split('\t',maxsplit = 4)
            entity = triple[0].replace("temp_placeholder",",")
            entity = entity.strip()
            entityList = entity.rsplit("/",1)
            entity = entityList[-1][:-1]
            if entity[0] != "_":
                if entity in dictionary:
                    earlierType = dictionary[entity]
                    newEntity = earlierType + [entity]
                    dictionary[entity] = list(set(newEntity))
                else:
                    dictionary[entity] = [entity]          
print("Schema type file written in type dictionary.")
i = 1
j = 1
with open(FULL_TYPES_INPUT_FILE_PATH,'r', encoding = "UTF-8") as infile:
    for line in infile:
        line = line.lower()
        #sample line: <http://yago-knowledge.org/resource/harald_ringstorff>	<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>	<http://yago-knowledge.org/resource/human>	.
        #expected output: {harald_ringstorff: [human, person]} i.e. entity: [type]
        j+=1
        if "type" in line:
            triple = line.replace(",","temp_placeholder")
            triple = triple.lower().split('\t',maxsplit = 4)
            entity = triple[0].replace("temp_placeholder",",")
            typeName = triple[2].replace("temp_placeholder",",")
            typeName = typeName.strip()
            #instead of full type url, extract its name only.
            typeList = typeName.rsplit("/",1)
            typeName = typeList[-1][:-1]
            entity = entity.strip()
            entityList = entity.rsplit("/",1)
            entity = entityList[-1][:-1]
            if entity in dictionary:
                earlierType = dictionary[entity]
                newEntity = earlierType + [typeName]
                dictionary[entity] = list(set(newEntity))
            else:
                dictionary[entity] = [typeName]
            i+=1
            if i % 1000000 == 0:
                print("Written lines:",i)
                print("Processed lines:",j)
print("Full types file written in type dictionary.")
i = 1
j = 1       
with open(SIMPLE_TYPES_INPUT_FILE_PATH,'r', encoding = "UTF-8") as infile:
    for line in infile:
        line = line.lower()
        #sample line: <http://yago-knowledge.org/resource/pulicat_lake_bird_sanctuary>	<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>	<http://schema.org/animalshelter>	.
        #expected output: {pulicat_lake_bird_sanctuary: [animalshelter]} i.e. entity: [type]
        j+=1
        if "type" in line:
            triple = line.replace(",","temp_placeholder")
            triple = triple.lower().split('\t',maxsplit = 4)
            entity = triple[0].replace("temp_placeholder",",")
            typeName = triple[2].replace("temp_placeholder",",")
            typeName = typeName.strip()
            #instead of full type url, extract its name only.
            typeList = typeName.rsplit("/",1) 
            typeName = typeList[-1][:-1]
            entity = entity.strip()
            entityList = entity.rsplit("/",1)
            entity = entityList[-1][:-1]
            if entity in dictionary:
                earlierType = dictionary[entity]
                newType = []
                newType = earlierType + [typeName]
                dictionary[entity] = list(set(newType))
            else:
                dictionary[entity] = [typeName]
            i+=1
            if i % 1000000 == 0:
                print("Written lines:",i)
                print("Processed lines:",j)
print("Simple types file written in type dictionary.")
file_pointer=open(TYPES_OUTPUT_FILE_PATH, 'wb')
pickle.dump(dictionary,file_pointer, protocol=pickle.HIGHEST_PROTOCOL)

#inheritance dictionary
#3. convert class file into dictionary. 
#This stores the types having thing as top level type as key and a list
# of its children as value.
dictionaryClass = findSubClasses(CLASS_INPUT_FILE_PATH)
topTypes = set()
for key in dictionaryClass:
    topTypes.add(key)

allSchemaEdges = findAllEdges(SCHEMA_FILE_PATH)
allClassEdges = findAllEdges(CLASS_INPUT_FILE_PATH)
allEdges = list(set(allSchemaEdges + allClassEdges))

hrchy = Hierarchy()
for edge in allEdges:
    hrchy.addEdge(edge[0],edge[1])
dictionary = {}
for key in dictionaryClass:
    dictionaryClass[key] = hrchy.BreadthFirstSearch(key)
for edge in dictionaryClass:
    visited_stats =hrchy.BreadthFirstSearch(edge)
    for result in visited_stats:
        if visited_stats[result]:
            if edge in dictionary:
                dictionary[edge].append(result)
            else:
                dictionary[edge] = [result]
 
file_pointer=open(CLASS_OUTPUT_FILE_PATH, 'wb')
pickle.dump(dictionary, file_pointer, protocol=pickle.HIGHEST_PROTOCOL)



#relationship dictionary
#4. convert fact file into fact dictionary. This stores the relation semantics.
i = 1
dictionary = {}
earlierRelation = []
with open(FACTS_INPUT_FILE_PATH,'r', encoding = "UTF-8") as infile:
    for line in infile:
        line = line.lower()
        triple = line.replace(",","temp_placeholder")
        triple = triple.lower().split('\t',maxsplit = 4)
        
        subject = triple[0].replace("temp_placeholder",",")
        subject = subject.strip()
        subjectList = subject.rsplit("/",1)
        subject = subjectList[-1][:-1]
        
        obj = triple[2].replace("temp_placeholder",",")
        obj = obj.strip()
        objList = obj.rsplit("/",1)
        obj = objList[-1][:-1]
        
        key = subject + "__"+obj
        
        predicate = triple[1].replace("temp_placeholder",",")
        predicate = predicate.strip()
        predList = predicate.rsplit("/",1)
        
        value = predList[-1][:-1]
      
        if key in dictionary:
            earlierRelation = dictionary[key]
            newRelation = []
            newRelation = earlierRelation + [value]
            dictionary[key] = list(set(newRelation))
        else:
            dictionary[key] = [value]
        i+=1
        if i % 1000000 == 0:
            print("Written lines:",i)
print("Fact file written in relation dictionary.")
file_pointer=open(FACTS_OUTPUT_FILE_PATH, 'wb')
pickle.dump(dictionary,file_pointer, protocol=pickle.HIGHEST_PROTOCOL)
file_pointer.close()
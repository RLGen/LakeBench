# -*- coding: utf-8 -*-


#from tkinter import E
import numpy as np
import pandas as pd
import csv
import glob
import time
import os
import os.path
from pathlib import Path
import sys
import generalFunctions as genFunc
import expandSearch as expand
#This function takes a column and determines whether it is text or numeric column
#This has been done using a well-known information retrieval technique
#Check each cell to see if it is text. Then if enough number of cells are text, the column is considered as a text column.

def getAccuracyResult(sortedTableList, expected_tables, k):
    i = 0
    tp = 0
    fp = 0
    for key in sortedTableList:
        data_lake_table = key[0].split(os.sep)[-1]
        if data_lake_table in expected_tables:
            tp = tp+1
        else:
            fp = fp+1
        i = i +1
        if (i >=k):
            break
    fp = k - tp
    return tp, fp

def getMatchingTables(item, weight, parameter):   
    returnList = []
    #print(merge)
    for each in item:
        temp = each
        tableName = temp[0]
        #tableName = genFunc.cleanTableName(temp[0])
        tableScore = temp[-1] * weight *parameter
        returnList.append((tableName, tableScore))
    return returnList


#compute synthesized CS for the query table
def computeSynthColumnSemantics(input_table, synth_type_kb):
    #synthInvertedIndex  = {}
    all_column_semantics = {}
    col_id = 0
    for (columnName, columnData) in input_table.iteritems():
        sem = {}
        #creating the lookup table for data lake tables
        if genFunc.getColumnType(input_table[columnName].tolist()) == 1:
            #print(table_name)
            input_table[columnName] = input_table[columnName].map(str)
            valueList = genFunc.preprocessListValues(input_table[columnName].unique())                   
            hit_found = 0                
            #find bag of semantics for each column
            for value in valueList:
                if value in synth_type_kb:
                    item = synth_type_kb[value]
                    hit_found += 1
                    for temp in item:
                        semName = temp[0]
                        semScore = temp[-1] 
                        if semName in sem:
                            sem[semName] +=semScore
                        else:
                            sem[semName] = semScore
            for every in sem:
                sem[every] = sem[every]/hit_found

            if str(col_id) in all_column_semantics:
                print("red flag!!!")
            else:
                all_column_semantics[str(col_id)] = sem
        col_id += 1
    return all_column_semantics

#compute synthesized relationship semantics for the query table
def computeSynthRelation(inputTable, subjectIndex, synthKB):
    label = "r"
    relationSemantics = {}
    synth_triple_dict = {}
    total_cols = inputTable.shape[1]
    subject_semantics = set()
    for i in range(0, total_cols -1):    
        if genFunc.getColumnType(inputTable.iloc[:,i].tolist()) == 1: #the subject in rdf triple should be a text column
            for j in range(i+1, total_cols):
                if genFunc.getColumnType(inputTable.iloc[:,j].tolist()) == 1: #the subject in rdf triple should be a text column
                    mergeRelSem = {}
                    dataFrameTemp = inputTable.iloc[:,[i,j]]
                    dataFrameTemp = (dataFrameTemp.drop_duplicates()).dropna()
                    projectedRowsNum = dataFrameTemp.shape[0]
                    
                    #find relation semantics for each value pairs of subjectIndex and j
                    for k in range(0,projectedRowsNum):
                        #extract subject and object
                        sub = genFunc.preprocessString(str(dataFrameTemp.iloc[k][0]).lower())
                        obj = genFunc.preprocessString(str(dataFrameTemp.iloc[k][1]).lower())
                        subNull = genFunc.checkIfNullString(sub)
                        objNull = genFunc.checkIfNullString(obj)
                        if subNull != 0 and objNull != 0:
                            item = []
                            value = sub+"__"+obj
                            if value in synthKB:
                                item = synthKB[value]
                                
                            else:
                                value = obj+"__"+sub
                                if value in synthKB:
                                    item = synthKB[value]
                                   
                            if len(item) > 0 :
                                for each in item:
                                    temp = each
                                    if temp[-1] >0:
                                        semName = temp[0]
                                        semScore = temp[-1] 
                                        if semName in mergeRelSem:
                                            mergeRelSem[semName] +=semScore/projectedRowsNum
                                        else:
                                            mergeRelSem[semName] = semScore/projectedRowsNum
                    

                    triple_list = []
                    for sem in mergeRelSem:
                        triple_list.append((sem, mergeRelSem[sem]))
                            
                    synth_triple_dict[str(i) + "-" + str(j)] = triple_list
                    if int(subjectIndex) == i or int(subjectIndex) == j:
                        for sem in mergeRelSem:
                            subject_semantics.add(sem)
                    for sem in mergeRelSem:
                        if sem in relationSemantics:
                            currentList = relationSemantics[sem]
                            currentScore = currentList[0]
                            newScore = currentScore + mergeRelSem[sem]
                            relationSemantics[sem] = [newScore, label]
                        else:
                            relationSemantics[sem] = [mergeRelSem[sem], label]
    return relationSemantics, synth_triple_dict, subject_semantics
  
#compute KB RS for the query table
def computeRelationSemantics(input_table, tab_id, LABEL_DICT, FACT_DICT):
    relation_bag_of_words = []
    total_cols = input_table.shape[1]
    #total_rows = input_table.shape[0]
    relation_dependencies = []
    entities_finding_relation = {}
    relation_dictionary = {}
    total_hits = {}
    #compute relation semantics
    for i in range(0, total_cols-1):
            #print("i=",i)
        if genFunc.getColumnType(input_table.iloc[:, i].tolist()) == 1: 
            #the subject in rdf triple should be a text column
            for j in range(i+1, total_cols):
                semantic_dict_forward = {}
                semantic_dict_backward = {}
                #print("j=",j)
                column_pairs = input_table.iloc[:, [i, j]]
                column_pairs = (column_pairs.drop_duplicates()).dropna()
                unique_rows_in_pair = column_pairs.shape[0]
                total_kb_forward_hits = 0
                total_kb_backward_hits = 0
                #print(column_pairs)
                #assign relation semantic to each value pair of i and j
                for k in range(0, unique_rows_in_pair):
                    #print(k)
                    #extract subject and object
                    found_relation = 0
                    subject_value = genFunc.preprocessString(str(column_pairs.iloc[k][0]).lower())
                    object_value = genFunc.preprocessString(str(column_pairs.iloc[k][1]).lower())
                    is_sub_null = genFunc.checkIfNullString(subject_value)
                    is_obj_null = genFunc.checkIfNullString(object_value)
                    if is_sub_null != 0:
                        sub_entities = LABEL_DICT.get(subject_value, "None")
                        if sub_entities != "None":
                            if is_obj_null != 0:    
                                obj_entities = LABEL_DICT.get(object_value, "None")
                                if obj_entities != "None":
                                    #As both are not null, search for relation semantics
                                    for sub_entity in sub_entities:
                                        for obj_entity in obj_entities:
                                            #preparing key to search in the fact file
                                            entity_forward = sub_entity + "__" + obj_entity
                                            entity_backward = obj_entity + "__" + sub_entity
                                            relation_forward = FACT_DICT.get(entity_forward, "None")
                                            relation_backward = FACT_DICT.get(entity_backward, "None")
                                            if relation_forward != "None":
                                                found_relation = 1
                                                total_kb_forward_hits += 1
                                                #keep track of the entity finding relation. We will use this to speed up the column semantics search
                                                key = str(i)+"_"+subject_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {sub_entity}
                                                else:
                                                    entities_finding_relation[key].add(sub_entity)
                                                key  = str(j) + "_" + object_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {obj_entity}
                                                else:
                                                    entities_finding_relation[key].add(obj_entity)
                                                for s in relation_forward:
                                                    if s in semantic_dict_forward:
                                                        semantic_dict_forward[s] += 1 #relation semantics in forward direction
                                                    else:
                                                        semantic_dict_forward[s] = 1
                                            if relation_backward != "None":
                                                found_relation = 1
                                                total_kb_backward_hits += 1
                                                #keep track of the entity finding relation. We will use this for column semantics search
                                                key = str(i)+"_"+subject_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {sub_entity}
                                                else:
                                                    entities_finding_relation[key].add(sub_entity)
                                                
                                                key  = str(j)+"_"+object_value
                                                if key not in entities_finding_relation:
                                                    entities_finding_relation[key] = {obj_entity}
                                                else:
                                                    entities_finding_relation[key].add(obj_entity)
                                                
                                                for s in relation_backward:
                                                    if s in semantic_dict_backward:
                                                        semantic_dict_backward[s] += 1 #relation semantics in reverse direction
                                                    else:
                                                        semantic_dict_backward[s] = 1
                if len(semantic_dict_forward) > 0:
                    relation_bag_of_words.append((max(semantic_dict_forward, key=semantic_dict_forward.get)+"-r", str(i)+"_"+str(j), max(semantic_dict_forward.values())/ total_kb_forward_hits))
                    relation_dependencies.append(str(i)+"-"+str(j))
                    relation_dictionary[str(i)+"-"+str(j)] = [(max(semantic_dict_forward, key=semantic_dict_forward.get), max(semantic_dict_forward.values())/ total_kb_forward_hits)]
                    total_hits[str(tab_id) +"-"+ str(i)+"-"+str(j)] = (total_kb_forward_hits, unique_rows_in_pair)
                if len(semantic_dict_backward) >0:
                    relation_bag_of_words.append((max(semantic_dict_backward, key=semantic_dict_backward.get)+"-r", str(j)+"_"+str(i), max(semantic_dict_backward.values())/ total_kb_backward_hits))
                    relation_dependencies.append(str(j)+"-"+str(i))
                    relation_dictionary[str(j)+"-"+str(i)] = [(max(semantic_dict_backward, key=semantic_dict_backward.get), max(semantic_dict_backward.values())/ total_kb_backward_hits)]
                    total_hits[str(tab_id) +"-"+ str(j)+"-"+str(i)] = (total_kb_backward_hits, unique_rows_in_pair)
    return relation_bag_of_words, entities_finding_relation, relation_dependencies, relation_dictionary, total_hits

#yago column semantics for query table
def computeColumnSemantics(input_table, subject_index, LABEL_DICT, TYPE_DICT, CLASS_DICT, RELATION_DICT, scoring_function):
    col_id = 0
    not_found_in_yago = []
    column_bag_of_words = []
    column_dictionary = {}
    subject_semantics = ""
    for (columnName, columnData) in input_table.iteritems():
        if genFunc.getColumnType(input_table[columnName].tolist()) == 1: #check column Type
            input_table[columnName] = input_table[columnName].map(str)                
            #get unique values in the column and preprocess them.
            value_list = genFunc.preprocessListValues(input_table[columnName].unique())
            #search values in KB 
            all_found_types = {}
            total_kb_hits = 0
            if str(subject_index) == str(col_id):
              label = "sc"
            else:
              label = "c"
            for value in value_list:
                current_entities = set()
                current_types = set()
                current_entities = RELATION_DICT.get(str(col_id) + "_"+ value, "None")
                #print(current_entities)
                if current_entities != "None":
                    total_kb_hits += 1
                    for entity in current_entities:
                        if entity in TYPE_DICT:
                            temp_type = TYPE_DICT[entity]
                            for entity_type in temp_type:
                                current_types.add(entity_type)
                    for each_type in current_types:
                        if each_type in all_found_types:
                            all_found_types[each_type] +=1
                        else:
                            all_found_types[each_type] = 1
                            
                else: 
                    current_entities = LABEL_DICT.get(value, "None")
                    if current_entities != "None": #found in KB
                        total_kb_hits += 1
                        for entity in current_entities:
                            if entity in TYPE_DICT:
                                temp_type = TYPE_DICT[entity]
                                for entity_type in temp_type:
                                    current_types.add(entity_type)
                        for each_type in current_types:
                            if each_type in all_found_types:
                                all_found_types[each_type] +=1
                            else:
                                all_found_types[each_type] = 1
                    else:
                        not_found_in_yago.append(value)    
                
            #find the top-level type with highest count.
            all_top_types = [v for v in sorted(all_found_types.items(), key=lambda kv: (-kv[1], kv[0])) if v[0] in CLASS_DICT]
            if all_top_types:
              selected_top_type = all_top_types[0][0]
              top_type_count = all_top_types[0][1]
              if label == "sc":
                subject_semantics = selected_top_type
              if scoring_function == 1:
                column_bag_of_words.append((selected_top_type+"-"+label, str(col_id), all_found_types[selected_top_type]/total_kb_hits))
                column_dictionary[int(col_id)] = [(selected_top_type, all_found_types[selected_top_type]/total_kb_hits)]               
              else:
                children_of_top_types = CLASS_DICT[selected_top_type]
                #add children of top types to the bag of word
                for each in all_found_types:
                    if each in children_of_top_types and (all_found_types[each] / top_type_count) >= 0:
                        column_bag_of_words.append((each+"-c", str(col_id), all_found_types[each]/total_kb_hits))
                        if int(col_id) not in column_dictionary:
                            column_dictionary[int(col_id)] = [(each, all_found_types[each]/total_kb_hits)]
                        else:
                            column_dictionary[int(col_id)].append((each, all_found_types[each]/total_kb_hits))
        col_id += 1
        
    return column_bag_of_words, column_dictionary, subject_semantics


                                   
if __name__ == "__main__":
    which_benchmark = 5
    while which_benchmark != 1 and which_benchmark != 2 and which_benchmark != 3 and which_benchmark != 4 and which_benchmark != 5:
      print("Press 1 for TUS Benchmark, 2 for SANTOS (small) benchmark, 3 for SANTOS (large) benchmark, 4 for webtable.")
      which_benchmark = int(input())
    map_k = 20
    if which_benchmark == 1:
        current_benchmark = "tus"
        map_k = 60
    elif which_benchmark == 2:
        current_benchmark = "santos"
        map_k = 10  
    elif which_benchmark == 3:
        current_benchmark = "real_tables"
    elif which_benchmark == 4:
        current_benchmark = 'webtable'
    else:
        current_benchmark = 'opendata'
    print("Press 1 for kb, 2 for synth and 3 for full.")
    which_mode = int(input())
    mode_list = ['kb', 'synth', 'full']
    #which_mode = 3 #1是kb， 2是synth， 3是两个都用的
    current_mode = mode_list[which_mode - 1]
    benchmarkLoadStart = time.time()
    print("Press k.")
    k = int(input())
    #load the YAGO KB from dictionary files and data lake table names
    #edit the path below according to the location of the pickle files. 
    YAGO_PATH = r"yago/yago_pickle/" 
    #edit the line below if the dlt are at different locations
    if which_benchmark == 4:
        QUERY_TABLE_PATH = '/data_ssd/webtable/large/small_query/'
    elif which_benchmark == 5:
        QUERY_TABLE_PATH = '/data_ssd/opendata/small/query/'
    else:
        QUERY_TABLE_PATH = r"benchmark/" + current_benchmark + "_benchmark/query/"

    LABEL_FILE_PATH = YAGO_PATH + "yago-wd-labels_dict.pickle"
    TYPE_FILE_PATH = YAGO_PATH + "yago-wd-full-types_dict.pickle" 
    CLASS_FILE_PATH = YAGO_PATH + "yago-wd-class_dict.pickle"
    FACT_FILE_PATH = YAGO_PATH + "yago-wd-facts_dict.pickle"
    
    FD_FILE_PATH = r"groundtruth/" + current_benchmark + "_FD_filedict.pickle"
    GROUND_TRUTH_PATH = r"groundtruth/" + current_benchmark + "UnionBenchmark.pickle"
    SUBJECT_COL_PATH = r"groundtruth/" + current_benchmark + "IntentColumnBenchmark.pickle"
    YAGO_MAIN_INVERTED_INDEX_PATH = r"hashmap/" + current_benchmark + "_and_query_main_yago_index.pickle"
    YAGO_MAIN_RELATION_INDEX_PATH = r"hashmap/" + current_benchmark + "_and_query_main_relation_index.pickle"
    YAGO_MAIN_PICKLE_TRIPLE_INDEX_PATH = r"hashmap/" + current_benchmark + "_and_query_main_triple_index.pickle"
    pickle_extension = "pbz2"
    
    SYNTH_TYPE_LOOKUP_PATH = r"hashmap/" + current_benchmark + "_synth_type_lookup."+pickle_extension
    SYNTH_RELATION_LOOKUP_PATH = r"hashmap/" + current_benchmark + "_synth_relation_lookup."+pickle_extension
    
    SYNTH_TYPE_KB_PATH =   r"hashmap/" + current_benchmark + "_synth_type_kb."+pickle_extension
    SYNTH_RELATION_KB_PATH =   r"hashmap/" + current_benchmark + "_synth_relation_kb."+pickle_extension
    
    SYNTH_TYPE_INVERTED_INDEX_PATH = r"hashmap/" + current_benchmark + "_synth_type_inverted_index."+pickle_extension
    SYNTH_RELATION_INVERTED_INDEX_PATH = r"hashmap/" + current_benchmark + "_synth_relation_inverted_index."+pickle_extension
    
    MAP_PATH = r"stats/" + current_benchmark + "_benchmark_result_by_santos_"+current_mode+".csv"
    FINAL_RESULT_PICKLE_PATH = r"stats/" + current_benchmark + "_benchmark_result_by_santos_"+current_mode+".pickle"
    
    TRUE_RESULTS_PATH = r"stats/" + current_benchmark +str(k) + "_benchmark_true_result_by_santos_"+current_mode+".csv"
    FALSE_RESULTS_PATH = r"stats/" + current_benchmark +str(k) + "_benchmark_false_result_by_santos_"+current_mode+".csv"
    QUERY_TIME_PATH = r"stats/" + current_benchmark +str(k) + "_benchmark_query_time_by_santos_"+current_mode+".csv"
    
    #load pickle files to the dictionary variables
    yago_loading_start_time = time.time()
    if which_benchmark == 4 or which_benchmark == 5:
        fd_dict = {}
    else:
        fd_dict = genFunc.loadDictionaryFromPickleFile(FD_FILE_PATH)#fd_dict是什么?
    if which_mode == 1 or which_mode == 3:
        label_dict = genFunc.loadDictionaryFromPickleFile(LABEL_FILE_PATH)
        type_dict = genFunc.loadDictionaryFromPickleFile(TYPE_FILE_PATH)
        class_dict = genFunc.loadDictionaryFromPickleFile(CLASS_FILE_PATH)
        fact_dict = genFunc.loadDictionaryFromPickleFile(FACT_FILE_PATH)
        yago_inverted_index = genFunc.loadDictionaryFromPickleFile(YAGO_MAIN_INVERTED_INDEX_PATH)
        yago_relation_index = genFunc.loadDictionaryFromPickleFile(YAGO_MAIN_RELATION_INDEX_PATH)
        main_index_triples = genFunc.loadDictionaryFromPickleFile(YAGO_MAIN_PICKLE_TRIPLE_INDEX_PATH)
    else:
        label_dict = {}
        type_dict = {}
        class_dict = {}
        fact_dict = {}
        yago_inverted_index = {}
        yago_relation_index = {}
        main_index_triples = {}
    
    #load synth indexes
    #synth_type_lookup = genFunc.loadDictionaryFromCsvFile(SYNTH_TYPE_LOOKUP_PATH)
    #synth_relation_lookup = genFunc.loadDictionaryFromCsvFile(SYNTH_RELATION_LOOKUP_PATH)
    if which_mode == 2 or which_mode == 3:
        synth_type_kb = genFunc.loadDictionaryFromPickleFile(SYNTH_TYPE_KB_PATH)
        synth_relation_kb = genFunc.loadDictionaryFromPickleFile(SYNTH_RELATION_KB_PATH)
        synth_type_inverted_index= genFunc.loadDictionaryFromPickleFile(SYNTH_TYPE_INVERTED_INDEX_PATH)
        synth_relation_inverted_index = genFunc.loadDictionaryFromPickleFile(SYNTH_RELATION_INVERTED_INDEX_PATH)
    else:
        synth_type_kb = {}
        synth_relation_kb = {}
        synth_type_inverted_index = {}
        synth_relation_inverted_index = {}    
    

    #load the union groundtruth and subject columns
    if which_benchmark == 4:
        ground_truth = genFunc.loadDictionaryFromPickleFile('groundtruth/webtableUnionBenchmark.pickle')
    elif which_benchmark == 5:
        ground_truth = genFunc.loadDictionaryFromPickleFile('groundtruth/opendataUnionBenchmark.pickle') 
    else:
        ground_truth = genFunc.loadDictionaryFromPickleFile(GROUND_TRUTH_PATH)
    subject_col = genFunc.loadDictionaryFromPickleFile(SUBJECT_COL_PATH)
    benchmarkLoadEnd = time.time()
    difference = int(benchmarkLoadEnd - benchmarkLoadStart)
    print("Time taken to load benchmarks in seconds:",difference)
    print("-----------------------------------------\n")
    scoring_function = 2
    computation_start_time = time.time()
    query_table_file_name = glob.glob(QUERY_TABLE_PATH+"*.csv")#把在原来的query中的去掉
    #orgin_table_file_name = glob.glob('/data_ssd/webtable/small_query_test/query/'+"*.csv")
    #query_table_file_name = [table for table in query_table_file_name if table not in orgin_table_file_name]
    truePositive = [0, 0, 0, 0, 0, 0, 0, 0]
    falsePositive = [0, 0, 0, 0, 0, 0, 0, 0]
    falseNegative = [0, 0, 0, 0, 0, 0, 0, 0]
    avg_pr = [[], [], [], [], [], [], [], []]
    avg_rc = [[], [], [], [], [], [], [], []]
    query_table_yielding_no_results = 0
    map_output_dict = {}
    true_output_dict = {}
    false_output_dict = {}
    total_queries = 1
    all_query_time = {}
    for table in query_table_file_name:
        flag = 0
        table_name = table.rsplit(os.sep, 1)[-1]
        print("Processing Table number:", total_queries)
        print("Table Name:", table_name)
        total_queries += 1
        if (table_name in ground_truth):
            expected_tables = ground_truth[table_name]
            totalPositive = len(expected_tables)
            #k = 5
            value_of_k = [5, 10, 20, 30, 40, 50, 60, len(expected_tables)]#这是个辅助数组，用来说明都有哪些k可以选择
        else:
            print("The ground truth for this table is missing.")
            expected_tables = []
            totalPositive = 0
           #continue
        current_query_time_start = time.time_ns()
        bagOfSemanticsFinal = []
        input_table = pd.read_csv(table, encoding='latin1',low_memory=False)
        unique_values = input_table.nunique().max()
        rowCardinality = {}
        rowCardinalityTotal = 0
        bag_of_semantics_final = []
        col_id = 0
        stemmed_file_name = Path(table).stem#提取出文件的名字，不带.csv
        #subject_index = subject_col[stemmed_file_name]
        subject_index = subject_col.get(stemmed_file_name, 0)
        if which_mode == 1 or which_mode == 3:
            relation_tuple_bag_of_words, entity_finding_relations, relation_dependencies, relation_dictionary, recent_hits = computeRelationSemantics(input_table, subject_index, label_dict, fact_dict)#计算relation语义
            column_tuple_bag_of_words, column_dictionary, subject_semantics = computeColumnSemantics(input_table, subject_index, label_dict, type_dict, class_dict, entity_finding_relations, scoring_function)#计算column语义
        else:
            relation_tuple_bag_of_words = []
            entities_finding_relation = {}
            relation_dependencies = []
            relation_dictionary = {}
            column_tuple_bag_of_words = []
            column_dictionary = {}
            subject_semantics = ""
        if which_mode == 2 or which_mode == 3:
            synthetic_relation_dictionary, synthetic_triples_dictionary, synth_subject_semantics = computeSynthRelation(input_table, subject_index, synth_relation_kb)
            synthetic_column_dictionary = computeSynthColumnSemantics(input_table, synth_type_kb)
            #synthetic_triples_dictionary = {col1-col2 : [(rel_name1, score1), (rel_name2, score2)]}
        else:
            synthetic_relation_dictionary = {}
            synthetic_triples_dictionary = {}
            synthetic_column_dictionary = {}
            synth_subject_semantics = set()
        current_relations = set()
        if table_name in fd_dict:
            current_relations = set(fd_dict[table_name])
        for item in relation_dependencies:    
            current_relations.add(item)
        query_table_triples = {}#这个三元组是干啥的？是语义图吗
        synth_query_table_triples = {}
        if len(column_dictionary) > 0:
                for i in range(0, max(column_dictionary.keys())):
                    subject_type = column_dictionary.get(i, "None")
                    if subject_type != "None":
                        for j in range(i+1, max(column_dictionary.keys()) + 1):
                            object_type = column_dictionary.get(j, "None")
                            relation_tuple_forward = "None"
                            relation_tuple_backward = "None"
                            if object_type != "None":
                                for subject_item in subject_type:
                                    for object_item in object_type:
                                        subject_name = subject_item[0]
                                        subject_score = subject_item[1]        
                                        object_name = object_item[0]
                                        object_score = object_item[1]
                                        if str(i) + "-" + str(j) in current_relations:
                                            relation_tuple_forward = relation_dictionary.get(str(i) + "-" + str(j), "None")
                                        if str(j) + "-" + str(i) in current_relations:
                                            relation_tuple_backward = relation_dictionary.get(str(j) + "-" + str(i), "None")
                                        column_pairs = str(i) + "-" + str(j)
                                        if relation_tuple_forward != "None":
                                            relation_name = relation_tuple_forward[0][0]
                                            relation_score = relation_tuple_forward[0][1]
                                            triple_dict_key = subject_name + "-" + relation_name + "-" + object_name
                                            triple_score = subject_score * relation_score * object_score
                                            if triple_dict_key in query_table_triples:
                                                if triple_score > query_table_triples[triple_dict_key][0]:
                                                    query_table_triples[triple_dict_key] = (triple_score, column_pairs)
                                            else:
                                              query_table_triples[triple_dict_key] = (triple_score, column_pairs)
                                        if relation_tuple_backward != "None":
                                            relation_name = relation_tuple_backward[0][0]
                                            relation_score = relation_tuple_backward[0][1]
                                            triple_dict_key = object_name + "-" + relation_name + "-" + subject_name
                                            triple_score = subject_score * relation_score * object_score
                                            if triple_dict_key in query_table_triples:
                                                if triple_score > query_table_triples[triple_dict_key][0]:
                                                   query_table_triples[triple_dict_key] = (triple_score, column_pairs)
                                            else:
                                              query_table_triples[triple_dict_key] =  (triple_score, column_pairs)
         #check if synthetic KB has found triples
        for key in synthetic_triples_dictionary:
            if len(synthetic_triples_dictionary[key]) > 0:
                synthetic_triples = synthetic_triples_dictionary[key]
                for synthetic_triple in synthetic_triples:
                    synthetic_triple_name = synthetic_triple[0]
                    synthetic_triple_score = synthetic_triple[1]
                    if synthetic_triple_name in synth_query_table_triples:
                        if synthetic_triple_score > synth_query_table_triples[synthetic_triple_name][0]:
                            synth_query_table_triples[synthetic_triple_name] = (synthetic_triple_score, key)
                    else:
                        synth_query_table_triples[synthetic_triple_name] = (synthetic_triple_score, key)
        query_table_triples = set(query_table_triples.items())
        synth_query_table_triples = set(synth_query_table_triples.items())
       
        total_triples = len(query_table_triples)
        
        table_count_final = {}
        eliminate_less_matching_tables = {}
        tables_containing_intent_column = {}
        #to make sure that the subject column is present
        if subject_semantics != "" and subject_semantics+"-c" in yago_inverted_index:
          intent_containing_tables = yago_inverted_index[subject_semantics+"-c"]
          for table_tuple in intent_containing_tables:
            tables_containing_intent_column[table_tuple[0]] = 1
        
        
        already_used_column = {}   
        for item in query_table_triples:
            matching_tables = main_index_triples.get(item[0], "None") #checks yago inv4erted index
            if matching_tables != "None":
                triple = item[0]
                query_score = item[1][0]
                col_pairs = item[1][1]
                for data_lake_table in matching_tables:
                    dlt_name = data_lake_table[0]
                    if triple in synth_subject_semantics:
                        tables_containing_intent_column[dlt_name] = 1
                    dlt_score = data_lake_table[1]
                    total_score = query_score * dlt_score
                    if (dlt_name, col_pairs) not in already_used_column:
                        if dlt_name not in table_count_final:
                            table_count_final[dlt_name] = total_score
                        else:
                            table_count_final[dlt_name] += total_score
                        already_used_column[(dlt_name, col_pairs)] = total_score
                    else:
                        if already_used_column[(dlt_name, col_pairs)] > total_score:
                            continue
                        else: #use better matching score
                            if dlt_name not in table_count_final:
                                table_count_final[dlt_name] = total_score
                            else:
                                table_count_final[dlt_name] -= already_used_column[(dlt_name, col_pairs)]
                                table_count_final[dlt_name] += total_score
                            already_used_column[(dlt_name, col_pairs)] = total_score
        synth_col_scores = {}
        for item in synth_query_table_triples:
            matching_tables = synth_relation_inverted_index.get(item[0], "None") #checks synth KB index
            if matching_tables != "None":
                triple = item[0]
                query_rel_score = item[1][0]
                col_pairs = item[1][1]
                for data_lake_table in matching_tables:
                    dlt_name = data_lake_table[0]
                    if triple in synth_subject_semantics:
                        tables_containing_intent_column[dlt_name] = 1
                    dlt_rel_score = data_lake_table[1][0]
                    dlt_col1 = data_lake_table[1][1]
                    dlt_col2 = data_lake_table[1][2]
                    query_col1 = col_pairs.split("-")[0]
                    query_col2 = col_pairs.split("-")[1]
                    dlt_col1_contents = {}
                    dlt_col2_contents = {}
                    query_col1_contents = {}
                    query_col2_contents = {}
                    if (dlt_name, dlt_col1) in synth_type_inverted_index:    
                        dlt_col1_contents = synth_type_inverted_index[(dlt_name, dlt_col1)]
                    if (dlt_name, dlt_col2) in synth_type_inverted_index:    
                        dlt_col2_contents = synth_type_inverted_index[(dlt_name, dlt_col2)]
                    if query_col1 in synthetic_column_dictionary:
                        query_col1_contents = synthetic_column_dictionary[query_col1]
                    if query_col2 in synthetic_column_dictionary:
                        query_col2_contents = synthetic_column_dictionary[query_col2]
                    #find intersection between dlt1 and query1
                    
                    max_score = [0,0,0,0]
                    if col_pairs +"-"+dlt_col1 + "-"+ dlt_col2 in synth_col_scores:
                        total_score = synth_col_scores[col_pairs +"-"+dlt_col1 + "-"+ dlt_col2] * dlt_rel_score * query_rel_score
                    else:
                        match_keys_11 = dlt_col1_contents.keys() & query_col1_contents.keys()
                        if len(match_keys_11) > 0:
                            for each_key in match_keys_11:
                                current_score = dlt_col1_contents[each_key] * query_col1_contents[each_key]
                                if current_score > max_score[0]:
                                    max_score[0] = current_score
                        
                        match_keys_12 = dlt_col1_contents.keys() & query_col2_contents.keys()
                        if len(match_keys_12) > 0:
                            for each_key in match_keys_12:
                                current_score = dlt_col1_contents[each_key] * query_col2_contents[each_key]
                                if current_score > max_score[1]:
                                    max_score[1] = current_score
                        
                        match_keys_21 = dlt_col2_contents.keys() & query_col1_contents.keys()
                        
                        if len(match_keys_21) > 0:
                            for each_key in match_keys_21:
                                current_score = dlt_col2_contents[each_key] * query_col1_contents[each_key]
                                if current_score > max_score[2]:
                                    max_score[2] = current_score
                        
                        match_keys_22 = dlt_col2_contents.keys() & query_col2_contents.keys()
                        
                        
                        if len(match_keys_22) > 0:
                            for each_key in match_keys_22:
                                current_score = dlt_col2_contents[each_key] * query_col2_contents[each_key]
                                if current_score > max_score[3]:
                                    max_score[3] = current_score
                        
                        
                        max_score = sorted(max_score, reverse = True)
                        synth_col_scores[col_pairs +"-"+dlt_col1 + "-"+ dlt_col2] = max_score[0] * max_score[1]    
                        total_score = query_rel_score * dlt_rel_score * max_score[0] * max_score[1]
                    if (dlt_name, col_pairs) not in already_used_column:
                        if dlt_name not in table_count_final:
                            table_count_final[dlt_name] = total_score
                        else:
                            table_count_final[dlt_name] += total_score
                        already_used_column[(dlt_name, col_pairs)] = total_score
                    else:
                        if already_used_column[(dlt_name, col_pairs)] > total_score:
                            continue
                        else: #use better matching RSCS' score
                            if dlt_name not in table_count_final:
                                table_count_final[dlt_name] = total_score
                            else:
                                table_count_final[dlt_name] -= already_used_column[(dlt_name, col_pairs)]
                                table_count_final[dlt_name] += total_score
                            already_used_column[(dlt_name, col_pairs)] = total_score   
          
        #to make sure that the match was because of intent column
        tables_to_throw = set()
        if len(tables_containing_intent_column) > 0:
          for shortlisted_table in table_count_final:
            if shortlisted_table not in tables_containing_intent_column:
              tables_to_throw.add(shortlisted_table)
        if len(tables_to_throw) > 0 and len(tables_to_throw) < len(table_count_final):
          for item in tables_to_throw:    
            del table_count_final[item]
            
        sortedTableList = sorted(table_count_final.items(), key=lambda x: x[1], reverse=True)
        temp_map = []
        for map_item in sortedTableList:
            temp_map.append(map_item[0])        
        map_output_dict[table_name] = temp_map
        current_query_time_end = time.time_ns()
        all_query_time[table_name] = int(current_query_time_end - current_query_time_start)/10**9
        if which_benchmark == 3:
            continue
        print("Dynamic K = ", k)
        print("The best matching tables are:")
        i = 0
        dtp = 0
        dfp = 0
        temp_true_results = []
        temp_false_results = []
        enum = 0

        for key in sortedTableList:
            data_lake_table = key[0].split(os.sep)[-1]
            if enum <5:
                print(data_lake_table, key[1])
            enum +=1
            if data_lake_table in expected_tables:
                dtp = dtp+1
                temp_true_results.append(data_lake_table)
                if enum <=5:
                    print("true")
            else:
                dfp = dfp+1
                temp_false_results.append(data_lake_table)
                if enum<= 5:
                    print("false")
            i = i +1
            #temp_true_results.append(data_lake_table)
            if (i >=k):
                break
        if dtp + dfp == 0 :
          query_table_yielding_no_results += 1
        true_output_dict[table_name] = temp_true_results
        false_output_dict[table_name] = temp_false_results
        print("Current True positives:", dtp)
        print("Current False Positives:", dfp)
        print("Current False Negative:", totalPositive - dtp)
   
    computation_end_time = time.time()
    difference = int(computation_end_time - computation_start_time)
    print("Time taken to process all query tables and print the results in seconds:", difference)
genFunc.saveDictionaryAsPickleFile(map_output_dict, FINAL_RESULT_PICKLE_PATH)


if which_benchmark != 3: #manual verification for real data lake benchmark
    with open(TRUE_RESULTS_PATH,"w",newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in true_output_dict.items():
            for value in v:
                w.writerow([k,value])

    with open(FALSE_RESULTS_PATH,"w",newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        for k, v in false_output_dict.items():
            for value in v:
                w.writerow([k,value])
# -*- coding: utf-8 -*-

import glob
import time
import pickle #version 4
import argparse
import pandas as pd
import generalFunctions as genFunc
import expandSearch as expand
import csv
import os

from tqdm import tqdm

#utility function used to convert dictionary to lists
def convertDictToList(dictionary):
    for semantics in dictionary:
        tables = []
        temp = dictionary[semantics]
        for items in temp:
            tables.append((items, temp[items]))
        dictionary[semantics] = tables 
    return dictionary


def merge_dicts(dict1, path):
    dict2 = genFunc.loadDictionaryFromPickleFile(path)
    merged_dict = dict1.copy()

    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key].extend(value)
        else:
            merged_dict[key] = value

    return merged_dict

#compute relationship semantics using yago
def computeRelationSemantics(input_table, tab_id, LABEL_DICT, FACT_DICT):
    relation_bag_of_words = []
    total_cols = input_table.shape[1]
    relation_dependencies = []
    entities_finding_relation = {}
    relation_dictionary = {}
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
                                                #keep track of the entity finding relation. We will use this for column semantics search
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
                if len(semantic_dict_backward) >0:
                    relation_bag_of_words.append((max(semantic_dict_backward, key=semantic_dict_backward.get)+"-r", str(j)+"_"+str(i), max(semantic_dict_backward.values())/ total_kb_backward_hits))
                    relation_dependencies.append(str(j)+"-"+str(i))
                    relation_dictionary[str(j)+"-"+str(i)] = [(max(semantic_dict_backward, key=semantic_dict_backward.get), max(semantic_dict_backward.values())/ total_kb_backward_hits)]

    return relation_bag_of_words, entities_finding_relation, relation_dependencies, relation_dictionary

#compute column semantics
def computeColumnSemantics(input_table, tab_id, LABEL_DICT, TYPE_DICT, CLASS_DICT, TYPE_SCORE_DICT, RELATION_DICT):
    col_id = 0
    not_found_in_yago = []
    column_bag_of_words = []
    column_dictionary = {}
    for (columnName, columnData) in input_table.iteritems():
        if genFunc.getColumnType(input_table[columnName].tolist()) == 1: #check column Type
            input_table[columnName] = input_table[columnName].map(str)                
            #get unique values in the column and preprocess them.
            value_list = genFunc.preprocessListValues(input_table[columnName].unique())
            all_found_types = {}
            total_kb_hits = 0
            for value in value_list:
                current_entities = set()
                current_types = set()
                current_entities = RELATION_DICT.get(str(col_id) + "_"+ value, "None")
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
            #print(all_found_types)    
            #find the top-level type with highest count.
            all_top_types = [v for v in sorted(all_found_types.items(), key=lambda kv: (-kv[1], kv[0])) if v[0] in CLASS_DICT]
            if all_top_types:
                selected_top_type = all_top_types[0][0]
                children_of_top_types = CLASS_DICT[selected_top_type]
                #add children of top types to the bag of word\
            for each in all_found_types:
                if each in children_of_top_types and each in type_score_dict:
                    penalized_score = type_score_dict[each][0]
                    #penalization score
                    current_type_score = all_found_types[each] * penalized_score
                    column_bag_of_words.append((each+"-c", str(col_id),current_type_score/ total_kb_hits))
                    if int(col_id) in column_dictionary:
                        column_dictionary[int(col_id)].append((each, current_type_score/ total_kb_hits))
                    else:
                        column_dictionary[int(col_id)] = [(each, current_type_score/total_kb_hits)]
        col_id += 1
        
    return column_bag_of_words, column_dictionary


def custom_slice(input_list, num, i):
    if num <= 0:
        raise ValueError("Number of slices (num) must be greater than 0")
    
    slice_size = len(input_list) // num
    if i < 0 or i >= num:
        raise ValueError("Index (i) is out of range")
    
    start = i * slice_size
    end = start + slice_size if i < num - 1 else None
    
    return input_list[start:end]


if __name__ == "__main__": 
    #添加一个benchmark，webtable_large,添加时间统计的部分,怎么追加呀

    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('-s', '--start', type=int, default=1, help="start folder_path number")
    parser.add_argument('-e', '--end', type=int, default=2, help="end folder_path number")
    args = parser.parse_args()
    start_id = args.start
    end_id = args.end

    which_benchmark = 5
    while which_benchmark != 1 and which_benchmark != 2 and which_benchmark != 3 and which_benchmark != 4 and which_benchmark != 5:
      print("Press 1 for TUS Benchmark, 2 for SANTOS (small) benchmark and 3 for SANTOS (large) benchmark 4 for webtable benchmark and 5 for opendata benchmark.") #small benchmark is referred as santos benchmark in the code
      which_benchmark = int(input())
    if which_benchmark == 1:
        current_benchmark = "tus"
    elif which_benchmark == 2:
        current_benchmark = "santos" 
    elif which_benchmark == 3:
        current_benchmark = "real_tables"
    elif which_benchmark == 4:
        current_benchmark = "webtable_large" 
    else:
        current_benchmark = 'opendata'
    #load the YAGO KB from dictionary files and data lake table names
    #edit the path below according to the location of the pickle files.
    YAGO_PATH = r"yago/yago_pickle/" 
    #edit the line below if the dlt are at different locations
    #DATA_LAKE_TABLE_PATH = r"../input/dataLakeTablesMethod1/testDLT/"
    if which_benchmark == 4:
        DATA_LAKE_TABLE_PATH = "/data_ssd/webtable/large/split_1/"
    elif which_benchmark == 5:
        DATA_LAKE_TABLE_PATH = "/data_ssd/opendata/small_query/query/"
    else:
        DATA_LAKE_TABLE_PATH = r"benchmark/" + current_benchmark + "_benchmark/datalake/"
    #DATA_LAKE_TABLE_PATH = r"/work/datalab/tus_benchmark/datalake/" 
    LABEL_FILE_PATH = YAGO_PATH + "yago-wd-labels_dict.pickle" 
    TYPE_FILE_PATH = YAGO_PATH + "yago-wd-full-types_dict.pickle" 
    CLASS_FILE_PATH = YAGO_PATH + "yago-wd-class_dict.pickle"
    FACT_FILE_PATH = YAGO_PATH + "yago-wd-facts_dict.pickle"
    TYPE_SCORE_FILE_PATH = YAGO_PATH + "yago-type-score.pickle"
    YAGO_MAIN_INVERTED_INDEX_PATH = r"hashmap/" + current_benchmark + str(start_id) + "_query_main_yago_index.pickle"
    YAGO_MAIN_RELATION_INDEX_PATH = r"hashmap/" + current_benchmark + str(start_id) + "_query_main_relation_index.pickle"
    YAGO_MAIN_PICKLE_TRIPLE_INDEX_PATH = r"hashmap/" + current_benchmark + str(start_id) + "_query_main_triple_index.pickle"
    YAGO_MAIN_CSV_TRIPLE_INDEX_PATH = r"stats/" + current_benchmark + str(start_id) + "_query_benchmark_main_triple_index.csv"
    INDIVIDUAL_TIME_FILE_PATH = r"stats/" + current_benchmark + str(start_id) + "_query_benchmark_individual_time.pickle"
    INDIVIDUAL_TIME_CSV_FILE_PATH = r"stats/" + current_benchmark + str(start_id) + "_query_benchmark_individual_time.csv"
    #load pickle files to the dictionary variables
    yago_loading_start_time = time.time()
    label_dict = genFunc.loadDictionaryFromPickleFile(LABEL_FILE_PATH)
    type_dict = genFunc.loadDictionaryFromPickleFile(TYPE_FILE_PATH)
    class_dict = genFunc.loadDictionaryFromPickleFile(CLASS_FILE_PATH)
    fact_dict = genFunc.loadDictionaryFromPickleFile(FACT_FILE_PATH)
    type_score_dict = genFunc.loadDictionaryFromPickleFile(TYPE_SCORE_FILE_PATH)
    
    yago_loading_end_time = time.time()
    print("Time taken to load yago in seconds:",
          int(yago_loading_end_time - yago_loading_start_time))
    
    #start processing the relation semantics
    yago_processing_start_time = time.time_ns()    
    #compute relation semantics
    if which_benchmark == 0:
        #取出split1 - 10中的所有csv格式的table
        DATA_LAKE_TABLES = []
        for folder in range(1, 2):
            folder_name = f'split_{folder}'
            folder_path = DATA_LAKE_TABLE_PATH + '/' + folder_name
            DATA_LAKE_TABLES.extend(glob.glob(f'{folder_path}/*.csv'))
    else:
        DATA_LAKE_TABLES = glob.glob(DATA_LAKE_TABLE_PATH + "*.csv")
    DATA_LAKE_TABLES = custom_slice(DATA_LAKE_TABLES, 5, start_id)
    tab_id = 0
    ignored_table = 0
    # main_inverted_index = genFunc.loadDictionaryFromPickleFile(r"hashmap/" + current_benchmark + "_main_yago_index.pickle")
    # main_relation_index = genFunc.loadDictionaryFromPickleFile(r"hashmap/" + current_benchmark + "_main_relation_index.pickle")
    main_inverted_index = {}
    main_relation_index = {}
    main_index_triples = {}
    individual_time = {}
    not_found_in_kb = 0
    for table in tqdm(DATA_LAKE_TABLES):
        try:
            input_table = pd.read_csv(table, encoding='latin1', on_bad_lines="skip", low_memory=False)
            #print("------------------------------")
            #print("Processing table :", tab_id)
            table_name = table.rsplit(os.sep,2)[-2] + '/' + table.rsplit(os.sep,1)[-1]
            #print("Table path:", table)
            individual_time_start = time.time_ns()
            relation_tuple_bag_of_words, entity_finding_relations, relation_dependencies, relation_dictionary = computeRelationSemantics(input_table, tab_id, label_dict, fact_dict)
            column_tuple_bag_of_words, column_dictionary = computeColumnSemantics(input_table, tab_id, label_dict, type_dict, class_dict, type_score_dict, entity_finding_relations)
            #store the relation dependencies to main_relation_index dictionary
            #format: {table_name:  [col1-col2, col3-col4, col1-col3, ...]}
            current_relations = set()
            # if table_name in fd_dict:
            #     current_relations = set(fd_dict[table_name])
            for item in relation_dependencies:    
                current_relations.add(item)
            main_relation_index[table_name] = current_relations
            if len(current_relations) == 0:
                not_found_in_kb += 1
            #total relations in yago = 133; 1 / 133 = 0.00751
            #total top level types in yago =  2714; 1 / 2714 = 0.000368 
            #print(max(column_dictionary.keys()))
            if len(column_dictionary) > 0 :
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
                                        if relation_tuple_forward != "None":
                                            relation_name = relation_tuple_forward[0][0]
                                            relation_score = relation_tuple_forward[0][1]
                                            triple_dict_key = subject_name + "-" + relation_name + "-" + object_name
                                            triple_score = subject_score * relation_score * object_score
                                            if triple_dict_key not in main_index_triples:
                                                main_index_triples[triple_dict_key] = {table_name: triple_score}
                                            else:
                                                current_tables = main_index_triples[triple_dict_key]
                                                if table_name in current_tables:
                                                    if triple_score > current_tables[table_name]:
                                                        current_tables[table_name] = triple_score
                                                else:
                                                    current_tables[table_name] = triple_score
                                                main_index_triples[triple_dict_key] = current_tables
                                        if relation_tuple_backward != "None":
                                            relation_name = relation_tuple_backward[0][0]
                                            relation_score = relation_tuple_backward[0][1]
                                            triple_dict_key = object_name + "-" + relation_name + "-" + subject_name
                                            triple_score = subject_score * relation_score * object_score
                                            if triple_dict_key not in main_index_triples:
                                                main_index_triples[triple_dict_key] = {table_name: triple_score}
                                            else:
                                                current_tables = main_index_triples[triple_dict_key]
                                                if table_name in current_tables:
                                                    if triple_score > current_tables[table_name]:
                                                        current_tables[table_name] = triple_score
                                                else:
                                                    current_tables[table_name] = triple_score
                                                main_index_triples[triple_dict_key] = current_tables
            #store the TBW to the main inverted index.
            # format: {semantic-type :[(tablename, columnId, semantic_score), (...)]}
            for semantics in relation_tuple_bag_of_words:
                if semantics[0] not in main_inverted_index:
                    main_inverted_index[semantics[0]] = [(table_name, semantics[1], semantics[2])]
                else:
                    main_inverted_index[semantics[0]].append((table_name, semantics[1], semantics[2]))
            for semantics in column_tuple_bag_of_words:
                if semantics[0] not in main_inverted_index:
                    main_inverted_index[semantics[0]] = [(table_name, semantics[1], semantics[2])]
                else:
                    main_inverted_index[semantics[0]].append((table_name, semantics[1], semantics[2]))
            individual_time_stop = time.time_ns()
            individual_time[table_name] = (individual_time_stop - individual_time_start) / 10**9.
            #print(relation_tuple_bag_of_words)
            #print(column_tuple_bag_of_words)
        except Exception as e: 
             #print(e)
             #print("This table is not readable (ignored)!!!")
             ignored_table+=1
        tab_id += 1
        
    #save main inverted index and relation index as pickle files.
    yago_processing_end_time = time.time_ns()
    print("Time taken to process all tables in seconds:",
          int(yago_processing_end_time - yago_processing_start_time)/10**9)
    print("Number of ignored tables:", ignored_table)
    print("Tables not found in KB:", not_found_in_kb)
    genFunc.saveDictionaryAsPickleFile(main_inverted_index, YAGO_MAIN_INVERTED_INDEX_PATH)
    genFunc.saveDictionaryAsPickleFile(main_relation_index, YAGO_MAIN_RELATION_INDEX_PATH)
    main_index_triples = convertDictToList(main_index_triples)
    #main_index_triples = merge_dicts(main_index_triples, r"hashmap/" + current_benchmark + "_main_triple_index.pickle")
    genFunc.saveDictionaryAsPickleFile(main_index_triples, YAGO_MAIN_PICKLE_TRIPLE_INDEX_PATH)
    genFunc.saveDictionaryAsPickleFile(individual_time, INDIVIDUAL_TIME_FILE_PATH)
    count = 0
    with open(YAGO_MAIN_CSV_TRIPLE_INDEX_PATH,"w",newline='', encoding="utf-8") as f:
      w = csv.writer(f)
      for k, v in main_index_triples.items():
        w.writerow([k,v])
        count +=1
        if count > 150:
            break
    with open(INDIVIDUAL_TIME_CSV_FILE_PATH,"w",newline='', encoding="utf-8") as f:
      w = csv.writer(f)
      for k, v in individual_time.items():
        w.writerow([k,v])

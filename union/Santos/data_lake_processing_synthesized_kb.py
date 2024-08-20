# -*- coding: utf-8 -*-


import glob
import time
import json
import random
import csv
import argparse
import pickle #version 4
import pandas as pd
import generalFunctions as genFunc
import sys
import os
from tqdm import tqdm

def convertSynthDictToList(synthKB):
    for value in synthKB:
        bagOfSemantics = []
        temp = synthKB[value]
        for items in temp:
            bagOfSemantics.append((items, temp[items]))
        synthKB[value] = bagOfSemantics 
    return synthKB

def convertSynthInvIndexToList(synthInvertedIndex):
    for semantics in synthInvertedIndex:
        tables = []
        temp = synthInvertedIndex[semantics]
        for items in temp:
            tables.append((items, temp[items]))
        synthInvertedIndex[semantics] = tables 
    return synthInvertedIndex

#This function takes the data lake table location as input and create the lookup table for the relationships.
#We need synthetic inverted index and synthetic KB for on-the-fly step. Lookup table is just an intermediate file.
def createRelationSemanticsLookupTable(filenames, FDDs):
    lookupTable = {}
    tab_id = 0
    ignored_table = 0
    for file in tqdm(filenames):
        try:
            inputTable = pd.read_csv(file, encoding='latin1')
            total_cols = inputTable.shape[1]
            fdKey = file.rsplit(os.sep,2)[-2] + '/' + file.rsplit(os.sep,1)[-1] #extract the name of table. Eg. extract "exampleTable.csv" from "../input/dataLakeTables/exampleTable.csv"
            currentTableFDs = []
            # print("Table recently visited for relation lookup creation:", fdKey)
            # print("table id:", tab_id)
            if fdKey in FDDs:
                currentTableFDs = FDDs[fdKey]
                fdFound = 1
                # print("FD found for table", file)
            else:
                fdFound = 0
                # print("FD not found for table", file)
            for i in range(0, total_cols-1):
                if genFunc.getColumnType(inputTable.iloc[:, i].tolist()) == 1: 
                    for j in range(i+1, total_cols):
                        if genFunc.getColumnType(inputTable.iloc[:, j].tolist()) == 1:
                            if str(i)+"-"+str(j) in currentTableFDs or str(j)+"-"+str(i) in currentTableFDs or fdFound == 0:
                                rel_sem = "r"+str(tab_id)+"_"+str(i)+"_"+str(j)
                                dataFrameTemp = inputTable.iloc[:, [i, j]]
                                dataFrameTemp = (dataFrameTemp.drop_duplicates()).dropna()
                                projectedRowsNum = dataFrameTemp.shape[0]
                                #assign relation semantic to each value pairs of i and j
                                for k in range(0, projectedRowsNum):
                                    #extract subject and object
                                    sub = genFunc.preprocessString(str(dataFrameTemp.iloc[k][0]).lower())
                                    obj = genFunc.preprocessString(str(dataFrameTemp.iloc[k][1]).lower())
                                    #print(sub)
                                    #print(obj)
                                    subNull = genFunc.checkIfNullString(sub)
                                    objNull = genFunc.checkIfNullString(obj)
                                    if subNull != 0 and objNull != 0:  #both not nulls
                                        temp = set()
                                        value = sub+"__"+obj
                                        #print(value)
                                        if value in lookupTable:
                                            temp = lookupTable[value]
                                        temp.add(rel_sem)
                                        lookupTable[value] = temp
            
        except Exception as e:
            # print(e)
            # print("This table is not readable (ignored)!!!")
            ignored_table+=1
        tab_id += 1
    print("Lookup table created for relation semantics.")
    return lookupTable



#This function takes the data lake table location as input and create the lookup table for the columns.
#We need synthetic inverted index and synthetic KB for on-the-fly step. Lookup table is just an intermediate file.
def createColumnSemanticsLookupTable(filenames, FDDs):
    lookupTable = {}
    tab_id = 0
    ignored_tables = 0
    for file in tqdm(filenames):
        try:
            inputTable = pd.read_csv(file, encoding='latin1', warn_bad_lines=True, error_bad_lines=False)
            col_id = 0
            table_name = file.rsplit(os.sep,2)[-2] + '/' + file.rsplit(os.sep,1)[-1]
            currentTableFDs = []
            #print("Table recently visited for type lookup creation:", table_name)
            #print("table id:", tab_id)
            current_fd_cols = set()
            fdFound = 0
            if table_name in FDDs:
                currentTableFDs = FDDs[table_name]
                for item in currentTableFDs:
                    explode = item.split("-")
                    for each in explode:
                        current_fd_cols.add(each)
                fdFound = 1
            #print(fdFound)
            for (columnName, columnData) in inputTable.iteritems():
                #creating the lookup table for data lake tables
                col_sem = "c"+str(tab_id)+"_"+str(col_id)
                if genFunc.getColumnType(inputTable[columnName].tolist()) == 1:
                    #print(table_name)
                    if str(col_id) in current_fd_cols or fdFound == 0: 
                        inputTable[columnName] = inputTable[columnName].map(str)
                        valueList = genFunc.preprocessListValues(inputTable[columnName].unique())                   
                        #print(valueList)
                        for value in valueList:
                            temp = set()
                            if value in lookupTable:
                                temp = lookupTable[value]
                            temp.add(col_sem)
                            lookupTable[value] = temp
                col_id += 1
        except Exception as e:
            #print(e)
            #print("This table is not readable (ignored)!!!")
            ignored_tables += 1
        tab_id += 1    
    #print("Ignored tables:", ignored_tables)
    return lookupTable

#This function takes the data lake table location as input and create the lookup table.
#We need synthetic inverted index and synthetic KB for on-the-fly step. Lookup table is just an intermediate file.
def createColumnSemanticsSynthKB(lookupTable, filenames, FDDs):
    synthKB = {}
    noise_free_lookup = {}
    for every in lookupTable:
        if len(lookupTable[every]) < 300:
            noise_free_lookup[every] = lookupTable[every]
    lookupTable = noise_free_lookup
    noise_free_lookup = {}
    #synthInvertedIndex  = {}
    main_table_col_index = {}
    ignored_table = 0
    tab_id = 0
    for file in tqdm(filenames):
        try:
            inputTable = pd.read_csv(file, encoding='latin1', warn_bad_lines=True, error_bad_lines=False)
            col_id = 0
            table_name = file.rsplit(os.sep,2)[-2] + '/' + file.rsplit(os.sep,1)[-1] #extract the name of table. Eg. extract "exampleTable.csv" from "../input/dataLakeTables/exampleTable.csv"
            #fdKey = genFunc.cleanTableName(fdKey)
            currentTableFDs = []
            # print("Table recently visited for type KB creation:", table_name)
            # print("table id:", tab_id)
            current_fd_cols = set()
            fdFound = 0
            if table_name in FDDs:
                #print("Fd found", table_name)
                currentTableFDs = FDDs[table_name]
                for item in currentTableFDs:
                    explode = item.split("-")
                    for each in explode:
                        current_fd_cols.add(each)
                fdFound = 1
            #print(fdFound)
            
            for (columnName, columnData) in inputTable.iteritems():
                sem = {}
                semList = set()
                #creating the lookup table for data lake tables
                if genFunc.getColumnType(inputTable[columnName].tolist()) == 1:
                    #print(table_name)
                    if str(col_id) in current_fd_cols or fdFound == 0: #check if the columns are string columns
                        inputTable[columnName] = inputTable[columnName].map(str)
                        valueList = genFunc.preprocessListValues(inputTable[columnName].unique())                   
                        divideBy = len(valueList)                
                        #find bag of semantics for each column
                        for value in valueList:
                            if value in lookupTable:
                                semList = lookupTable[value]
                                for s in semList:
                                    if s in sem:
                                        sem[s] += (1/divideBy)
                                    else:
                                        sem[s] = (1/divideBy)
                            #sem dictionary contains the semantic token for the current column
                        #Assign each value in the current column with the types in sem
                        for value in valueList:
                            if value in lookupTable:
                                temp = {}
                                if value in synthKB: #if value was already processed for previous tables.
                                    temp = synthKB[value]
                                    for s in sem:
                                        if s in temp:
                                            temp[s] = max(sem[s], temp[s]) #take maximum score when same value gets same type from multiple columns
                                        else:
                                            temp[s] = sem[s]
                                    synthKB[value] = temp
                                else:
                                    for s in sem:
                                        temp[s] = sem[s]
                                    synthKB[value] = temp
                                
                     
                        if (table_name, str(col_id)) in main_table_col_index:
                            #print("red flag!!!")
                            pass
                        else:
                            main_table_col_index[table_name+ '*.*' +str(col_id)] = sem
                col_id += 1
        except Exception as e:
            #print(e)
            #print("This table is not readable (ignored)!!!")
            ignored_table += 1                
        tab_id += 1
    #All tables are processed. For convenience convert the dictionary inside synthKB to list      
    synthKB = convertSynthDictToList(synthKB)
    #print(synthInvertedIndex)
    return synthKB, main_table_col_index     
    
def createRelationSemanticsSynthKB(lookupTable, filenames, FDDs):
    synthKB = {}
    synthInvertedIndex  = {}
    ignored_table = 0
    tab_id = 0
    for file in tqdm(filenames):
        try:
            inputTable = pd.read_csv(file, encoding='latin1')
            #print(inputTable)
            total_cols = inputTable.shape[1]
            #total_rows = inputTable.shape[0]
            table_name = file.rsplit(os.sep,2)[-2] + '/' + file.rsplit(os.sep,1)[-1] #extract the name of table. Eg. extract "exampleTable.csv" from "../input/dataLakeTables/exampleTable.csv"
            #fdKey = genFunc.cleanTableName(fdKey)
            currentTableFDs = []
            # print("Computing relation semantics for table", table_name)
            # print("Table Number:", tab_id)
            if table_name in FDDs:
                currentTableFDs = FDDs[table_name]
                fdFound = 1
            else:
                fdFound = 0
            for i in range (0,total_cols-1):
                if genFunc.getColumnType(inputTable.iloc[:,i].tolist()) == 1: #the subject in rdf triple should be a text column
                    for j in range(i+1, total_cols):
                        if genFunc.getColumnType(inputTable.iloc[:, j].tolist()) == 1:
                            sem = {}
                            semList = set() 
                            if (str(i)+"-"+str(j)) in currentTableFDs or (str(j)+"-"+str(i)) in currentTableFDs or fdFound == 0:
                                dataFrameTemp = inputTable.iloc[:,[i,j]]
                                dataFrameTemp = (dataFrameTemp.drop_duplicates()).dropna()
                                projectedRowsNum = dataFrameTemp.shape[0]
                                
                                #assign relation semantic to each value pairs of i and j
                                for k in range(0,projectedRowsNum):
                                    #extract subject and object
                                    sub = genFunc.preprocessString(str(dataFrameTemp.iloc[k][0]).lower())
                                    obj = genFunc.preprocessString(str(dataFrameTemp.iloc[k][1]).lower())
                                    subNull = genFunc.checkIfNullString(sub)
                                    objNull = genFunc.checkIfNullString(obj)
                                    if subNull != 0 and objNull != 0:
                                        value = sub+"__"+obj
                                        semList = lookupTable[value]
                                        for s in semList:
                                            if s in sem:
                                                sem[s]+= (1/projectedRowsNum)
                                            else:
                                                sem[s] = (1/projectedRowsNum)
                                #sem dictionary contains the semantic token for the current column
                                #Assign each value in the current column with the types in sem and add to synthKB
                                for k in range(0,projectedRowsNum):
                                    temp = {}
                                    #extract subject and object
                                    sub = genFunc.preprocessString(str(dataFrameTemp.iloc[k][0]).lower())
                                    obj = genFunc.preprocessString(str(dataFrameTemp.iloc[k][1]).lower())
                                    subNull = genFunc.checkIfNullString(sub)
                                    objNull = genFunc.checkIfNullString(obj)
                                    if subNull != 0 and objNull != 0:
                                        value = sub+"__"+obj
                                        
                                        if value in synthKB: #if value was already processed for previous tables.
                                            temp = synthKB[value]
                                            for s in sem:
                                                if s in temp:
                                                    temp[s] = max(sem[s],temp[s]) #take maximum score when same value gets same type from multiple columns
                                                else:
                                                    temp[s] = sem[s]
                                            synthKB[value] = temp
                                        else:
                                            for s in sem:
                                                temp[s] = sem[s]
                                            synthKB[value] = temp
                                for s in sem:
                                    key = s #+"-r"
                                    if key not in synthInvertedIndex:
                                        synthInvertedIndex[key] = {table_name: (sem[s],str(i),str(j))}
                                    else:
                                        current_tables = synthInvertedIndex[key]
                                        if table_name in current_tables:
                                            if sem[s] > current_tables[table_name][0]:
                                                current_tables[table_name] = (sem[s], str(i), str(j))
                                        else:
                                            current_tables[table_name] = (sem[s], str(i), str(j))
                                        synthInvertedIndex[key] = current_tables
        except Exception as e:
            # print(e)
            # print("This table is not readable (ignored)!!!")
            ignored_table+=1                
        tab_id += 1
        
    #All tables are processed. For convenience convert the dictionary inside synthKB to list      
    synthKB = convertSynthDictToList(synthKB)
    #print(synthInvertedIndex)
    synthInvertedIndex = convertSynthInvIndexToList(synthInvertedIndex)
    return synthKB, synthInvertedIndex     


def process_dict(data):
    for key, value in data.items():
        if isinstance(value, set):
            data[key] = list(value)
    return data

def save_dict_to_json(data, path):
    for key, value in data.items():
        if isinstance(value, set):
            data[key] = list(value) 
    with open(path, 'w') as json_file:
        json.dump(data, json_file)


def custom_slice(input_list, num, i):
    if num <= 0:
        raise ValueError("Number of slices (num) must be greater than 0")
    
    slice_size = len(input_list) // num
    if i < 0 or i >= num:
        raise ValueError("Index (i) is out of range")
    
    start = i * slice_size
    end = start + slice_size if i < num - 1 else None
    
    return input_list[start:end]


def choice_table(lst, fraction):
    # 检查fraction是否合法
    if fraction <= 0:
        return lst
    
    # 计算应该选择的元素数量
    num_elements_to_select = len(lst) // fraction
    
    # 使用random.sample来随机选择元素
    selected_elements = random.sample(lst, num_elements_to_select)
    
    return selected_elements

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="命令行参数示例")
    parser.add_argument('-s', '--start', type=int, default=0, help="start folder_path number")
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
        current_benchmark = "webtable"
    else:
        current_benchmark = "opendata"
    #load the data lake tables and declare the path to save the syhthetic KB, inverted index and lookup table
    now = time.time()
    if which_benchmark == 4:
        datalakeTablepath = '/data_ssd/webtable/large/split_1/'
    elif which_benchmark == 5:
        datalakeTablepath = '/data_ssd/opendata/small/'
    else:
        datalakeTablepath = r"benchmark/" + current_benchmark + "_benchmark/datalake/"
    
    if which_benchmark == 5:
        child_folders = ['datasets_CAN', 'datasets_SG', 'datasets_UK', 'datasets_USA']
        dataLakeTables = []
        for i in range(0, 4):
            folder_path = datalakeTablepath+ child_folders[i]
            dataLakeTables.extend(glob.glob(f'{folder_path}/*.csv'))
        
        dataLakeTables = choice_table(dataLakeTables, 20)
        queryTables = glob.glob('/data_ssd/opendata/small_query/query/' + "*.*")
        dataLakeTables.extend(queryTables)
    else:
        dataLakeTables = glob.glob(datalakeTablepath + "*.csv")

    #dataLakeTables = custom_slice(dataLakeTables, 3, start_id)
    
    #pickle_extension = "pbz2"
    pickle_extension = "json"
    synthTypeLookupPath = r"hashmap/" + current_benchmark  + "_synth_type_lookup."+pickle_extension
    synthRelationLookupPath = r"hashmap/" + current_benchmark  + "_synth_relation_lookup."+pickle_extension
    
    synthTypeKBPath =   r"hashmap/" + current_benchmark  + "_synth_type_kb."+pickle_extension
    synthRelationKBPath =   r"hashmap/" + current_benchmark   + "_synth_relation_kb."+pickle_extension
    
    
    synthTypeInvertedIndexPath = r"hashmap/" + current_benchmark  + "_synth_type_inverted_index."+pickle_extension
    synthRelationInvertedIndexPath = r"hashmap/" + current_benchmark  + "_synth_relation_inverted_index."+pickle_extension
    
    synthRelationInvertedIndexCSVPath = r"stats/" + current_benchmark  + "_synth_main_triple_index.csv"
    synthTimeTakenPath = r"stats/" + current_benchmark  + "_synth_preprocessing_time.csv"
    FD_FILE_PATH = r"groundtruth/" + current_benchmark + "_FD_filedict.pickle"
    later = time.time()
    difference = int(later - now)
    if which_benchmark == 4 or which_benchmark == 5 or which_benchmark == 3:
        FDDict = {}
    else:
        FDDict = genFunc.loadDictionaryFromPickleFile(FD_FILE_PATH)
    #create the lookup table for the column semantics. 
    #The keys are the attribute values 
    #and value is the list of synthetic column semantics provided to them.
    
    
    startLookupTypes = time.time()
    lookupTable = createColumnSemanticsLookupTable(dataLakeTables, FDDict)
    #lookupTable = genFunc.loadDictionaryFromPickleFile('hashmap/webtable_synth_type_lookup.pbz2')
    endLookupTypes = time.time()
    type_lookup_time = int(endLookupTypes - startLookupTypes)
    #genFunc.saveDictionaryAsPickleFile(lookupTable,synthTypeLookupPath)
    save_dict_to_json(lookupTable, synthTypeLookupPath)
    startTypeKBTime = time.time()
    synthTypeKB, main_table_col_index = createColumnSemanticsSynthKB(lookupTable, dataLakeTables, FDDict)
    endTypeKBTime = time.time()
    lookupTable = {}
    type_kb_time = int(endTypeKBTime - startTypeKBTime)
    type_total_time = int(endTypeKBTime - startLookupTypes)
    print("Type dictionary created!!!")
    print("------------------------")
    #save all times:
    timeDict = {}
    timeDict["type lookup time"] = type_lookup_time
    timeDict["type kb time"] = type_kb_time
    timeDict["type total time"] = type_total_time
    # with open(synthTimeTakenPath,"w",newline='', encoding="utf-8") as f:
    #   w = csv.writer(f)
    #   for k, v in timeDict.items():
    #     w.writerow([k,v])

    #genFunc.saveDictionaryAsPickleFile(main_table_col_index, synthTypeInvertedIndexPath)
    save_dict_to_json(main_table_col_index, synthTypeInvertedIndexPath)
    main_table_col_index = {}
    #genFunc.saveDictionaryAsPickleFile(synthTypeKB, synthTypeKBPath)
    save_dict_to_json(synthTypeKB, synthTypeKBPath)
    synthTypeKB = {}

    #start of relationship
    relation_lookup_start = time.time()
    lookupTable = createRelationSemanticsLookupTable(dataLakeTables, FDDict)
    relation_lookup_end = time.time()
    relation_lookup_time = int(relation_lookup_end - relation_lookup_start)
    #genFunc.saveDictionaryAsPickleFile(lookupTable, synthRelationLookupPath)
    save_dict_to_json(lookupTable, synthRelationLookupPath)
    relation_kb_start = time.time()
    synthRelationKB, relation_inverted_index = createRelationSemanticsSynthKB(lookupTable, dataLakeTables, FDDict)
    lookupTable = {}
    relation_kb_end = time.time()
    relation_kb_time = int(relation_kb_end - relation_kb_start)
    relation_total_time = relation_kb_time + relation_lookup_time
    #genFunc.saveDictionaryAsPickleFile(relation_inverted_index, synthRelationInvertedIndexPath)
    save_dict_to_json(relation_inverted_index, synthRelationInvertedIndexPath)
    count = 0
    # with open(synthRelationInvertedIndexCSVPath,"w",newline='', encoding="utf-8") as f:
    #   w = csv.writer(f)
    #   for k, v in relation_inverted_index.items():
    #     w.writerow([k,v])
    #     count += 1
    #     if count > 100:
    #         break
    relation_inverted_index = {}
    #genFunc.saveDictionaryAsPickleFile(synthRelationKB, synthRelationKBPath)
    save_dict_to_json(synthRelationKB, synthRelationKBPath)
    #end of the relationship processing
    
    
    timeDict["relation lookup time"] = relation_lookup_time
    timeDict["relation kb time"] = relation_kb_time
    timeDict["relation total time"] = relation_total_time
    
    
    print("Time taken:")
    print("---------------")
    print("Type lookup time", type_lookup_time)
    print("Type kb time", type_kb_time)
    print("Type total time = ", type_total_time)
    
    print("Relation lookup time", relation_lookup_time)
    print("Relation kb time", relation_kb_time)
    print("Relation total time = ", relation_total_time)
    
    # with open(synthTimeTakenPath,"w",newline='', encoding="utf-8") as f:
    #   w = csv.writer(f)
    #   for k, v in timeDict.items():
    #     w.writerow([k,v])
    later = time.time()
    difference = int(later - now)
    print("Time taken to process and save the inverted indices in seconds:", difference)
    
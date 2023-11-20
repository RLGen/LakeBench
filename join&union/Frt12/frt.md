<div>
    <h1>Frt12</h1>
</div>

<h2>Quick Start</h2>

## Step1: offline

### input

① `Yago_ File_ Path` (a dictionary that stores entity labels)

② `Candidate_Folder_Path` (candidate table folder path)

③ `Query_Folder_Path` (query table folder path)

### output

① Location:~/offline_ All/

② Result:
1) `Candidate_Label_Value.pkl` (label weight of all candidate tables)
2) `File_Names.pkl` (file names for all candidate tables)
3) `Entity_Label.pkl` (entity label file for all candidate tables)
4) `Candidate_Attributes_List.pkl` (attribute list of all candidate tables)
5) `Can_Entity_List.pkl` (a list of entities for all candidate tables)
6) `One_ATT_Freq.pkl` (frequency of each attribute appearing in all candidate tables)
7) `Two_ATT_Freq.pkl` (the frequency at which two attributes that appear in any candidate table appear in all candidate tables)

## Step2: online

### union search

(1) File location:~/online_ Process/all_ Entity_ Completion. py

(2) Input:
① `Query_Folder_Path` (path to the folder where the query table is stored)

② `Query_Label_Path` (label for storing query tables - path to the physical folder)

③ `Candidate_Label_Value.pkl` (label weight of all candidate tables)

④ `File_Names.pkl` (file names for all candidate tables)

⑤ `Entity_Label.pkl` (entity label file for all candidate tables)

⑥ `Candidate_Attributes_List.pkl` (attribute list of all candidate tables)

⑦ `Can_Entity_List.pkl` (a list of entities for all candidate tables)

(3) Output:
① Location:`~/output/opendata_Output_Union`

② Result: Output a table

Note that the table contains the name of each candidate table and the union table name of the top k found in the query

### join search

(1) File location:~/online_Process/all_Schema_Completion. py

(2) Input:
① `Query_Folder_Path` (path to the folder where the query table is stored)

② `Query_Label_Path` (label for storing query tables - path to the physical folder)

③ `One_ATT_Freq.pkl` (frequency of each attribute appearing in all candidate tables)

④ `Two_ATT_Freq.pkl` (the frequency at which two attributes that appear in any candidate table appear in all candidate tables)

⑤ `Candidate_Label_Value.pkl` (label weight of all candidate tables)

⑥ `File_Names.pkl` (file names for all candidate tables)

⑦ `Can_Entity_List.pkl` (a list of entities for all candidate tables)

(3) Output:
① Location:`~/output/opendata_Output_Join`

② Result: Output a table

Note that name of each candidate table+name of the queried top k table for joining+column name for joining



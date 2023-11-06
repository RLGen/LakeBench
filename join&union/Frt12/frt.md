<div>
    <h1>Frt12</h1>
</div>

<h2>Quick Start</h2>

**Step1: offline**

***input***

① Yago_ File_ Path (a dictionary that stores entity labels)
② Candidate_ Folder_ Path (candidate table folder path)
③ Query_ Folder_ Path (query table folder path)

***output***

① Location:~/offline_ All/
② Result:
1) Candidate_ Label_ Value.pkl (label weight of all candidate tables)
2) File_ Names. pkl (file names for all candidate tables)
3) Entity_ Label.pkl (entity label file for all candidate tables)
4) Candidate_ Attributes_ List.pkl (attribute list of all candidate tables)
5) Can_ Entity_ List.pkl (a list of entities for all candidate tables)
6) One_ ATT_ Freq.pkl (frequency of each attribute appearing in all candidate tables)
7) Two_ ATT_ Freq.pkl (the frequency at which two attributes that appear in any candidate table appear in all candidate tables)

**Step2: online**

***union***

(1) File location:~/online_ Process/all_ Entity_ Completion. py

(2) Input:
① Query_ Folder_ Path (path to the folder where the query table is stored)
② Query_ Label_ Path (label for storing query tables - path to the physical folder)
③ Candidate_ Label_ Value.pkl (label weight of all candidate tables)
④ File_ Names. pkl (file names for all candidate tables)
⑤ Entity_ Label.pkl (entity label file for all candidate tables)
⑥ Candidate_ Attributes_ List.pkl (attribute list of all candidate tables)
⑦ Can_ Entity_ List.pkl (a list of entities for all candidate tables)

(3) Output:
① Location:~/output/opendata_ Output_ Union
② Result: Output a table

Note that the table contains the name of each candidate table and the union table name of the top k found in the query

***join***

(1) File location:~/online_ Process/all_ Schema_ Completion. py

(2) Input:
① Query_ Folder_ Path (path to the folder where the query table is stored)
② Query_ Label_ Path (label for storing query tables - path to the physical folder)
③ One_ ATT_ Freq.pkl (frequency of each attribute appearing in all candidate tables)
④ Two_ ATT_ Freq.pkl (the frequency at which two attributes that appear in any candidate table appear in all candidate tables)
⑤ Candidate_ Label_ Value.pkl (label weight of all candidate tables)
⑥ File_ Names. pkl (file names for all candidate tables)
⑦ Can_ Entity_ List.pkl (a list of entities for all candidate tables)

(3) Output:
① Location:~/output/opendata_ Output_ Join
② Result: Output a table

Note that name of each candidate table+name of the queried top k table for joining+column name for joining



# infogather
Paper：InfoGather- Entity Augmentation and Attribute Discovery By Holistic Matching with Web Tables   code for join and union



# offline:
python join_creatofflineIndex_webtable_opendata.py 
location：13022 /home/lijiajun/infogather/

script: join_creatofflineIndex_webtable_opendata.py  
run commond: python join_creatofflineIndex_webtable_opendata.py
* parameters:  
    * 1 datalist: list, dataset list   
    * 2 indexstorepath: string, the path of storing index  
    * 3 columnValue_rate: float, the columnValue importance of the column  
    * 4  columnName_rate :  float, the columnName importance of the column  
    * 5 columnWith_rate : float, the columnWith importance of the column  
    * 6 dataset_large_or_small: sting , large or small  
    * 7 num_walks: int, the superparameter of ppr  
    * 8 reset_pro: float,the superparameter of ppr  


# online:  

location：13022  /home/lijiajun/infogather    
script: join_queryonline_opendata.py/join_queryonline_webtable.py    

run commond: python join_creatofflineIndex_webtable_opendata.py  

* parameters  
  * 1 queryfilepath:string the querytablefilepath
  * 2 columnname: the query column  

# get topk:  

topk: join_creat_topkfile.py/join_creat_topkfile.py  
location：13022  /home/lijiajun/infogather  
script: python join_creat_topkfile.py  
run commond: python join_creatofflineIndex_webtable_opendata.py  
* parameters：  
  * 1 filepath: string,the index of final_res_dic.pkl filepath  
  * 2 storepath: string, the result of topk file store path  


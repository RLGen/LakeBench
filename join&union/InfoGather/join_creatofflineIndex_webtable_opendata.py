"""

this file is used tp compute the offline of webtable set and opendata

"""

from infogether.join_offline_create_index import creatInfogatherOffineIndex

# -------------------------------------create opendata offline index-----------------------------------------


filepath = "/data_ssd/opendata/small/"
queryfilepath = "/data_ssd/opendata/small/query/"
indexstorepath = "/data/lijiajun/infogather/opendata/index/"
datalist = [filepath, queryfilepath]
creatInfogatherOffineIndex(datalist,indexstorepath,
                               columnValue_rate=0.333,columnName_rate=0.3333,columnWith_rate=0.3333,
                               similar_thres= 0.7,values_maxlen_inverse = 200,dataset_large_or_small = "large",
                               num_walks = 300,reset_pro = 0.3)


# -------------------------------------creat webtable offline index-----------------------------------------
filepath = "/data_ssd/webtable/large/split_1"
queryfilepath = "/data_ssd/webtable/small_query/query"
datalist = [filepath, queryfilepath]
# store index filepath
indexstorepath = "/data/lijiajun/infogather/webtables/index/"
creatInfogatherOffineIndex(datalist,indexstorepath,
                               columnValue_rate=0.333,columnName_rate=0.3333,columnWith_rate=0.3333,
                               similar_thres= 0.7,values_maxlen_inverse = 200,dataset_large_or_small = "large",
                               num_walks = 300,reset_pro = 0.3)





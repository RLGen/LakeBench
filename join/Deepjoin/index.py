from sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import os
import pickle
import time
import sys
import argparse
from tqdm import tqdm
torch.cuda.empty_cache()
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def extractVectors(dfs, ds_path, dataFolder, augment, sample, table_order, run_id, singleCol=False):
    if singleCol:
        model_path = "/home/benchmark/starmie-main/" \
                     "model_%s_%s_%s_%dsingleCol.pt" % (augment, sample, table_order, run_id)
    else:
        model_path = "/home/benchmark/starmie-main/" \
                     "model_%s_%s_%s_%d.pt" % (augment, sample, table_order, run_id)
    print(f"model_path: {model_path}")
    ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    # load_checkpoint from sdd/pretain

    model, trainset = load_checkpoint(ckpt, ds_path)

    return inference_on_tables(dfs, model, trainset, batch_size=528)


def get_df(dataFolder):
    '''
    Get the DataFrames of each table in a folder
    Args:
        dataFolder: filepath to the folder with all tables
    Return:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    '''

    # dataFiles = glob.glob(dataFolder+"/*.csv")
    dataFiles = os.listdir(dataFolder)

    dataDFs = {}
    columns = 0

    ind = 0
    for file in sorted(dataFiles):
        ind += 1
        if file == "CSV0000000000000435.csv": continue
        df = pd.read_csv(os.path.join(dataFolder, file), nrows=1000, encoding="ISO-8859-1", low_memory=False, lineterminator='\n')
        # if len(df) > 1000:
        #     # get first 1000 rows
        #     df = df.head(1000)
        filename = file.split("/")[-1]
        dataDFs[filename] = df

        print(f"ind: {ind}, file: {file}, columns: {df.shape[1]}")
        columns += df.shape[1]

    return dataDFs, columns


if __name__ == '__main__':
    ''' Get the model features by calling model inference from sdd/pretrain
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="webtable")
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true", default=False)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--table_order", type=str, default='column')  # column-ordered or row-ordered (always use column)
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)

    hp = parser.parse_args()
    print(f'hp.save_model: {hp.save_model}, hp.single_column: {hp.single_column}')

    # START PARAMETER: defining the benchmark (dataFolder), if it is a single column baseline,
    # run_id, table_order, and augmentation operators and sampling method if they are different from default
    dataFolder = hp.benchmark
    isSingleCol = hp.single_column
    if dataFolder == 'webtable':
        ao = 'drop_col'
        sm = 'tfidf_entity'
        if isSingleCol:
            ao = 'drop_cell'
    ao = 'drop_cell'
    sm = 'alphaHead'

    run_id = hp.run_id
    table_order = hp.table_order
    # END PARAMETER

    # Change the data paths to where the benchmarks are stored
    if dataFolder == 'opendata':
        DATAPATH = "/data/opendata/small/"
        dataDir = ['datasets_UK', 'datasets_SG', 'datasets_CAN', 'datasets_USA']
    elif dataFolder == 'opendata_large':
        DATAPATH = "/data/opendata/large/"
        dataDir = ['UK', 'SG', 'CAN', 'USA']
    elif dataFolder == 'webtable':
        DATAPATH = '/data/webtable/small/'
        dataDir = ['split_1']
    elif dataFolder == 'webtable_large':
        DATAPATH = '/data/webtable/large/'
        dataDir = ['split_1', 'split_2', 'split_3', 'split_4', 'split_5', 'split_6']
    elif dataFolder == 'webtable_small_query':
        DATAPATH = "/data/webtable/small/"
        dataDir = ['small_query']
    elif dataFolder == 'webtable_large_query':
        DATAPATH = "/data/webtable/large/"
        dataDir = ['large_query']
    elif dataFolder == 'opendata_small_query':
        DATAPATH = "/data/webtable/small/"
        dataDir = ['small_query']
    elif dataFolder == 'opendata_large_query':
        DATAPATH = "/data/webtable/large/"
        dataDir = ['large_query']

    inference_times = 0
    # dataDir is the query and data lake
    for dir in tqdm(dataDir):
        print("//==== ", dir)
        DATAFOLDER = DATAPATH + dir
        print(f"Datafloder: {DATAFOLDER}")

        dfs, col_count = get_df(DATAFOLDER)

        dataEmbeds = []
        dfs_totalCount = len(dfs)
        print(f"The number of total tables {dfs_totalCount} in {dir}")
        print(f"The number of total columns {col_count} in {dir}")

        # Extract model vectors, and measure model inference time
        start_time = time.time()
        cl_features = extractVectors(list(dfs.values()), DATAPATH, dataFolder, ao, sm, table_order, run_id, singleCol=isSingleCol)
        inference_times += time.time() - start_time
        print("%s %s inference time: %d seconds" % (dataFolder, dir, time.time() - start_time))
        for i, file in enumerate(dfs):
            # get features for this file / dataset
            cl_features_file = np.array(cl_features[i])
            dataEmbeds.append((file, cl_features_file))

        
        saveDir = dir

        if isSingleCol:
            # output_path = "/data/data/yuanqin/results/opendata/small/vectors_cl_%s_%s_%s_%s_%d_singleCol.pkl" % (
            output_path = "/data/final_result/starmie/webtable/webtable_%s.pkl" % (
                saveDir)
        else:
            output_path = "/data/final_result/starmie/webtable/webtable_%s.pkl" % (
                saveDir)
        if hp.save_model:
            pickle.dump(dataEmbeds, open(output_path, "wb"))
        print("Benchmark: ", dataFolder)
        print(f"output: {output_path}")
        print("--- Total Inference Time: %s seconds ---" % (inference_times))

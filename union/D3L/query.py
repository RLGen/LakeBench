import numpy as np
import pandas as pd
import csv
import os
import multiprocessing
from multiprocessing import Process, Queue
from tqdm import tqdm
import time
import argparse
import glob
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from d3l.indexing.feature_extraction.values.glove_embedding_transformer import GloveTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

def split_list(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg
        result.append(lst[start:end])
        start = end

    return result

def sub_process(query_tables, queue, output_folder, qe, dataloader, k_value):
    for i, table_name_with_extension in enumerate(query_tables):

        table_name = os.path.splitext(table_name_with_extension)[0]
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        results_file = os.path.join(output_folder, f"{table_name}.csv")

        if os.path.exists(results_file):
            print(f"Skip query table {i + 1}. Because the result file already exists:{results_file}")
            queue.put(1)  
            continue

        # querying
        results, extended_results = qe.table_query(table=dataloader.read_table(table_name=table_name),
                                                   aggregator=None, k=k_value, verbose=True)

        with open(results_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(['query_table', 'candidate_table', 'query_col_name', 'candidate_col_name'])

            for result in extended_results:
                query_table = table_name
                candidate_table = os.path.basename(result[0])

                for x, column_info in enumerate(result[1]):
                    column_info_name = f"column_info{x+1}"
                    globals()[column_info_name] = column_info[0]

                    row = [f"{query_table}.csv", f"{candidate_table}.csv", column_info[0][0], column_info[0][1]]
                    writer.writerow(row)

        print(f"Results for query table {i + 1} have been written to {results_file}")
        queue.put(1)
    queue.put((-1, "test-pid"))

def merge_csv(output_folder, combined_file_path):
    csv_files = glob.glob(output_folder + "/*.csv")

    combined_data = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    combined_data.to_csv(combined_file_path, index=False)
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Description of your program')

# Add arguments
parser.add_argument('--output_folder', type=str, help='Path to the output folder')
parser.add_argument('--combined_file_path', type=str, help='Path to the combined file')
parser.add_argument('--k', type=int, help='Value for k')
parser.add_argument('--split_num', type=int, default=1, help='Value for split_num')
parser.add_argument('--query_tables_folder', type=str, help='Path to query tables folder')
parser.add_argument('--name_index_file', type=str, help='Name index file')
parser.add_argument('--format_index_file', type=str, help='Format index file')
parser.add_argument('--value_index_file', type=str, help='Value index file')
parser.add_argument('--embedding_index_file', type=str, help='Embedding index file')
parser.add_argument('--distribution_index_file', type=str, help='Distribution index file')

# Parse command line arguments
args = parser.parse_args()

# Use the args
output_folder = args.output_folder
combined_file_path = args.combined_file_path
k = args.k
query_tables_folder = args.query_tables_folder
root_path = query_tables_folder
split_num = args.split_num
name_index_file = args.name_index_file
format_index_file = args.format_index_file
value_index_file = args.value_index_file
embedding_index_file = args.embedding_index_file
distribution_index_file = args.distribution_index_file

# CSV data loader
dataloader = CSVDataLoader(root_path=root_path, sep=',')
print("LOADED!")

# Load index files from disk if they have been created
name_index = unpickle_python_object(name_index_file)
print("Name Unpickled!")
format_index = unpickle_python_object(format_index_file)
print("Format Unpickled!")
value_index = unpickle_python_object(value_index_file)
print("Value Unpickled!")
embedding_index = unpickle_python_object(embedding_index_file)
print("Embedding Unpickled!")
distribution_index = unpickle_python_object(distribution_index_file)
print("Distribution Unpickled!")
print("Index LOADED!")

if __name__ == "__main__":
    # Record start time
    start_time = time.time()

    # Get all the file names from the folder
    query_tables = [f for f in os.listdir(query_tables_folder) if f.endswith(".csv")]

    # Create QueryEngine instance
    qe = QueryEngine(name_index, format_index, value_index, embedding_index, distribution_index)

    sub_file_ls = split_list(query_tables, split_num)
    process_list = []

    # Create a queue for each process
    queues = [Queue() for i in range(split_num)]
    finished = [False for i in range(split_num)]

    # Create a progress bar for each process
    bars = [tqdm(total=len(sub_file_ls[i]), desc=f"bar-{i}", position=i) for i in range(split_num)]
    results = [None for i in range(split_num)]

    for i in range(split_num):
        process = Process(target=sub_process, args=(sub_file_ls[i], queues[i], output_folder, qe, dataloader, k))
        process_list.append(process)
        process.start()

    while True:
        for i in range(split_num):
            queue = queues[i]
            bar = bars[i]
            try:
                res = queue.get_nowait()
                if isinstance(res, tuple) and res[0] == -1:
                    finished[i] = True
                    results[i] = res[1]
                    continue
                bar.update(res)
            except Exception as e:
                continue

        if all(finished):
            break

    for process in process_list:
        process.join()

    
    merge_csv(output_folder, combined_file_path)

    # Record end time
    end_time = time.time()

    # Calculate run time in seconds
    run_time = end_time - start_time

    # Print the run time
    print(f"Running Time:{run_time} s")

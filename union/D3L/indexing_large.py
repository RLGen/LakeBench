import psutil
import time
import numpy as np

from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex

from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader

from d3l.querying.query_engine import QueryEngine

from d3l.utils.functions import pickle_python_object, unpickle_python_object

from d3l.indexing.feature_extraction.values.glove_embedding_transformer import GloveTransformer

from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss

if __name__ == "__main__":
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # Your code here
    dataloader = CSVDataLoader(
        root_path='/data_ssd/webtable/large',
        sep=','
    )

    print("Loaded!")
    # Create new indexes
    
    name_index = NameIndex(dataloader=dataloader)
    pickle_python_object(name_index,'./name_webtable_large_test.lsh')
    print("Name: SAVED!")

    format_index = FormatIndex(dataloader=dataloader)
    pickle_python_object(format_index,'./format_webtable_large_test.lsh')
    print("Format: SAVED!")


    value_index = ValueIndex(dataloader=dataloader)
    pickle_python_object(value_index, './value_webtable_large_test.lsh')
    print("Value: SAVED!")



    embedding_index = EmbeddingIndex(dataloader=dataloader,index_cache_dir="./")
    pickle_python_object(embedding_index, './embedding_webtable_large_test.lsh')
    print("Embedding: SAVED!")


    distribution_index = DistributionIndex(dataloader=dataloader)
    pickle_python_object(distribution_index, './distribution_webtable_large_test.lsh')
    print("Distribution: SAVED!")







    end_memory = get_memory_usage()
    end_time = time.time()

    memory_used = end_memory - start_memory
    elapsed_time = end_time - start_time

    print(f"Memory used: {memory_used} bytes")
    print(f"Elapsed time: {elapsed_time} seconds")



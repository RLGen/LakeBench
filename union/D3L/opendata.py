import numpy as np
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from d3l.indexing.feature_extraction.values.glove_embedding_transformer import GloveTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

from memory_profiler import profile

# CSV data loader
dataloader = CSVDataLoader(
    root_path='G:/large/data_ssd/opendata/small/datasets_USA',
    sep=','
)

@profile
def main():
    # Create new indexes
    name_index = NameIndex(dataloader=dataloader)
    pickle_python_object(name_index, './name_open.lsh')
    print("Name: SAVED!")

    format_index = FormatIndex(dataloader=dataloader)
    pickle_python_object(format_index, './format_open.lsh')
    print("Format: SAVED!")

    value_index = ValueIndex(dataloader=dataloader)
    pickle_python_object(value_index, './value_open.lsh')
    print("Value: SAVED!")

    embedding_index = EmbeddingIndex(dataloader=dataloader)
    pickle_python_object(embedding_index, './embedding_open.lsh')
    print("Embedding: SAVED!")

    distribution_index = DistributionIndex(dataloader=dataloader)
    pickle_python_object(distribution_index, './distribution_open.lsh')
    print("Distribution: SAVED!")

if __name__ == "__main__":
    main()

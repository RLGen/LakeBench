import psutil
import time
import numpy as np
import argparse
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex
from d3l.input_output.dataloaders import PostgresDataLoader, CSVDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object
from d3l.indexing.feature_extraction.values.glove_embedding_transformer import GloveTransformer
from d3l.indexing.feature_extraction.values.fasttext_embedding_transformer import FasttextTransformer

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Process some integers.')

# Add arguments to the parser
parser.add_argument('--root_path', help='Root path of the data')
parser.add_argument('--name_file', help='Path to save name index')
parser.add_argument('--format_file', help='Path to save format index')
parser.add_argument('--value_file', help='Path to save value index')
parser.add_argument('--embedding_file', help='Path to save embedding index')
parser.add_argument('--distribution_file', help='Path to save distribution index')

# Parse the arguments
args = parser.parse_args()

root_path = args.root_path
name_file = args.name_file
format_file = args.format_file
value_file = args.value_file
embedding_file = args.embedding_file
distribution_file = args.distribution_file

dataloader = CSVDataLoader(root_path=root_path, sep=',')

print("Loaded!")
# Create new indexes
name_index = NameIndex(dataloader=dataloader)
pickle_python_object(name_index, name_file)
print("Name: SAVED!")

format_index = FormatIndex(dataloader=dataloader)
pickle_python_object(format_index, format_file)
print("Format: SAVED!")

value_index = ValueIndex(dataloader=dataloader)
pickle_python_object(value_index, value_file)
print("Value: SAVED!")

embedding_index = EmbeddingIndex(dataloader=dataloader, index_cache_dir="./")
pickle_python_object(embedding_index, embedding_file)
print("Embedding: SAVED!")

distribution_index = DistributionIndex(dataloader=dataloader)
pickle_python_object(distribution_index, distribution_file)
print("Distribution: SAVED!")

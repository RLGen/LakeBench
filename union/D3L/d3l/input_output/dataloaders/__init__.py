"""
Copyright (C) 2021 Alex Bogatu.
This file is part of the D3L Data Discovery Framework.
Notes
-----
This module exposes data reading functionality.
"""
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Union, Dict, Any
from d3l.input_output.dataloaders.typing import DBType

import pandas as pd
import sqlalchemy
import os


class DataLoader(ABC):
    @abstractmethod
    def get_counts(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Reads the non-null and distinct cardinality of each column of a table.

        Parameters
        ----------
        table_name : str
            The table name.
        schema_name : Optional[str]
            The name of the schema if needed for database loaders.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            A collection of tuples of non-null and distinct cardinalities.

        """
        pass

    @abstractmethod
    def get_columns(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve the column names of the given table.

        Parameters
        ----------
        table_name : str
            The table name.
        schema_name : Optional[str]
            The name of the schema if needed for database loaders.

        Returns
        -------
        List[str]
            A collection of column names as strings.

        """
        pass

    @abstractmethod
    def get_tables(
        self,
        schema_name: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve all the table names existing under the given root.

        Parameters
        ----------
        root_name : str
            The name of the schema if needed for database loaders.

        Returns
        -------
        List[str]
            A list of table names.
        """
        pass

    @abstractmethod
    def read_table(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
        table_columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Read the table data into a pandas DataFrame.

        Parameters
        ----------
        table_name : str
            The table name.
        schema_name : Optional[str]
            The name of the schema if needed for database loaders.
        table_columns : Optional[List[str]]
            A list of columns to be read.
        chunk_size : int
            The number of rows to read at one time.
            If None then the full table is returned.

        Returns
        -------
        Union[pd.DataFrame, Iterator[pd.DataFrame]]
            The entire table data or a Dataframe with *chunksize* rows.

        """
        pass


class PostgresDataLoader(DataLoader):
    def __init__(
        self,
        db_name: str,
        db_host: Optional[str] = None,
        db_port: Optional[int] = None,
        db_username: Optional[str] = None,
        db_password: Optional[str] = None,
    ):
        """
        The main data reading object.

        Parameters
        ----------
        db_name : str
            The database name (or path if sqlite is used) where the given tables are stored.
        db_host : str
            The IP address/host name of the database server.
        db_port : str
            The port number of the database server.
        db_username : str
            The username used to connect to the database server.
        db_password : str
            The password used to connect to the database server.
        """
        self.db_type = DBType.POSTGRESQL

        self.__db_host = db_host
        self.__db_port = db_port
        self.__db_username = db_username
        self.__db_password = db_password
        self.__db_name = db_name

    def get_db_engine(self) -> Optional[sqlalchemy.engine.Engine]:
        """
        This method creates a new SQLAlchemy engine useful to create database connections.

        Returns
        -------
        sqlalchemy.engine.Engine
            A new SQLAlchemy connection engine.
            Returns None if the database type is unrecognised.

        """
        return sqlalchemy.create_engine(
            "postgresql://{}:{}@{}:{}/{}".format(
                self.__db_username,
                self.__db_password,
                self.__db_host,
                self.__db_port,
                self.__db_name,
            )
        )

    def get_counts(
        self, table_name: str, schema_name: Optional[str] = None
    ) -> Dict[str, Tuple[int, int]]:
        """
        Reads the non-null and distinct cardinality of each column of a table.

        Parameters
        ----------
        table_name : str
            The table name.
        schema_name : Optional[str]
            The name of the schema if needed for database loaders.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            A collection of tuples of non-null and distinct cardinalities.

        """

        assert (
            schema_name is not None
        ), "The schema name must be provided for postgres databases!"

        db_engine = self.get_db_engine()
        metadata = sqlalchemy.MetaData(schema=schema_name)
        table_object = sqlalchemy.Table(
            table_name, metadata, autoload=True, autoload_with=db_engine
        )

        count_queries = []
        for column in table_object.c:
            count_queries.extend(
                [
                    sqlalchemy.func.count(column).label(
                        str(column.name) + "_count_not_null"
                    ),
                    sqlalchemy.func.count(sqlalchemy.distinct(column)).label(
                        str(column.name) + "_count_distinct"
                    ),
                ]
            )

        query_object = sqlalchemy.select(count_queries)
        column_counts = pd.read_sql(query_object, db_engine)
        column_counts = {
            str(c.name): (
                column_counts[str(c.name) + "_count_not_null"].item(),
                column_counts[str(c.name) + "_count_distinct"].item(),
            )
            for c in table_object.c
        }
        return column_counts

    def get_columns(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve the column names of the given table.

        Parameters
        ----------
        table_name : str
            The table name.
        schema_name : Optional[str]
            The name of the schema if needed for database loaders.

        Returns
        -------
        List[str]
            A collection of column names as strings.

        """
        assert (
            schema_name is not None
        ), "The schema name must be provided for postgres databases!"

        db_engine = self.get_db_engine()
        metadata = sqlalchemy.MetaData(schema=schema_name)
        table_object = sqlalchemy.Table(
            table_name, metadata, autoload=True, autoload_with=db_engine
        )

        column_names = [str(column) for column in table_object.c]
        return column_names

    def get_tables(
        self,
        schema_name: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve all the table names existing under the given root.

        Parameters
        ----------
        schema_name : Optional[str]
            The name of the schema if needed for database loaders.

        Returns
        -------
        List[str]
            A list of table names. Each name will have the form <schema_name>.<table_name>.

        """

        assert (
            schema_name is not None
        ), "The schema name must be provided for postgres databases!"

        db_engine = self.get_db_engine()
        metadata = sqlalchemy.MetaData()
        pg_tables = sqlalchemy.Table(
            "pg_tables", metadata, autoload=True, autoload_with=db_engine
        )

        query_tables = sqlalchemy.select(
            [pg_tables.columns.schemaname, pg_tables.columns.tablename]
        ).where((pg_tables.columns.schemaname == schema_name))

        results_of_tables = pd.read_sql(query_tables, db_engine)
        results_of_tables = [
            (str(row["schemaname"]) + "." + str(row["tablename"]))
            for _, row in results_of_tables.iterrows()
        ]
        return results_of_tables

    def read_table(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
        table_columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Read the table data into a pandas DataFrame.

        Parameters
        ----------
        table_name : str
            The table name.
        schema_name : Optional[str]
            The name of the schema if needed for database loaders.
        table_columns : Optional[List[str]]
            A list of columns to be read.
        chunk_size : int
            The number of rows to read at one time.
            If None then the full table is returned.

        Returns
        -------
        Union[pd.DataFrame, Iterator[pd.DataFrame]]
            The entire table data or a Dataframe with *chunksize* rows.

        """

        assert (
            schema_name is not None
        ), "The schema name must be provided for postgres databases!"

        db_engine = self.get_db_engine()
        metadata = sqlalchemy.MetaData(schema=schema_name)
        table_object = sqlalchemy.Table(
            table_name, metadata, autoload=True, autoload_with=db_engine
        )
        if table_columns is None:
            col_list = [table_object]
        elif isinstance(table_columns, list):
            col_list = [col for col in table_object.c if col.name in table_columns]
        else:
            raise ValueError(
                "Expected db_columns of type list but got {}.".format(
                    type(table_columns)
                )
            )
        query_object = sqlalchemy.select(col_list)
        return pd.read_sql(query_object, db_engine, chunksize=chunk_size)


class CSVDataLoader(DataLoader):
    def __init__(self, root_path: str, **loading_kwargs: Any):
        """
        Create a new CSV file loader instance.
        Parameters
        ----------
        root_path : str
            A locally existing directory where all CSV files can be found.
        loading_kwargs : Any
            Pandas-specific CSV reading arguments.
            Note that all CSV file in the given root directory are expected to have the same formatting details,
            e.g., separator, encoding, etc.
        """

        if not os.path.isdir(root_path):
            raise FileNotFoundError(
                "The {} root directory was not found locally. "
                "A CSV loader must have an existing directory associated!".format(
                    root_path
                )
            )

        self.root_path = root_path
        if self.root_path[-1] != "/":
            self.root_path = self.root_path + "/"
        self.loading_kwargs = loading_kwargs

    def get_counts(
        self, table_name: str, schema_name: Optional[str] = None
    ) -> Dict[str, Tuple[int, int]]:
        """
        Reads the non-null and distinct cardinality of each column of a table.

        Parameters
        ----------
        table_name : str
            The table (i.e, file) name without the parent directory path and *without* the CSV extension.
        schema_name : Optional[str]
            This is ignored for file loaders.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            A collection of tuples of non-null and distinct cardinalities.

        """

        file_path = self.root_path + table_name + ".csv"
        print(file_path)
        data_df = pd.read_csv(file_path, **self.loading_kwargs)
        column_counts = {
            str(col): (data_df[col].count(), data_df[col].nunique())
            for col in data_df.columns
        }
        return column_counts

    def get_columns(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve the column names of the given table.

        Parameters
        ----------
        table_name : str
            The table (i.e, file) name without the parent directory path and *without* the CSV extension.
        schema_name : Optional[str]
           This is ignored for file loaders.

        Returns
        -------
        List[str]
            A collection of column names as strings.

        """
        file_path = self.root_path + table_name + ".csv"
        data_df = pd.read_csv(file_path, nrows=1, **self.loading_kwargs)
        return data_df.columns.tolist()

    def get_tables(
        self,
        schema_name: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve all the table names existing under the given root.

        Parameters
        ----------
        schema_name : Optional[str]
            This is ignored for file loaders.

        Returns
        -------
        List[str]
            A list of table (i.e., file) names that *do not* include full paths or file extensions.

        """
        result_of_tables = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.lower().endswith(".csv"):
                    file_path = os.path.join(root, file)
                    file_rel_path = os.path.relpath(file_path, self.root_path)
                    table_name = os.path.splitext(file_rel_path)[0]
                    result_of_tables.append(table_name)
        
        return result_of_tables
    
    def get_tables_new(
        self, split_num, no,
        schema_name: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve all the table names existing under the given root.

        Parameters
        ----------
        schema_name : Optional[str]
            This is ignored for file loaders.

        Returns
        -------
        List[str]
            A list of table (i.e., file) names that *do not* include full paths or file extensions.

        """
        result_of_tables = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.lower().endswith(".csv"):
                    file_path = os.path.join(root, file)
                    file_rel_path = os.path.relpath(file_path, self.root_path)
                    table_name = os.path.splitext(file_rel_path)[0]
                    result_of_tables.append(table_name)
        sub_sets_files = split_list(result_of_tables, split_num)[no]
        return sub_sets_files




        #result_of_tables = [
        #    ".".join(f.split(".")[:-1])
        #    for f in os.listdir(self.root_path)
        #    if str(f)[-4:].lower() == ".csv"
        #]
        #return result_of_tables

    def read_table(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
        table_columns: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        """
        Read the table data into a pandas DataFrame.

        Parameters
        ----------
        table_name : str
            The table (i.e, file) name without the parent directory path and *without* the CSV extension.
        schema_name : Optional[str]
            This is ignored for file loaders.
        table_columns : Optional[List[str]]
            A list of columns to be read.
        chunk_size : int
            The number of rows to read at one time.
            If None then the full table is returned.

        Returns
        -------
        Union[pd.DataFrame, Iterator[pd.DataFrame]]
            The entire table data or a Dataframe with *chunksize* rows.

        """

        file_path = self.root_path + table_name + ".csv"
        try:
            if table_columns is not None:
                return pd.read_csv(
                    file_path,
                    usecols=table_columns,
                    chunksize=chunk_size,
                    low_memory=False,
                    lineterminator='\n',
                    # error_bad_lines=False, # Deprecated in future versions
                    # warn_bad_lines=False, # Deprecated in future versions
                    **self.loading_kwargs
                )
            return pd.read_csv(
                file_path,
                chunksize=chunk_size,
                low_memory=False,
                lineterminator='\n',
                # error_bad_lines=False, # Deprecated in future versions
                # warn_bad_lines=False, # Deprecated in future versions
                **self.loading_kwargs
            )
        except Exception as e:
            # 异常发生时，记录文件名并输出
            print(f"Error reading table: {table_name}")
            print(f"Error message: {str(e)}")




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
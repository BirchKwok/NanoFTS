from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union
import numpy as np

from .base import BaseIndex

class DocumentInserter:
    """Document inserter, responsible for handling bulk document insertion"""
    
    def __init__(self, 
                 index: BaseIndex,
                 num_workers: int = 8,
                 batch_size: int = 10000,
                 shard_size: int = 500_000):
        """
        Initialize the document inserter
        
        Args:
            index (BaseIndex): The index instance
            num_workers (int): The number of worker threads for parallel processing
            batch_size (int): The batch size
            shard_size (int): The shard size
        """
        self.index = index
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shard_size = shard_size
        self._batch_count = 0

    def add_documents(self, 
                     doc_ids: Union[int, List[int]], 
                     docs: Union[Dict[str, Union[str, int, float]], 
                               List[Dict[str, Union[str, int, float]]]]) -> None:
        """
        Bulk add documents
        
        Args:
            doc_ids (int | list): The document ID or ID list
            docs (dict | list): The document content or document list
        """
        # Standardize the input
        if isinstance(doc_ids, int):
            doc_ids = [doc_ids]
            docs = [docs] if isinstance(docs, dict) else docs
        else:
            docs = docs if isinstance(docs, list) else [docs]
        
        if len(doc_ids) != len(docs):
            raise ValueError("The length of the document ID list and the document list must be the same")
        
        # Process large batches of documents in parallel
        total_docs = len(docs)
        chunk_size = max(10000, total_docs // (self.num_workers * 2))
        
        def process_chunk(start_idx: int, chunk_docs: List[dict]) -> None:
            for i, doc in enumerate(chunk_docs):
                self.index.add_terms(doc_ids[start_idx + i], doc)
        
        if total_docs < chunk_size:
            process_chunk(0, docs)
        else:
            chunks = []
            for i in range(0, total_docs, chunk_size):
                chunks.append((i, docs[i:i + chunk_size]))
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                executor.map(lambda x: process_chunk(*x), chunks)
        
        self._batch_count += len(docs)
        
        # Check if merging and saving is needed
        if len(self.index.index_buffer) >= self.index.buffer_size * 4:
            self.index.merge_buffer()
        
        if self._batch_count >= self.batch_size * 4:
            self.flush()

    def update_document(self, 
                       doc_id: int, 
                       fields: Union[Dict[str, Union[str, int, float]], 
                       List[Dict[str, Union[str, int, float]]]]):
        """Update a document in the index
        
        Args:
            doc_id: The document ID to update
            fields: The fields to update
        """
        self.index.update_terms(doc_id, fields)

    def batch_update_document(self, 
                             doc_ids: List[int], 
                             fields: List[Dict[str, Union[str, int, float]]]):
        """批量更新多个文档
        
        Args:
            doc_ids: 文档ID列表
            fields: 文档字段列表，与doc_ids一一对应
        """
        if len(doc_ids) != len(fields):
            raise ValueError("文档ID列表和文档字段列表长度不匹配")
            
        self.index.batch_update_terms(doc_ids, fields)

    def flush(self) -> None:
        """Flush the buffer and save"""
        self.index.merge_buffer()
        if self._batch_count > 0:
            self.index.build_word_index()
            if self.index.index_dir:
                shard_id = self._batch_count // self.shard_size
                self.index.save(incremental=True)
            self._batch_count = 0

    def from_pandas(self, df, id_column=None, text_columns=None):
        """Import data from a pandas DataFrame
        
        Args:
            df (pandas.DataFrame): The pandas DataFrame to import
            id_column (str): The name of the document ID column, if None then use the row index
            text_columns (list): The list of text columns to index, if None then use all string columns
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Use from_pandas requires installing pandas, please execute: pip install nanofts[pandas]")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame object")

        # Get the document ID
        if id_column is None:
            doc_ids = df.index.tolist()
        else:
            doc_ids = df[id_column].tolist()

        # Get the columns to index
        if text_columns is None:
            text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Build the document list
        docs = []
        for _, row in df[text_columns].iterrows():
            doc = {col: str(val) for col, val in row.items() if pd.notna(val)}
            docs.append(doc)

        # Add documents and flush
        self.add_documents(doc_ids, docs)
        self.flush()

    def from_polars(self, df, id_column=None, text_columns=None):
        """Import data from a polars DataFrame
        
        Args:
            df (polars.DataFrame): The polars DataFrame to import
            id_column (str): The name of the document ID column, if None then use the row index
            text_columns (list): The list of text columns to index, if None then use all string columns
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Use from_polars requires installing polars, please execute: pip install nanofts[polars]")

        if not isinstance(df, pl.DataFrame):
            raise TypeError("df must be a polars DataFrame object")

        # Get the document ID
        if id_column is None:
            doc_ids = list(range(len(df)))
        else:
            doc_ids = df[id_column].to_list()

        # Get the columns to index
        if text_columns is None:
            text_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]

        # Build the document list
        docs = []
        for row in df.select(text_columns).iter_rows():
            doc = {col: str(val) for col, val in zip(text_columns, row) if val is not None}
            docs.append(doc)

        # Add documents and flush
        self.add_documents(doc_ids, docs)
        self.flush() 
    
    def from_arrow(self, table, id_column=None, text_columns=None):
        """Import data from an arrow table
        
        Args:
            table (pyarrow.Table): The pyarrow Table object
            id_column (str): The name of the document ID column, if None then use the row index
            text_columns (list): The list of text columns to index, if None then use all string columns
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("Use from_arrow requires installing pyarrow, please execute: pip install nanofts[pyarrow]")

        if not isinstance(table, pa.Table):
            raise TypeError("table must be a pyarrow Table object")

        # Get the columns to index
        if text_columns is None:
            text_columns = [field.name for field in table.schema 
                           if pa.types.is_string(field.type)]
            if id_column and id_column in text_columns:
                text_columns.remove(id_column)

        # Get the document ID
        if id_column is None:
            doc_ids = range(table.num_rows)
        else:
            # Fix: correctly handle the id column
            id_array = table.column(id_column)
            if pa.types.is_integer(id_array.type):
                doc_ids = id_array.to_numpy()
            else:
                # If not an integer type, try to convert
                doc_ids = [int(val.as_py()) for val in id_array]

        # Optimize: use batch processing to handle large tables
        batch_size = 50000
        num_batches = (table.num_rows + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, table.num_rows)
            
            # Use slice to avoid full copy
            batch = table.slice(start_idx, end_idx - start_idx)
            
            # Use Arrow's column access directly
            docs = []
            for i in range(batch.num_rows):
                doc = {}
                for col in text_columns:
                    val = batch.column(col)[i].as_py()
                    if val is not None:
                        doc[col] = str(val)
                docs.append(doc)
            
            # Add this batch of documents
            batch_doc_ids = (doc_ids[start_idx:end_idx] if isinstance(doc_ids, (list, np.ndarray)) 
                            else range(start_idx, end_idx))
            self.add_documents(list(batch_doc_ids), docs)  # Ensure converted to list

        # Flush at the end
        self.flush()

    def from_parquet(self, path, id_column=None, text_columns=None):
        """Import data from a parquet file
        
        Args:
            path (str): The path to the parquet file
            id_column (str): The name of the document ID column, if None then use the row index
            text_columns (list): The list of text columns to index, if None then use all string columns
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("Use from_parquet requires installing pyarrow, please execute: pip install nanofts[pyarrow]")

        # Optimize: use memory mapping to read large files
        table = pq.read_table(path, memory_map=True)
        self.from_arrow(table, id_column, text_columns)

    def from_csv(self, path, id_column=None, text_columns=None):
        """Import data from a csv file
        
        Args:
            path (str): The path to the csv file
            id_column (str): The name of the document ID column, if None then use the row index
            text_columns (list): The list of text columns to index, if None then use all string columns
        """
        try:
            import polars as pl
        except ImportError:
            raise ImportError("Use from_csv requires installing polars, please execute: pip install nanofts[polars]")

        df = pl.read_csv(path)
        self.from_polars(df, id_column, text_columns)  
        
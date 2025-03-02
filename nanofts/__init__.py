import re
import shutil
from typing import List, Union, Dict
from pathlib import Path

from pyroaring import BitMap

from .factory import IndexFactory, IndexType
from .inserter import DocumentInserter


class FullTextSearch:
    def __init__(self, index_dir: str = None, 
                 max_chinese_length: int = 4, 
                 num_workers: int = 8,
                 shard_size: int = 500_000,
                 min_term_length: int = 2,
                 auto_save: bool = True,
                 batch_size: int = 10000,
                 drop_if_exists: bool = False,
                 buffer_size: int = 100000,
                 index_type: str = "inverted",
                 ngram_size: int = 2):
        """
        Initialize the full-text search engine

        Args:
            index_dir (str): The directory to store the index files, if None, use in-memory index
            max_chinese_length (int): The maximum length of Chinese substrings, default is 4
            num_workers (int): The number of worker threads for parallel indexing, default is 8
            shard_size (int): The number of documents per shard, default is 500,000
            min_term_length (int): The minimum length of a term, default is 2
            auto_save (bool): Whether to save to disk automatically, default is True
            batch_size (int): The number of documents to process in each batch, default is 10000
            drop_if_exists (bool): Whether to delete the index files if they exist, default is False
            buffer_size (int): The size of the memory buffer, default is 100000
            index_type (str): The type of index, optional values: inverted, ngram, default is inverted
            ngram_size (int): The size of the n-gram, only used for NGram index, default is 2
        """
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.index_dir = Path(index_dir) if index_dir else None
        
        if drop_if_exists and self.index_dir and self.index_dir.exists():
            shutil.rmtree(self.index_dir)
                
        self.max_chinese_length = max_chinese_length
        self.num_workers = num_workers
        self.shard_size = shard_size
        self.min_term_length = min_term_length
        self.auto_save = auto_save
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        try:
            index_type_enum = IndexType(index_type)
        except ValueError:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        self.inverted_index = IndexFactory.create_index(
            index_type=index_type_enum,
            index_dir=index_dir,
            max_chinese_length=max_chinese_length,
            min_term_length=min_term_length,
            buffer_size=buffer_size,
            ngram_size=ngram_size
        )
        
        self.inserter = DocumentInserter(
            index=self.inverted_index,
            num_workers=num_workers,
            batch_size=batch_size,
            shard_size=shard_size
        )
        
        self._batch_count = 0
        
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.inverted_index.load()

    def add_document(self, doc_id: Union[int, List[int]], 
                    fields: Union[Dict[str, Union[str, int, float]], 
                                List[Dict[str, Union[str, int, float]]]]):
        """Add documents to the index
        
        Args:
            doc_id: The document ID or a list of document IDs
            fields: The fields to index
        """
        self.inserter.add_documents(doc_id, fields)

    def search(self, query: str) -> Union[BitMap, List[tuple[int, float]]]:
        """Search for a query
        
        Args:
            query: The query to search for
        Returns:
            Union[BitMap, List[tuple[int, float]]]: The document ID set or a list of (document ID, similarity)
        """
        return self.inverted_index.search(query)

    def flush(self):
        """Flush the buffer and save to disk"""
        self.inserter.flush()

    def remove_document(self, doc_id: int):
        """Remove a document from the index
        
        Args:
            doc_id: The document ID to remove
        """
        self.inverted_index.remove_document(doc_id)
        if self.index_dir:
            self.inverted_index.save(incremental=True)

    def batch_remove_document(self, doc_ids: List[int]):
        """批量删除多个文档
        
        Args:
            doc_ids: 要删除的文档ID列表，例如 [1, 2, 3]
        """
        self.inserter.batch_remove_document(doc_ids)

    def update_document(self, doc_id: int, 
                       fields: Union[Dict[str, Union[str, int, float]], 
                       List[Dict[str, Union[str, int, float]]]]):
        """Update a document in the index
        
        Args:
            doc_id: The document ID to update
            fields: The fields to update
        """
        self.inserter.update_document(doc_id, fields)

    def batch_update_document(self, doc_ids: List[int], 
                             fields: List[Dict[str, Union[str, int, float]]]):
        """批量更新多个文档
        
        Args:
            doc_ids: 文档ID列表，例如 [1, 2, 3]
            fields: 文档字段列表，与doc_ids一一对应，例如 [{"title": "doc1"}, {"title": "doc2"}, {"title": "doc3"}]
        """
        self.inserter.batch_update_document(doc_ids, fields)

    def from_pandas(self, df, id_column=None, text_columns=None):
        """Import data from a pandas DataFrame
        
        Args:
            df: pandas DataFrame object
            id_column: The name of the document ID column, if None, use the row index
            text_columns: The list of text columns to index, if None, use all string columns
        """
        self.inserter.from_pandas(df, id_column, text_columns)

    def from_polars(self, df, id_column=None, text_columns=None):
        """Import data from a polars DataFrame
        
        Args:
            df: polars DataFrame object
            id_column: The name of the document ID column, if None, use the row index
            text_columns: The list of text columns to index, if None, use all string columns
        """
        self.inserter.from_polars(df, id_column, text_columns)

    def from_arrow(self, table, id_column=None, text_columns=None):
        """Import data from a pyarrow Table
        
        Args:
            table: pyarrow Table object
            id_column: The name of the document ID column, if None, use the row index
            text_columns: The list of text columns to index, if None, use all string columns
        """
        self.inserter.from_arrow(table, id_column, text_columns)

    def from_parquet(self, path, id_column=None, text_columns=None):
        """Import data from a parquet file
        
        Args:
            path: The path to the parquet file
            id_column: The name of the document ID column, if None, use the row index
            text_columns: The list of text columns to index, if None, use all string columns
        """
        self.inserter.from_parquet(path, id_column, text_columns)

    def from_csv(self, path, id_column=None, text_columns=None):
        """Import data from a CSV file
        
        Args:
            path: The path to the CSV file
            id_column: The name of the document ID column, if None, use the row index
            text_columns: The list of text columns to index, if None, use all string columns
        """
        self.inserter.from_csv(path, id_column, text_columns)


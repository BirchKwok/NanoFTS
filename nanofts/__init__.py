import re
import shutil
from typing import List, Union, Dict
from pathlib import Path

from pyroaring import BitMap

from .factory import IndexFactory, IndexType
from .inserter import DocumentInserter


class FullTextSearch:
    def __init__(self, 
                 index_dir: str = None, 
                 max_chinese_length: int = 4, 
                 num_workers: int = 8,
                 shard_size: int = 500_000,
                 min_term_length: int = 2,
                 auto_save: bool = True,
                 batch_size: int = 10000,
                 drop_if_exists: bool = False,
                 buffer_size: int = 100000,
                 fuzzy_threshold: float = 0.4,
                 fuzzy_max_distance: int = 2):
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
            fuzzy_threshold (float): The similarity threshold for fuzzy search (0.0-1.0), default is 0.7
            fuzzy_max_distance (int): The maximum edit distance for fuzzy search, default is 2
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
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_max_distance = fuzzy_max_distance
        
        index_type_enum = IndexType("inverted")
            
        self.inverted_index = IndexFactory.create_index(
            index_type=index_type_enum,
            index_dir=index_dir,
            max_chinese_length=max_chinese_length,
            min_term_length=min_term_length,
            buffer_size=buffer_size,
            fuzzy_threshold=fuzzy_threshold,
            fuzzy_max_distance=fuzzy_max_distance
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
        """添加一个或多个文档到索引
        
        Args:
            doc_id: 单个文档ID或文档ID列表
            fields: 单个文档的字段或文档字段列表，与doc_id一一对应
            
        Examples:
            # 添加单个文档
            fts.add_document(1, {"title": "doc1", "content": "content1"})
            
            # 批量添加多个文档
            fts.add_document([1, 2], [
                {"title": "doc1", "content": "content1"},
                {"title": "doc2", "content": "content2"}
            ])
        """
        self.inserter.add_documents(doc_id, fields)

    def search(self, query: str, enable_fuzzy: bool = False, min_results: int = 5) -> Union[BitMap, List[tuple[int, float]]]:
        """Search for a query
        
        Args:
            query: The query to search for
            enable_fuzzy: Whether to enable fuzzy search, default is False
            min_results: Minimum number of results before enabling fuzzy search, default is 5
            
        Returns:
            Union[BitMap, List[tuple[int, float]]]: The document ID set or a list of (document ID, similarity)
            
        Examples:
            # 精确搜索
            results = fts.search("搜索查询")
            
            # 启用模糊搜索
            results = fts.search("搜索查询", enable_fuzzy=True)
            
            # 设置最小结果数阈值
            results = fts.search("搜索查询", enable_fuzzy=True, min_results=10)
        """
        return self.inverted_index.search(query, enable_fuzzy=enable_fuzzy, min_results=min_results)

    def fuzzy_search(self, query: str, min_results: int = 5) -> Union[BitMap, List[tuple[int, float]]]:
        """便捷的模糊搜索方法
        
        Args:
            query: The query to search for
            min_results: Minimum number of results before enabling fuzzy search, default is 5
            
        Returns:
            Union[BitMap, List[tuple[int, float]]]: The document ID set or a list of (document ID, similarity)
        """
        return self.search(query, enable_fuzzy=True, min_results=min_results)

    def flush(self):
        """Flush the buffer and save to disk"""
        self.inserter.flush()

    def remove_document(self, doc_id: Union[int, List[int]]):
        """从索引中删除一个或多个文档
        
        Args:
            doc_id: 单个文档ID或要删除的文档ID列表
            
        Examples:
            # 删除单个文档
            fts.remove_document(1)
            
            # 批量删除多个文档
            fts.remove_document([1, 2, 3])
        """
        if isinstance(doc_id, int):
            self.inverted_index.remove_document(doc_id)
        else:
            self.inverted_index.batch_remove_document(doc_id)
            
        if self.index_dir:
            self.inverted_index.save(incremental=True)

    def update_document(self, doc_id: Union[int, List[int]], 
                       fields: Union[Dict[str, Union[str, int, float]], 
                                  List[Dict[str, Union[str, int, float]]]]):
        """更新一个或多个文档
        
        Args:
            doc_id: 单个文档ID或文档ID列表
            fields: 单个文档的更新字段或文档字段列表，与doc_id一一对应
            
        Examples:
            # 更新单个文档
            fts.update_document(1, {"title": "new title", "content": "new content"})
            
            # 批量更新多个文档
            fts.update_document([1, 2], [
                {"title": "new title 1", "content": "new content 1"},
                {"title": "new title 2", "content": "new content 2"}
            ])
        """
        if isinstance(doc_id, int):
            self.inserter.update_document(doc_id, fields)
        else:
            self.inserter.batch_update_document(doc_id, fields)

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

    def get_fuzzy_config(self) -> Dict[str, Union[float, int]]:
        """获取模糊搜索配置
        
        Returns:
            Dict containing fuzzy search configuration
        """
        return {
            "fuzzy_threshold": self.fuzzy_threshold,
            "fuzzy_max_distance": self.fuzzy_max_distance
        }

    def set_fuzzy_config(self, fuzzy_threshold: float = None, fuzzy_max_distance: int = None):
        """设置模糊搜索配置
        
        Args:
            fuzzy_threshold: The similarity threshold for fuzzy search (0.0-1.0)
            fuzzy_max_distance: The maximum edit distance for fuzzy search
        """
        if fuzzy_threshold is not None:
            self.fuzzy_threshold = fuzzy_threshold
            self.inverted_index.fuzzy_threshold = fuzzy_threshold
            
        if fuzzy_max_distance is not None:
            self.fuzzy_max_distance = fuzzy_max_distance
            self.inverted_index.fuzzy_max_distance = fuzzy_max_distance


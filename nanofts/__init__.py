import re

import shutil

from typing import List, Union, Dict
from pathlib import Path

from nanofts.lru import LRUCache
from nanofts.index import InvertedIndex
from nanofts.inserter import DocumentInserter


class FullTextSearch:
    def __init__(self, index_dir: str = None, 
                 max_chinese_length: int = 4, 
                 num_workers: int = 8,  # 增加工作线程数
                 shard_size: int = 500_000,  # 增加分片大小
                 min_term_length: int = 2,   # 最小词长度
                 auto_save: bool = True,     # 是否自动保存
                 batch_size: int = 10000,  # 增加批处理大小
                 drop_if_exists: bool = False,
                 buffer_size: int = 100000):  # 增加缓冲区大小
        """
        初始化全文搜索器。

        Args:
            index_dir: 索引文件存储目录，如果为None则使用内存索引
            max_chinese_length: 中文子串的最大长度，默认为4个字符
            num_workers: 并行构建索引的工作进程数，默认为8
            shard_size: 每个分片包含的文档数，默认50万
            min_term_length: 最小词长度，小于此长度的词不会被索引
            auto_save: 是否自动保存到磁盘，默认为True
            batch_size: 批量处理大小，达到此数量时才更新词组索引和保存，默认10000
            drop_if_exists: 如果索引文件存在，是否删除，默认为False
            buffer_size: 内存缓冲区大小，达到此大小时才写入磁盘，默认10万
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
        
        # 初始化索引
        self.inverted_index = InvertedIndex(
            index_dir=self.index_dir,
            max_chinese_length=max_chinese_length,
            min_term_length=min_term_length,
            buffer_size=buffer_size
        )
        
        # 初始化插入器
        self.inserter = DocumentInserter(
            inverted_index=self.inverted_index,
            num_workers=num_workers,
            batch_size=batch_size,
            shard_size=shard_size
        )
        
        # 查询缓存
        self.cache = LRUCache(maxsize=10000)
        
        # 批处理计数器
        self._batch_count = 0
        
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.inverted_index.load()

    def add_document(self, doc_id: Union[int, List[int]], 
                    fields: Union[Dict[str, Union[str, int, float]], 
                                List[Dict[str, Union[str, int, float]]]]):
        """添加文档到索引"""
        self.inserter.add_documents(doc_id, fields)

    def search(self, query: str) -> List[int]:
        """搜索查询"""
        return self.inverted_index.search(query)

    def flush(self):
        """刷新缓冲区并保存"""
        self.inserter.flush()

    def remove_document(self, doc_id: int):
        """删除文档"""
        self.inverted_index.remove_document(doc_id)
        if self.index_dir:
            self.inverted_index.save(doc_id // self.shard_size, incremental=True)
        self.cache = LRUCache(maxsize=self.cache.maxsize)

    def from_pandas(self, df, id_column=None, text_columns=None):
        """从pandas DataFrame导入数据"""
        self.inserter.from_pandas(df, id_column, text_columns)

    def from_polars(self, df, id_column=None, text_columns=None):
        """从polars DataFrame导入数据"""
        self.inserter.from_polars(df, id_column, text_columns)

    def from_arrow(self, table, id_column=None, text_columns=None):
        """
        从pyarrow Table导入数据。

        Args:
            table: pyarrow Table对象
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有string类型的列
        """
        self.inserter.from_arrow(table, id_column, text_columns)

    def from_parquet(self, path, id_column=None, text_columns=None):
        """
        从parquet文件导入数据。

        Args:
            path: parquet文件路径
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有string类型的列
        """
        self.inserter.from_parquet(path, id_column, text_columns)

    def from_csv(self, path, id_column=None, text_columns=None):
        """
        从csv文件导入数据。

        Args:
            path: csv文件路径
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有string类型的列
        """
        self.inserter.from_csv(path, id_column, text_columns)


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
        初始化全文搜索引擎

        Args:
            index_dir: 索引文件存储目录，如果为None则使用内存索引
            max_chinese_length: 中文子串最大长度，默认为4
            num_workers: 并行索引的工作线程数，默认为8
            shard_size: 每个分片的文档数，默认为500,000
            min_term_length: 最小词条长度，默认为2
            auto_save: 是否自动保存到磁盘，默认为True
            batch_size: 每批处理的文档数，默认为10000
            drop_if_exists: 如果索引文件存在是否删除，默认为False
            buffer_size: 内存缓冲区大小，默认为100000
            index_type: 索引类型，可选值：inverted、ngram，默认为inverted
            ngram_size: n-gram大小，仅用于NGram索引，默认为2
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
        
        # 创建索引实例
        try:
            index_type_enum = IndexType(index_type)
        except ValueError:
            raise ValueError(f"不支持的索引类型: {index_type}")
            
        self.inverted_index = IndexFactory.create_index(
            index_type=index_type_enum,
            index_dir=index_dir,
            max_chinese_length=max_chinese_length,
            min_term_length=min_term_length,
            buffer_size=buffer_size,
            ngram_size=ngram_size
        )
        
        # 初始化插入器
        self.inserter = DocumentInserter(
            index=self.inverted_index,
            num_workers=num_workers,
            batch_size=batch_size,
            shard_size=shard_size
        )
        
        # 批处理计数器
        self._batch_count = 0
        
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.inverted_index.load()

    def add_document(self, doc_id: Union[int, List[int]], 
                    fields: Union[Dict[str, Union[str, int, float]], 
                                List[Dict[str, Union[str, int, float]]]]):
        """添加文档到索引
        
        Args:
            doc_id: 文档ID或ID列表
            fields: 要索引的字段
        """
        self.inserter.add_documents(doc_id, fields)

    def search(self, query: str) -> Union[BitMap, List[tuple[int, float]]]:
        """搜索查询
        
        Args:
            query: 要搜索的查询
        Returns:
            Union[BitMap, List[tuple[int, float]]]: 文档ID集合或(文档ID, 相似度)列表
        """
        return self.inverted_index.search(query)

    def flush(self):
        """刷新缓冲区并保存到磁盘"""
        self.inserter.flush()

    def remove_document(self, doc_id: int):
        """从索引中移除文档
        
        Args:
            doc_id: 要移除的文档ID
        """
        self.inverted_index.remove_document(doc_id)
        if self.index_dir:
            self.inverted_index.save(incremental=True)

    def from_pandas(self, df, id_column=None, text_columns=None):
        """从pandas DataFrame导入数据
        
        Args:
            df: pandas DataFrame对象
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列列表，如果为None则使用所有字符串列
        """
        self.inserter.from_pandas(df, id_column, text_columns)

    def from_polars(self, df, id_column=None, text_columns=None):
        """从polars DataFrame导入数据
        
        Args:
            df: polars DataFrame对象
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列列表，如果为None则使用所有字符串列
        """
        self.inserter.from_polars(df, id_column, text_columns)

    def from_arrow(self, table, id_column=None, text_columns=None):
        """从pyarrow Table导入数据
        
        Args:
            table: pyarrow Table对象
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列列表，如果为None则使用所有字符串列
        """
        self.inserter.from_arrow(table, id_column, text_columns)

    def from_parquet(self, path, id_column=None, text_columns=None):
        """从parquet文件导入数据
        
        Args:
            path: parquet文件路径
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列列表，如果为None则使用所有字符串列
        """
        self.inserter.from_parquet(path, id_column, text_columns)

    def from_csv(self, path, id_column=None, text_columns=None):
        """从CSV文件导入数据
        
        Args:
            path: CSV文件路径
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列列表，如果为None则使用所有字符串列
        """
        self.inserter.from_csv(path, id_column, text_columns)


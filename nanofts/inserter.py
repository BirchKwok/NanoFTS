from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Union
import numpy as np

from nanofts.index import InvertedIndex


class DocumentInserter:
    """文档插入器，负责处理文档的批量插入"""
    
    def __init__(self, 
                 inverted_index: InvertedIndex,
                 num_workers: int = 8,
                 batch_size: int = 10000,
                 shard_size: int = 500_000):
        """
        初始化文档插入器
        
        Args:
            inverted_index: 倒排索引实例
            num_workers: 并行处理的工作线程数
            batch_size: 批处理大小
            shard_size: 分片大小
        """
        self.index = inverted_index
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shard_size = shard_size
        self._batch_count = 0

    def add_documents(self, 
                     doc_ids: Union[int, List[int]], 
                     docs: Union[Dict[str, Union[str, int, float]], 
                               List[Dict[str, Union[str, int, float]]]]) -> None:
        """
        批量添加文档
        
        Args:
            doc_ids: 文档ID或ID列表
            docs: 文档内容或文档列表
        """
        # 标准化输入
        if isinstance(doc_ids, int):
            doc_ids = [doc_ids]
            docs = [docs] if isinstance(docs, dict) else docs
        else:
            docs = docs if isinstance(docs, list) else [docs]
        
        if len(doc_ids) != len(docs):
            raise ValueError("文档ID列表和文档列表长度必须相同")
        
        # 并行处理大批量文档
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
        
        # 检查是否需要合并和保存
        if len(self.index.index_buffer) >= self.index.buffer_size * 4:
            self.index.merge_buffer()
        
        if self._batch_count >= self.batch_size * 4:
            self.flush()

    def flush(self) -> None:
        """刷新缓冲区并保存"""
        self.index.merge_buffer()
        if self._batch_count > 0:
            self.index.build_word_index()
            if self.index.index_dir:
                shard_id = self._batch_count // self.shard_size
                self.index.save(shard_id, incremental=True)
            self._batch_count = 0

    def from_pandas(self, df, id_column=None, text_columns=None):
        """从pandas DataFrame导入数据"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("使用from_pandas需要安装pandas，请执行: pip install nanofts[pandas]")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df必须是pandas DataFrame对象")

        # 获取文档ID
        if id_column is None:
            doc_ids = df.index.tolist()
        else:
            doc_ids = df[id_column].tolist()

        # 获取要索引的列
        if text_columns is None:
            text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # 构建文档列表
        docs = []
        for _, row in df[text_columns].iterrows():
            doc = {col: str(val) for col, val in row.items() if pd.notna(val)}
            docs.append(doc)

        # 添加文档并刷新
        self.add_documents(doc_ids, docs)
        self.flush()

    def from_polars(self, df, id_column=None, text_columns=None):
        """从polars DataFrame导入数据"""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("使用from_polars需要安装polars，请执行: pip install nanofts[polars]")

        if not isinstance(df, pl.DataFrame):
            raise TypeError("df必须是polars DataFrame对象")

        # 获取文档ID
        if id_column is None:
            doc_ids = list(range(len(df)))
        else:
            doc_ids = df[id_column].to_list()

        # 获取要索引的列
        if text_columns is None:
            text_columns = [col for col in df.columns if df[col].dtype == pl.Utf8]

        # 构建文档列表
        docs = []
        for row in df.select(text_columns).iter_rows():
            doc = {col: str(val) for col, val in zip(text_columns, row) if val is not None}
            docs.append(doc)

        # 添加文档并刷新
        self.add_documents(doc_ids, docs)
        self.flush() 
    
    def from_arrow(self, table, id_column=None, text_columns=None):
        """从arrow表导入数据"""
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("使用from_arrow需要安装pyarrow，请执行: pip install nanofts[pyarrow]")

        if not isinstance(table, pa.Table):
            raise TypeError("table必须是pyarrow Table对象")

        # 获取要索引的列
        if text_columns is None:
            text_columns = [field.name for field in table.schema 
                           if pa.types.is_string(field.type)]
            if id_column and id_column in text_columns:
                text_columns.remove(id_column)

        # 获取文档ID
        if id_column is None:
            doc_ids = range(table.num_rows)
        else:
            # 修复：正确处理id列
            id_array = table.column(id_column)
            if pa.types.is_integer(id_array.type):
                doc_ids = id_array.to_numpy()
            else:
                # 如果不是整数类型，尝试转换
                doc_ids = [int(val.as_py()) for val in id_array]

        # 优化：使用批处理处理大表
        batch_size = 50000
        num_batches = (table.num_rows + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, table.num_rows)
            
            # 使用slice避免完整复制
            batch = table.slice(start_idx, end_idx - start_idx)
            
            # 直接使用Arrow的列访问
            docs = []
            for i in range(batch.num_rows):
                doc = {}
                for col in text_columns:
                    val = batch.column(col)[i].as_py()
                    if val is not None:
                        doc[col] = str(val)
                docs.append(doc)
            
            # 添加这一批文档
            batch_doc_ids = (doc_ids[start_idx:end_idx] if isinstance(doc_ids, (list, np.ndarray)) 
                            else range(start_idx, end_idx))
            self.add_documents(list(batch_doc_ids), docs)  # 确保转换为列表

        # 最后刷新
        self.flush()

    def from_parquet(self, path, id_column=None, text_columns=None):
        """从parquet文件导入数据"""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("使用from_parquet需要安装pyarrow，请执行: pip install nanofts[pyarrow]")

        # 优化：使用内存映射读取大文件
        table = pq.read_table(path, memory_map=True)
        self.from_arrow(table, id_column, text_columns)

    def from_csv(self, path, id_column=None, text_columns=None):
        """从csv文件导入数据"""
        try:
            import polars as pl
        except ImportError:
            raise ImportError("使用from_csv需要安装polars，请执行: pip install nanofts[polars]")

        df = pl.read_csv(path)
        self.from_polars(df, id_column, text_columns)  
        
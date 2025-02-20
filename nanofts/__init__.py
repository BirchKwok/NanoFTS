import re

import shutil
import msgpack
from collections import defaultdict
from typing import List, Union, Dict, Tuple
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

from pyroaring import BitMap

from nanofts.lru import LRUCache


class FullTextSearch:
    def __init__(self, index_dir: str = None, 
                 max_chinese_length: int = 4, 
                 num_workers: int = 4,
                 shard_size: int = 100_000,  # 每个分片包含的文档数
                 min_term_length: int = 2,   # 最小词长度
                 auto_save: bool = True,     # 是否自动保存
                 batch_size: int = 1000,     # 批量处理大小
                 drop_if_exists: bool = False,
                 buffer_size: int = 10000):  # 内存缓冲区大小
        """
        初始化全文搜索器。

        Args:
            index_dir: 索引文件存储目录，如果为None则使用内存索引
            max_chinese_length: 中文子串的最大长度，默认为4个字符
            num_workers: 并行构建索引的工作进程数，默认为4
            shard_size: 每个分片包含的文档数，默认10万
            min_term_length: 最小词长度，小于此长度的词不会被索引
            auto_save: 是否自动保存到磁盘，默认为True
            batch_size: 批量处理大小，达到此数量时才更新词组索引和保存，默认1000
            drop_if_exists: 如果索引文件存在，是否删除，默认为False
            buffer_size: 内存缓冲区大小，达到此大小时才写入磁盘，默认10000
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
        
        # 使用 LRU 缓存管理内存中的索引
        self.cache = LRUCache(maxsize=10000)
        self.modified_keys = set()
        
        # 内存索引和缓冲区
        self.index = defaultdict(BitMap)
        self.word_index = defaultdict(BitMap)
        self.index_buffer = defaultdict(set)  # 文档缓冲区
        
        # 批量处理计数器
        self._batch_count = 0
        
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    def _get_shard_path(self, shard_id: int) -> Path:
        """获取分片文件路径"""
        return self.index_dir / 'shards' / f'shard_{shard_id}.apex'

    def _save_index(self, incremental: bool = True):
        """将索引保存到磁盘，使用分片存储"""
        if not incremental:
            # 完整保存时，先清理旧的分片
            shards_dir = self.index_dir / 'shards'
            if shards_dir.exists():
                for f in shards_dir.glob('*.apex'):
                    f.unlink()
            shards_dir.mkdir(exist_ok=True)
        
        # 按文档ID范围分片
        shards = defaultdict(dict)
        for term, bitmap in self.index.items():
            if len(term) < self.min_term_length:  # 跳过过短的词
                continue
            # 直接遍历 bitmap，无需排序，因为 bitmap 本身就是有序的
            for doc_id in bitmap:
                shard_id = doc_id // self.shard_size
                if term not in shards[shard_id]:
                    shards[shard_id][term] = BitMap()
                shards[shard_id][term].add(doc_id)
        
        # 保存每个分片
        for shard_id, shard_data in shards.items():
            shard_path = self._get_shard_path(shard_id)
            self._save_shard(shard_data, shard_path, incremental)
        
        # 保存词组索引
        word_dir = self.index_dir / 'word'
        word_dir.mkdir(exist_ok=True)
        self._save_shard(self.word_index, word_dir / "index.apex", incremental)
        
        if incremental:
            self.modified_keys.clear()

    def _save_shard(self, shard_data: Dict[str, BitMap], shard_path: Path, incremental: bool):
        """保存单个分片"""
        if not shard_data:
            if shard_path.exists():
                shard_path.unlink()
            return
        
        # 如果是增量更新，先读取现有数据
        existing_meta = {}
        existing_data = {}
        if incremental and shard_path.exists():
            with open(shard_path, 'rb') as f:
                meta_size = int.from_bytes(f.read(4), 'big')
                meta_data = f.read(meta_size)
                existing_meta = msgpack.unpackb(meta_data, raw=False)
                
                for key, (offset, size) in existing_meta.items():
                    if key not in self.modified_keys:
                        f.seek(4 + meta_size + offset)
                        existing_data[key] = f.read(size)
        
        # 准备新数据
        data = {}
        meta = {}
        offset = 0
        
        # 处理现有数据
        for key, bitmap_data in existing_data.items():
            meta[key] = (offset, len(bitmap_data))
            data[key] = bitmap_data
            offset += len(bitmap_data)
        
        # 处理新数据
        for key, bitmap in shard_data.items():
            if not bitmap:  # 跳过空的 BitMap
                continue
            if not incremental or key in self.modified_keys:
                bitmap_data = self._bitmap_to_bytes(bitmap)
                data[key] = bitmap_data
                meta[key] = (offset, len(bitmap_data))
                offset += len(bitmap_data)
        
        # 如果没有数据要保存，删除文件
        if not data:
            if shard_path.exists():
                shard_path.unlink()
            return
        
        # 保存分片
        shard_path.parent.mkdir(exist_ok=True)
        with open(shard_path, 'wb') as f:
            meta_data = msgpack.packb(meta, use_bin_type=True)
            f.write(len(meta_data).to_bytes(4, 'big'))
            f.write(meta_data)
            for bitmap_data in data.values():
                f.write(bitmap_data)

    def _load_index(self) -> bool:
        """从磁盘加载索引"""
        try:
            # 加载所有分片
            shards_dir = self.index_dir / 'shards'
            if not shards_dir.exists():
                return False
            
            for shard_path in shards_dir.glob('*.apex'):
                self._load_shard(shard_path)
            
            # 加载词组索引
            word_dir = self.index_dir / 'word'
            if word_dir.exists():
                word_index_path = word_dir / "index.apex"
                if word_index_path.exists():
                    self._load_shard(word_index_path, is_word_index=True)
            
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False

    def _load_shard(self, shard_path: Path, is_word_index: bool = False):
        """加载单个分片"""
        if not shard_path.exists():
            return
        
        with open(shard_path, 'rb') as f:
            meta_size = int.from_bytes(f.read(4), 'big')
            meta_data = f.read(meta_size)
            meta = msgpack.unpackb(meta_data, raw=False)
            
            for key, (offset, size) in meta.items():
                if len(key) >= self.min_term_length:  # 只加载符合最小长度要求的词
                    f.seek(4 + meta_size + offset)
                    bitmap_data = f.read(size)
                    if is_word_index:
                        self.word_index[key] |= self._bytes_to_bitmap(bitmap_data)
                    else:
                        self.index[key] |= self._bytes_to_bitmap(bitmap_data)

    @staticmethod
    def _process_chunk(chunk: Tuple[int, List[Dict[str, Union[str, int, float]]], int, int]) -> Dict[str, List[int]]:
        """
        并行处理数据块，返回部分索引结果
        
        Args:
            chunk: (起始ID, 数据块, 最大中文长度, 最小词长度)的元组
        """
        start_id, docs, max_chinese_length, min_term_length = chunk
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        result = defaultdict(set)
        
        # 预编译正则表达式和常用变量
        word_splitter = re.compile(r'\s+')
        chinese_cache = {}  # 缓存中文处理结果
        
        for i, doc in enumerate(docs):
            doc_id = start_id + i
            
            # 分别处理每个字段
            for field_value in doc.values():
                field_str = str(field_value).lower()
                
                # 处理完整字段
                if len(field_str) >= min_term_length:
                    result[field_str].add(doc_id)
                
                # 处理中文部分
                for match in chinese_pattern.finditer(field_str):
                    seg = match.group()
                    # 使用缓存避免重复计算
                    if seg not in chinese_cache:
                        n = len(seg)
                        substrings = {seg[j:j + length] 
                                    for length in range(min_term_length, min(n + 1, max_chinese_length + 1))
                                    for j in range(n - length + 1)}
                        chinese_cache[seg] = substrings
                    
                    # 从缓存中获取子串
                    for substr in chinese_cache[seg]:
                        result[substr].add(doc_id)
                
                # 处理词组
                if ' ' in field_str:
                    # 对整个词组建立索引
                    result[field_str].add(doc_id)
                    # 对单词建立索引
                    words = word_splitter.split(field_str)
                    for word in words:
                        if len(word) >= min_term_length and not chinese_pattern.search(word):
                            result[word].add(doc_id)
        
        # 使用列表推导式优化返回值创建
        return {k: list(v) for k, v in result.items() if v}

    def add_document(self, doc_id: Union[int, List[int]], fields: Union[Dict[str, Union[str, int, float]], List[Dict[str, Union[str, int, float]]]]):
        """
        添加文档到索引。支持单条文档和批量文档插入。

        Args:
            doc_id: 文档ID，可以是单个整数或整数列表
            fields: 文档字段，可以是单个字典或字典列表。每个字典的值可以是字符串、整数或浮点数
        """
        # 转换为列表格式以统一处理
        if isinstance(doc_id, int):
            doc_ids = [doc_id]
            docs = [fields] if isinstance(fields, dict) else fields
        else:
            doc_ids = doc_id
            docs = fields if isinstance(fields, list) else [fields]
        
        # 验证输入
        if len(doc_ids) != len(docs):
            raise ValueError("文档ID列表和文档列表长度必须相同")
        
        # 优化chunk大小计算
        total_docs = len(docs)
        if total_docs < 1000:
            # 小批量数据直接处理
            results = [self._process_chunk((doc_ids[0], docs, self.max_chinese_length, self.min_term_length))]
        else:
            # 大批量数据并行处理
            chunk_size = max(1000, total_docs // self.num_workers)
            chunks = []
            
            for i in range(0, total_docs, chunk_size):
                chunk_docs = docs[i:i + chunk_size]
                chunk_start_id = doc_ids[i]
                chunk = (chunk_start_id, chunk_docs, self.max_chinese_length, self.min_term_length)
                chunks.append(chunk)
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self._process_chunk, chunks))
        
        # 使用缓冲区合并结果
        for result in results:
            for key, doc_ids in result.items():
                if len(key) >= self.min_term_length:
                    self.index_buffer[key].update(doc_ids)
        
        # 更新批处理计数
        self._batch_count += len(docs)
        
        # 当缓冲区达到阈值时，合并到主索引
        if len(self.index_buffer) >= self.buffer_size:
            self._merge_buffer()
        
        # 当达到批处理大小时，更新词组索引并保存
        if self._batch_count >= self.batch_size:
            self._merge_buffer()  # 确保所有数据都已合并
            self._build_word_index()
            if self.auto_save and self.index_dir:
                self._save_index(incremental=True)
            self._batch_count = 0

    def _merge_buffer(self):
        """合并缓冲区到主索引"""
        for key, doc_ids in self.index_buffer.items():
            if doc_ids:  # 只处理非空集合
                self.index[key] |= BitMap(doc_ids)
                self.modified_keys.add(key)
        self.index_buffer.clear()

    def flush(self):
        """
        强制将当前的更改保存到磁盘，并更新词组索引。
        在批量添加完成后调用此方法以确保所有更改都已保存。
        """
        if self.index_buffer:
            self._merge_buffer()
        if self._batch_count > 0:
            self._build_word_index()
            if self.index_dir:
                self._save_index(incremental=True)
            self._batch_count = 0

    def _build_word_index(self):
        """构建词组反向索引"""
        # 使用临时字典收集所有词的文档ID
        temp_word_index = defaultdict(set)
        word_splitter = re.compile(r'\s+')
        
        for field_str, doc_ids in self.index.items():
            if ' ' in field_str:
                # 只处理包含空格的字段
                words = word_splitter.split(field_str.lower())
                doc_ids_list = list(doc_ids)
                for word in words:
                    if not self.chinese_pattern.search(word) and len(word) >= self.min_term_length:
                        temp_word_index[word].update(doc_ids_list)
        
        # 批量更新词组索引
        self.word_index.clear()
        for word, doc_ids in temp_word_index.items():
            if doc_ids:  # 只添加非空集合
                self.word_index[word] = BitMap(doc_ids)

    def search(self, query: str) -> List[int]:
        """搜索实现"""
        query_key = query.lower()  # 统一转换为小写

        # 使用缓存获取结果
        cached_result = self.cache.get(query_key)
        if cached_result is not None:
            return list(cached_result)  # BitMap已经是有序的，直接转换为列表

        result = BitMap()
        
        # 先尝试精确匹配
        if query_key in self.index:
            result |= self.index[query_key]
            self.cache.put(query_key, result)
            return list(result)  # 直接转换为列表，无需排序
        
        # 检查是否为中文查询
        if self.chinese_pattern.search(query_key):
            # 中文查询：尝试子串匹配
            for length in range(len(query_key), self.min_term_length - 1, -1):
                for i in range(len(query_key) - length + 1):
                    substr = query_key[i:i + length]
                    if substr in self.index:
                        result |= self.index[substr]
                        if result:  # 如果找到匹配，就返回
                            self.cache.put(query_key, result)
                            return list(result)  # 直接转换为列表
        else:
            # 非中文查询：词组匹配
            if ' ' in query_key:
                words = query_key.split()
                word_results = []
                
                # 收集每个单词的匹配结果
                for word in words:
                    if len(word) >= self.min_term_length:
                        word_result = BitMap()
                        if word in self.index:
                            word_result |= self.index[word]
                        if not word_result:  # 如果有任何一个词没有匹配，直接返回空结果
                            self.cache.put(query_key, BitMap())
                            return []
                        word_results.append(word_result)
                
                # 使用交集获取包含所有单词的文档
                if word_results:
                    result = word_results[0]
                    for other_result in word_results[1:]:
                        result &= other_result

        # 缓存结果
        self.cache.put(query_key, result)
        return list(result)  # 直接转换为列表，无需排序

    def _bitmap_to_bytes(self, bitmap: BitMap) -> bytes:
        """将 BitMap 转换为字节串"""
        return bitmap.serialize()
    
    def _bytes_to_bitmap(self, data: bytes) -> BitMap:
        """将字节串转换回 BitMap"""
        return BitMap.deserialize(data)

    def remove_document(self, doc_id: int):
        """从索引中删除文档"""
        # 先从主索引中删除
        modified = False
        keys_to_remove = []
        for key, doc_ids in self.index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                modified = True
                if not doc_ids:
                    keys_to_remove.append(key)
        
        # 删除空的键
        for key in keys_to_remove:
            del self.index[key]
        
        # 从词组索引中删除
        keys_to_remove = []
        for key, doc_ids in self.word_index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                modified = True
                if not doc_ids:
                    keys_to_remove.append(key)
        
        # 删除空的键
        for key in keys_to_remove:
            del self.word_index[key]
        
        # 如果有修改，保存到磁盘
        if modified and self.index_dir:
            self._save_index(incremental=True)
        
        # 清除缓存
        self.cache = LRUCache(maxsize=self.cache.maxsize)

    def from_pandas(self, df, id_column=None, text_columns=None):
        """
        从pandas DataFrame导入数据。

        Args:
            df: pandas DataFrame对象
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有object和string类型的列
        """
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
        self.add_document(doc_ids, docs)
        self.flush()
    def from_polars(self, df, id_column=None, text_columns=None):
        """
        从polars DataFrame导入数据。

        Args:
            df: polars DataFrame对象
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有Utf8类型的列
        """
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
        self.add_document(doc_ids, docs)
        self.flush()

    def from_arrow(self, table, id_column=None, text_columns=None):
        """
        从pyarrow Table导入数据。

        Args:
            table: pyarrow Table对象
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有string类型的列
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError("使用from_arrow需要安装pyarrow，请执行: pip install nanofts[pyarrow]")

        if not isinstance(table, pa.Table):
            raise TypeError("table必须是pyarrow Table对象")

        # 获取文档ID
        if id_column is None:
            doc_ids = list(range(len(table)))
        else:
            doc_ids = table[id_column].to_pylist()

        # 获取要索引的列
        if text_columns is None:
            text_columns = [field.name for field in table.schema 
                           if pa.types.is_string(field.type)]
            if id_column in text_columns:
                text_columns.remove(id_column)

        # 构建文档列表
        table_dict = table.select(text_columns).to_pydict()
        docs = []
        for i in range(len(table)):
            doc = {}
            for col in text_columns:
                val = table_dict[col][i]
                if val is not None:
                    doc[col] = str(val)
            docs.append(doc)

        # 添加文档并刷新
        self.add_document(doc_ids, docs)
        self.flush()

    def from_parquet(self, path, id_column=None, text_columns=None):
        """
        从parquet文件导入数据。

        Args:
            path: parquet文件路径
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有string类型的列
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError("使用from_parquet需要安装pyarrow，请执行: pip install nanofts[pyarrow]")

        table = pq.read_table(path)
        self.from_arrow(table, id_column, text_columns)

    def from_csv(self, path, id_column=None, text_columns=None):
        """
        从csv文件导入数据。

        Args:
            path: csv文件路径
            id_column: 文档ID列名，如果为None则使用行索引
            text_columns: 要索引的文本列名列表，如果为None则使用所有string类型的列
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("使用from_csv需要安装pandas，请执行: pip install nanofts[pandas]")
        
        df = pd.read_csv(path)
        self.from_pandas(df, id_column, text_columns)


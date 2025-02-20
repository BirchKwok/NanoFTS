import re
import msgpack
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Union, Tuple

from pyroaring import BitMap


class InvertedIndex:
    """倒排索引实现"""
    
    def __init__(self, 
                 index_dir: Path = None,
                 max_chinese_length: int = 4,
                 min_term_length: int = 2,
                 buffer_size: int = 100000):
        """
        初始化倒排索引。
        
        Args:
            index_dir: 索引文件存储目录
            max_chinese_length: 中文子串的最大长度
            min_term_length: 最小词长度
            buffer_size: 内存缓冲区大小
        """
        self.index_dir = index_dir
        self.max_chinese_length = max_chinese_length
        self.min_term_length = min_term_length
        self.buffer_size = buffer_size
        
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.modified_keys = set()
        
        # 主索引和缓冲区
        self.index = defaultdict(BitMap)
        self.word_index = defaultdict(BitMap)
        self.index_buffer = defaultdict(BitMap)
        
        # 中文处理缓存
        self._global_chinese_cache = {}

    def add_terms(self, doc_id: int, terms: Dict[str, Union[str, int, float]]) -> None:
        """添加文档的词条到索引"""
        for field_value in terms.values():
            field_str = str(field_value).lower()
            
            # 处理完整字段
            if len(field_str) >= self.min_term_length:
                self.index_buffer[field_str].add(doc_id)
            
            # 处理中文
            for match in self.chinese_pattern.finditer(field_str):
                seg = match.group()
                if seg not in self._global_chinese_cache:
                    n = len(seg)
                    substrings = {seg[j:j + length] 
                                for length in range(self.min_term_length, 
                                                  min(n + 1, self.max_chinese_length + 1))
                                for j in range(n - length + 1)}
                    self._global_chinese_cache[seg] = substrings
                
                for substr in self._global_chinese_cache[seg]:
                    self.index_buffer[substr].add(doc_id)
            
            # 处理词组
            if ' ' in field_str:
                self.index_buffer[field_str].add(doc_id)
                words = field_str.split()
                for word in words:
                    if len(word) >= self.min_term_length and not self.chinese_pattern.search(word):
                        self.index_buffer[word].add(doc_id)

    def merge_buffer(self) -> None:
        """合并缓冲区到主索引"""
        if not self.index_buffer:
            return
            
        for key, bitmap in self.index_buffer.items():
            if len(bitmap) > 0:
                if key in self.index:
                    self.index[key] |= bitmap
                else:
                    self.index[key] = bitmap
                self.modified_keys.add(key)
        
        self.index_buffer.clear()

    def build_word_index(self) -> None:
        """构建词组索引"""
        temp_word_index = defaultdict(set)
        
        for field_str, doc_ids in self.index.items():
            if ' ' in field_str:
                words = field_str.split()
                doc_ids_list = list(doc_ids)
                for word in words:
                    if not self.chinese_pattern.search(word) and len(word) >= self.min_term_length:
                        temp_word_index[word].update(doc_ids_list)
        
        self.word_index.clear()
        for word, doc_ids in temp_word_index.items():
            if doc_ids:
                self.word_index[word] = BitMap(doc_ids)

    def search(self, query: str) -> BitMap:
        """搜索查询词"""
        # 快速路径 - 直接返回
        query = query.strip().lower()
        if not query:
            return BitMap()
        
        # 1. 精确匹配 - 使用 get 而不是 in 操作符
        result = self.index.get(query)
        if result is not None:
            return result
        
        # 2. 词组查询优化 - 比中文查询更快，所以先处理
        if ' ' in query:
            words = query.split()
            if not words:
                return BitMap()
            
            # 预分配列表避免动态增长
            results = []
            min_size = float('inf')
            min_idx = 0
            
            # 一次性获取所有词的文档集
            for i, word in enumerate(words):
                if len(word) < self.min_term_length:
                    continue
                    
                docs = self.index.get(word)
                if docs is None:
                    return BitMap()  # 快速失败
                
                size = len(docs)
                if size == 0:
                    return BitMap()
                
                if size < min_size:
                    min_size = size
                    min_idx = len(results)
                
                results.append(docs)
            
            if not results:
                return BitMap()
            
            # 优化：直接使用最小结果集
            if min_idx > 0:
                results[0], results[min_idx] = results[min_idx], results[0]
            
            # 快速路径：只有一个词
            if len(results) == 1:
                return results[0]
            
            # 高效求交集
            result = results[0]
            for other in results[1:]:
                result &= other
                if not result:  # 提前返回空结果
                    return BitMap()
            
            return result
        
        # 3. 中文查询优化
        if self.chinese_pattern.search(query):
            n = len(query)
            if n < self.min_term_length:
                return BitMap()
            
            # 优化：直接使用最长可能的子串
            max_len = min(n, self.max_chinese_length)
            
            # 3.1 尝试最长匹配
            for i in range(n - max_len + 1):
                substr = query[i:i + max_len]
                result = self.index.get(substr)
                if result is not None:
                    if len(result) < 1000:  # 结果集较小时直接返回
                        return result
                    # 保存第一个匹配结果
                    first_match = result
                    
                    # 3.2 尝试与相邻子串求交集
                    if i > 0:
                        prev_substr = query[i-1:i-1+max_len]
                        prev_docs = self.index.get(prev_substr)
                        if prev_docs is not None:
                            temp = result & prev_docs
                            if temp:
                                return temp
                    
                    if i < n - max_len:
                        next_substr = query[i+1:i+1+max_len]
                        next_docs = self.index.get(next_substr)
                        if next_docs is not None:
                            temp = result & next_docs
                            if temp:
                                return temp
                    
                    return first_match
            
            # 3.3 回退到最小长度匹配
            for i in range(n - self.min_term_length + 1):
                substr = query[i:i + self.min_term_length]
                result = self.index.get(substr)
                if result is not None:
                    return result
        
        return BitMap()

    def remove_document(self, doc_id: int) -> None:
        """从索引中删除文档"""
        keys_to_remove = []
        
        # 从主索引中删除
        for key, doc_ids in self.index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                if not doc_ids:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.index[key]
        
        # 从词组索引中删除
        keys_to_remove = []
        for key, doc_ids in self.word_index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                if not doc_ids:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.word_index[key]

    def save(self, shard_id: int, incremental: bool = True) -> None:
        """保存索引分片"""
        if not self.index_dir:
            return
            
        shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.apex'
        self._save_shard(self.index, shard_path, incremental)
        
        # 保存词组索引
        word_dir = self.index_dir / 'word'
        word_dir.mkdir(exist_ok=True)
        self._save_shard(self.word_index, word_dir / "index.apex", incremental)
        
        if incremental:
            self.modified_keys.clear()

    def _save_shard(self, shard_data: Dict[str, BitMap], shard_path: Path, incremental: bool) -> None:
        """保存单个分片"""
        if not shard_data:
            if shard_path.exists():
                shard_path.unlink()
            return
            
        # 处理增量更新
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
            if not bitmap:
                continue
            if not incremental or key in self.modified_keys:
                bitmap_data = bitmap.serialize()
                data[key] = bitmap_data
                meta[key] = (offset, len(bitmap_data))
                offset += len(bitmap_data)
        
        # 保存
        if data:
            shard_path.parent.mkdir(exist_ok=True)
            with open(shard_path, 'wb') as f:
                meta_data = msgpack.packb(meta, use_bin_type=True)
                f.write(len(meta_data).to_bytes(4, 'big'))
                f.write(meta_data)
                for bitmap_data in data.values():
                    f.write(bitmap_data)
        elif shard_path.exists():
            shard_path.unlink()

    def load(self) -> bool:
        """加载索引"""
        if not self.index_dir:
            return False
            
        try:
            # 加载分片
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

    def _load_shard(self, shard_path: Path, is_word_index: bool = False) -> None:
        """加载单个分片"""
        if not shard_path.exists():
            return
            
        with open(shard_path, 'rb') as f:
            meta_size = int.from_bytes(f.read(4), 'big')
            meta_data = f.read(meta_size)
            meta = msgpack.unpackb(meta_data, raw=False)
            
            for key, (offset, size) in meta.items():
                if len(key) >= self.min_term_length:
                    f.seek(4 + meta_size + offset)
                    bitmap_data = f.read(size)
                    bitmap = BitMap.deserialize(bitmap_data)
                    if is_word_index:
                        self.word_index[key] |= bitmap
                    else:
                        self.index[key] |= bitmap 

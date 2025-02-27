import re
import msgpack
from pathlib import Path
from collections import defaultdict
from typing import Dict, Union, Optional

from pyroaring import BitMap
from functools import lru_cache
import xxhash

from .base import BaseIndex


class InvertedIndex(BaseIndex):
    """倒排索引实现"""
    
    def __init__(self, 
                 index_dir: Optional[Path] = None,
                 max_chinese_length: int = 4,
                 min_term_length: int = 2,
                 buffer_size: int = 100000,
                 shard_bits: int = 8,
                 cache_size: int = 1000):
        """
        初始化倒排索引
        
        Args:
            index_dir: 索引文件存储目录
            max_chinese_length: 中文子串最大长度
            min_term_length: 最小词条长度
            buffer_size: 内存缓冲区大小
            shard_bits: 分片位数
            cache_size: 缓存大小
        """
        self.index_dir = index_dir
        self.max_chinese_length = max_chinese_length
        self.min_term_length = min_term_length
        self.buffer_size = buffer_size
        
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.modified_keys = set()
        
        # 主索引和缓冲区
        self.word_index = defaultdict(BitMap)
        self.index_buffer = defaultdict(BitMap)
        
        # 中文处理缓存
        self._global_chinese_cache = {}
        
        self.shard_bits = shard_bits
        self.shard_count = 1 << shard_bits
        self.cache_size = cache_size
        
        self._init_cache()

    def _init_cache(self) -> None:
        """初始化LRU缓存"""
        @lru_cache(maxsize=self.cache_size)
        def get_bitmap(term: str) -> Optional[BitMap]:
            shard_id = self._get_shard_id(term)
            return self._load_term_bitmap(term, shard_id)
        
        self._bitmap_cache = get_bitmap

    def _get_shard_id(self, term: str) -> int:
        """计算词条的分片ID"""
        return xxhash.xxh32(term.encode()).intdigest() & (self.shard_count - 1)

    def _load_term_bitmap(self, term: str, shard_id: int) -> Optional[BitMap]:
        """从磁盘加载词条的位图"""
        if not self.index_dir:
            return None
            
        shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.apex'
        if not shard_path.exists():
            return None
            
        try:
            with open(shard_path, 'rb') as f:
                meta_size = int.from_bytes(f.read(4), 'big')
                meta_data = f.read(meta_size)
                meta = msgpack.unpackb(meta_data, raw=False)
                
                if term not in meta:
                    return None
                    
                offset, size = meta[term]
                f.seek(4 + meta_size + offset)
                bitmap_data = f.read(size)
                return BitMap.deserialize(bitmap_data)
        except:
            return None

    def add_terms(self, doc_id: int, terms: Dict[str, Union[str, int, float]]) -> None:
        """添加文档词条到索引"""
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

        # 当buffer达到阈值时自动合并
        if len(self.index_buffer) >= self.buffer_size:
            self.merge_buffer()

    def search(self, query: str) -> BitMap:
        """搜索查询"""
        # 快速路径 - 直接返回
        query = query.strip().lower()
        if not query:
            return BitMap()
        
        # 修改精确匹配部分
        result = self._bitmap_cache(query)
        if result is not None:
            return result.copy()  # 返回副本避免缓存被修改
        
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
                    
                docs = self._bitmap_cache(word)
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
                result = self._bitmap_cache(substr)
                if result is not None:
                    if len(result) < 1000:  # 结果集较小时直接返回
                        return result
                    # 保存第一个匹配结果
                    first_match = result
                    
                    # 3.2 尝试与相邻子串求交集
                    if i > 0:
                        prev_substr = query[i-1:i-1+max_len]
                        prev_docs = self._bitmap_cache(prev_substr)
                        if prev_docs is not None:
                            temp = result & prev_docs
                            if temp:
                                return temp
                    
                    if i < n - max_len:
                        next_substr = query[i+1:i+1+max_len]
                        next_docs = self._bitmap_cache(next_substr)
                        if next_docs is not None:
                            temp = result & next_docs
                            if temp:
                                return temp
                    
                    return first_match
            
            # 3.3 回退到最小长度匹配
            for i in range(n - self.min_term_length + 1):
                substr = query[i:i + self.min_term_length]
                result = self._bitmap_cache(substr)
                if result is not None:
                    return result
        
        return BitMap()

    def remove_document(self, doc_id: int) -> None:
        """从索引中移除文档"""
        if not self.index_dir:
            return
        
        # 从缓冲区删除
        keys_to_remove = []
        for key, doc_ids in self.index_buffer.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                if not doc_ids:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.index_buffer[key]
        
        # 从分片中删除
        shards_dir = self.index_dir / 'shards'
        if shards_dir.exists():
            for shard_path in shards_dir.glob('*.apex'):
                try:
                    # 读取分片数据
                    with open(shard_path, 'rb') as f:
                        meta_size = int.from_bytes(f.read(4), 'big')
                        meta_data = f.read(meta_size)
                        meta = msgpack.unpackb(meta_data, raw=False)
                        
                        modified = False
                        new_data = {}
                        new_meta = {}
                        offset = 0
                        
                        # 处理每个term
                        for term, (term_offset, size) in meta.items():
                            f.seek(4 + meta_size + term_offset)
                            bitmap_data = f.read(size)
                            bitmap = BitMap.deserialize(bitmap_data)
                            
                            if doc_id in bitmap:
                                bitmap.discard(doc_id)
                                modified = True
                                self.modified_keys.add(term)
                                
                            if bitmap:  # 只保存非空bitmap
                                bitmap_data = bitmap.serialize()
                                new_data[term] = bitmap_data
                                new_meta[term] = (offset, len(bitmap_data))
                                offset += len(bitmap_data)
                        
                        # 如果有修改，重写分片文件
                        if modified:
                            if new_data:
                                with open(shard_path, 'wb') as f:
                                    meta_data = msgpack.packb(new_meta, use_bin_type=True)
                                    f.write(len(meta_data).to_bytes(4, 'big'))
                                    f.write(meta_data)
                                    for bitmap_data in new_data.values():
                                        f.write(bitmap_data)
                            else:
                                # 如果分片为空，删除文件
                                shard_path.unlink()
                                
                except Exception as e:
                    print(f"处理分片 {shard_path} 时出错: {e}")
                    continue
        
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
        
        # 清空缓存
        self._bitmap_cache.cache_clear()

    def merge_buffer(self) -> None:
        """合并缓冲区"""
        if not self.index_buffer:
            return
            
        # 按分片分组
        sharded_buffer = defaultdict(lambda: defaultdict(BitMap))
        for term, bitmap in self.index_buffer.items():
            shard_id = self._get_shard_id(term)
            sharded_buffer[shard_id][term] = bitmap
        
        # 分片写入
        for shard_id, terms in sharded_buffer.items():
            self._merge_shard(shard_id, terms)
        
        # 清空buffer
        self.index_buffer.clear()
        # 清空缓存
        self._bitmap_cache.cache_clear()

    def _merge_shard(self, shard_id: int, terms: Dict[str, BitMap]) -> None:
        """合并单个分片的数据"""
        shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.apex'
        
        # 读取现有数据
        existing_meta = {}
        existing_data = {}
        if shard_path.exists():
            with open(shard_path, 'rb') as f:
                meta_size = int.from_bytes(f.read(4), 'big')
                meta_data = f.read(meta_size)
                existing_meta = msgpack.unpackb(meta_data, raw=False)
                
                for term, (offset, size) in existing_meta.items():
                    f.seek(4 + meta_size + offset)
                    bitmap_data = f.read(size)
                    existing_data[term] = bitmap_data
        
        # 合并数据
        new_data = {}
        new_meta = {}
        offset = 0
        
        for term, bitmap_data in existing_data.items():
            if term in terms:  # 需要更新
                bitmap = BitMap.deserialize(bitmap_data)
                bitmap |= terms[term]
                bitmap_data = bitmap.serialize()
                del terms[term]
            
            new_data[term] = bitmap_data
            new_meta[term] = (offset, len(bitmap_data))
            offset += len(bitmap_data)
        
        # 添加新terms
        for term, bitmap in terms.items():
            bitmap_data = bitmap.serialize()
            new_data[term] = bitmap_data
            new_meta[term] = (offset, len(bitmap_data))
            offset += len(bitmap_data)
        
        # 保存
        if new_data:
            shard_path.parent.mkdir(exist_ok=True)
            with open(shard_path, 'wb') as f:
                meta_data = msgpack.packb(new_meta, use_bin_type=True)
                f.write(len(meta_data).to_bytes(4, 'big'))
                f.write(meta_data)
                for bitmap_data in new_data.values():
                    f.write(bitmap_data)

    def save(self, incremental: bool = True) -> None:
        """保存索引"""
        if not self.index_dir:
            return
            
        # 将缓冲区数据按分片保存
        if self.index_buffer:
            self.merge_buffer()
        
        # 保存词组索引
        if self.word_index:
            word_dir = self.index_dir / 'word'
            word_dir.mkdir(exist_ok=True)
            self._save_shard(self.word_index, word_dir / "index.apex", incremental)
        
        if incremental:
            self.modified_keys.clear()

    def _save_shard(self, shard_data: Dict[str, BitMap], shard_path: Path, incremental: bool) -> None:
        """保存单个分片
        
        Args:
            shard_data: 分片数据
            shard_path: 分片路径
            incremental: 是否增量保存
        """
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
        """加载单个分片
        
        Args:
            shard_path: 分片路径
            is_word_index: 是否是词组索引
        """
        if not shard_path.exists():
            return
            
        try:
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
                            # 将数据加入缓冲区
                            self.index_buffer[key] |= bitmap
            
                # 如果不是词组索引，则合并到磁盘
                if not is_word_index and self.index_buffer:
                    self.merge_buffer()
                
        except Exception as e:
            print(f"加载分片 {shard_path} 时出错: {e}")

    def build_word_index(self) -> None:
        """构建词组索引"""
        if not self.index_dir:
            return
        
        temp_word_index = defaultdict(set)
        shards_dir = self.index_dir / 'shards'
        
        # 从所有分片中读取数据
        if shards_dir.exists():
            for shard_path in shards_dir.glob('*.apex'):
                try:
                    with open(shard_path, 'rb') as f:
                        meta_size = int.from_bytes(f.read(4), 'big')
                        meta_data = f.read(meta_size)
                        meta = msgpack.unpackb(meta_data, raw=False)
                        
                        for term, (offset, size) in meta.items():
                            if ' ' in term:  # 只处理包含空格的词组
                                f.seek(4 + meta_size + offset)
                                bitmap_data = f.read(size)
                                bitmap = BitMap.deserialize(bitmap_data)
                                
                                # 处理词组中的单词
                                words = term.split()
                                doc_ids = list(bitmap)
                                for word in words:
                                    if not self.chinese_pattern.search(word) and len(word) >= self.min_term_length:
                                        temp_word_index[word].update(doc_ids)
                except Exception as e:
                    print(f"处理分片 {shard_path} 时出错: {e}")
                    continue
        
        # 处理缓冲区中的数据
        for term, bitmap in self.index_buffer.items():
            if ' ' in term:
                words = term.split()
                doc_ids = list(bitmap)
                for word in words:
                    if not self.chinese_pattern.search(word) and len(word) >= self.min_term_length:
                        temp_word_index[word].update(doc_ids)
        
        # 转换为BitMap并保存
        self.word_index.clear()
        for word, doc_ids in temp_word_index.items():
            if doc_ids:
                self.word_index[word] = BitMap(doc_ids)
        
        # 保存词组索引
        if self.index_dir:
            word_dir = self.index_dir / 'word'
            word_dir.mkdir(exist_ok=True)
            self._save_shard(self.word_index, word_dir / "index.apex", False) 
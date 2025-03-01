import re
import msgpack
from pathlib import Path
from collections import defaultdict
from typing import Dict, Union, Optional, List

from pyroaring import BitMap
from functools import lru_cache
import xxhash

from .base import BaseIndex


class InvertedIndex(BaseIndex):
    """Inverted index implementation"""
    
    def __init__(self, 
                 index_dir: Optional[Path] = None,
                 max_chinese_length: int = 4,
                 min_term_length: int = 2,
                 buffer_size: int = 100000,
                 shard_bits: int = 8,
                 cache_size: int = 1000):
        """
        Initialize the inverted index
        
        Args:
            index_dir: The directory to store the index files
            max_chinese_length: The maximum length of Chinese substrings
            min_term_length: The minimum length of terms
            buffer_size: The size of the memory buffer
            shard_bits: The number of bits for the shard
            cache_size: The size of the cache
        """
        self.index_dir = index_dir
        self.max_chinese_length = max_chinese_length
        self.min_term_length = min_term_length
        self.buffer_size = buffer_size
        
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.modified_keys = set()
        
        # Main index and buffer
        self.word_index = defaultdict(BitMap)
        self.index_buffer = defaultdict(BitMap)
        
        # Chinese processing cache
        self._global_chinese_cache = {}
        
        self.shard_bits = shard_bits
        self.shard_count = 1 << shard_bits
        self.cache_size = cache_size
        
        self._init_cache()

    def _init_cache(self) -> None:
        """Initialize the LRU cache"""
        @lru_cache(maxsize=self.cache_size)
        def get_bitmap(term: str) -> Optional[BitMap]:
            shard_id = self._get_shard_id(term)
            return self._load_term_bitmap(term, shard_id)
        
        self._bitmap_cache = get_bitmap

    def _get_shard_id(self, term: str) -> int:
        """Calculate the shard ID of the term"""
        return xxhash.xxh32(term.encode()).intdigest() & (self.shard_count - 1)

    def _load_term_bitmap(self, term: str, shard_id: int) -> Optional[BitMap]:
        """Load the bitmap of the term from the disk"""
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
        """Add the document terms to the index"""
        for field_value in terms.values():
            field_str = str(field_value).lower()
            
            # Process the full field
            if len(field_str) >= self.min_term_length:
                self.index_buffer[field_str].add(doc_id)
            
            # Process Chinese
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
            
            # Process the phrase
            if ' ' in field_str:
                self.index_buffer[field_str].add(doc_id)
                words = field_str.split()
                for word in words:
                    if len(word) >= self.min_term_length and not self.chinese_pattern.search(word):
                        self.index_buffer[word].add(doc_id)

        # Merge the buffer when it reaches the threshold
        if len(self.index_buffer) >= self.buffer_size:
            self.merge_buffer()

    def search(self, query: str) -> BitMap:
        """Search for a query"""
        # Quick path - return directly
        query = query.strip().lower()
        if not query:
            return BitMap()
        
        # Modify the exact match part
        result = self._bitmap_cache(query)
        if result is not None:
            return result.copy()  # Return a copy to avoid modifying the cache
        
        # 2. Phrase query optimization - faster than Chinese query, so process it first
        if ' ' in query:
            words = query.split()
            if not words:
                return BitMap()
            
            # Pre-allocate the list to avoid dynamic growth
            results = []
            min_size = float('inf')
            min_idx = 0
            
            # Get all the document sets of all the words at once
            for i, word in enumerate(words):
                if len(word) < self.min_term_length:
                    continue
                    
                docs = self._bitmap_cache(word)
                if docs is None:
                    return BitMap()  # Quick failure
                
                size = len(docs)
                if size == 0:
                    return BitMap()
                
                if size < min_size:
                    min_size = size
                    min_idx = len(results)
                
                results.append(docs)
            
            if not results:
                return BitMap()
            
            # Optimization: directly use the smallest result set
            if min_idx > 0:
                results[0], results[min_idx] = results[min_idx], results[0]
            
            # Quick path: only one word
            if len(results) == 1:
                return results[0]
            
            # Efficient intersection
            result = results[0]
            for other in results[1:]:
                result &= other
                if not result:  # Return empty result early
                    return BitMap()
            
            return result
        
        # 3. Chinese query optimization
        if self.chinese_pattern.search(query):
            n = len(query)
            if n < self.min_term_length:
                return BitMap()
            
            # Optimization: directly use the longest possible substring
            max_len = min(n, self.max_chinese_length)
            
            # 3.1 Try the longest match
            for i in range(n - max_len + 1):
                substr = query[i:i + max_len]
                result = self._bitmap_cache(substr)
                if result is not None:
                    if len(result) < 1000:  # Return the result when it is small
                        return result
                    # Save the first match result
                    first_match = result
                    
                    # 3.2 Try to intersect with the adjacent substring
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
            
            # 3.3 Fall back to the minimum length match
            for i in range(n - self.min_term_length + 1):
                substr = query[i:i + self.min_term_length]
                result = self._bitmap_cache(substr)
                if result is not None:
                    return result
        
        return BitMap()

    def remove_document(self, doc_id: int) -> None:
        """Remove the document from the index"""
        if not self.index_dir:
            return
        
        # Remove from the buffer
        keys_to_remove = []
        for key, doc_ids in self.index_buffer.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                if not doc_ids:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.index_buffer[key]
        
        # Remove from the shards
        shards_dir = self.index_dir / 'shards'
        if shards_dir.exists():
            for shard_path in shards_dir.glob('*.apex'):
                try:
                    # Read the shard data
                    with open(shard_path, 'rb') as f:
                        meta_size = int.from_bytes(f.read(4), 'big')
                        meta_data = f.read(meta_size)
                        meta = msgpack.unpackb(meta_data, raw=False)
                        
                        modified = False
                        new_data = {}
                        new_meta = {}
                        offset = 0
                        
                        # Process each term
                        for term, (term_offset, size) in meta.items():
                            f.seek(4 + meta_size + term_offset)
                            bitmap_data = f.read(size)
                            bitmap = BitMap.deserialize(bitmap_data)
                            
                            if doc_id in bitmap:
                                bitmap.discard(doc_id)
                                modified = True
                                self.modified_keys.add(term)
                                
                            if bitmap:  # Only save non-empty bitmap
                                bitmap_data = bitmap.serialize()
                                new_data[term] = bitmap_data
                                new_meta[term] = (offset, len(bitmap_data))
                                offset += len(bitmap_data)
                        
                        # If there are modifications, rewrite the shard file
                        if modified:
                            if new_data:
                                with open(shard_path, 'wb') as f:
                                    meta_data = msgpack.packb(new_meta, use_bin_type=True)
                                    f.write(len(meta_data).to_bytes(4, 'big'))
                                    f.write(meta_data)
                                    for bitmap_data in new_data.values():
                                        f.write(bitmap_data)
                            else:
                                # If the shard is empty, delete the file
                                shard_path.unlink()
                                
                except Exception as e:
                    print(f"Error processing shard {shard_path}: {e}")
                    continue
        
        # Remove from the word index
        keys_to_remove = []
        for key, doc_ids in self.word_index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
                self.modified_keys.add(key)
                if not doc_ids:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.word_index[key]
        
        # Clear the cache
        self._bitmap_cache.cache_clear()

    def update_terms(self, doc_id: int, terms: Dict[str, Union[str, int, float]]) -> None:
        """Update the terms of the document"""
        # 获取文档当前的所有词条
        current_terms = set()
        
        # 从缓冲区中获取
        for term, bitmap in self.index_buffer.items():
            if doc_id in bitmap:
                current_terms.add(term)
                
        # 从分片文件中获取
        if self.index_dir:
            shards_dir = self.index_dir / 'shards'
            if shards_dir.exists():
                for shard_path in shards_dir.glob('*.apex'):
                    try:
                        with open(shard_path, 'rb') as f:
                            meta_size = int.from_bytes(f.read(4), 'big')
                            meta_data = f.read(meta_size)
                            meta = msgpack.unpackb(meta_data, raw=False)
                            
                            for term, (offset, size) in meta.items():
                                f.seek(4 + meta_size + offset)
                                bitmap_data = f.read(size)
                                bitmap = BitMap.deserialize(bitmap_data)
                                if doc_id in bitmap:
                                    current_terms.add(term)
                    except Exception as e:
                        print(f"Error reading shard {shard_path}: {e}")
                        continue
        
        # 从词索引中获取
        for term, bitmap in self.word_index.items():
            if doc_id in bitmap:
                current_terms.add(term)
        
        # 生成新的词条集合
        new_terms = set()
        for field_value in terms.values():
            field_str = str(field_value).lower()
            
            # 处理完整字段
            if len(field_str) >= self.min_term_length:
                new_terms.add(field_str)
            
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
                
                new_terms.update(self._global_chinese_cache[seg])
            
            # 处理短语
            if ' ' in field_str:
                new_terms.add(field_str)
                words = field_str.split()
                for word in words:
                    if len(word) >= self.min_term_length and not self.chinese_pattern.search(word):
                        new_terms.add(word)
        
        # 找出需要删除和添加的词条
        terms_to_remove = current_terms - new_terms
        terms_to_add = new_terms - current_terms
        
        # 删除旧词条
        for term in terms_to_remove:
            # 从缓冲区删除
            if term in self.index_buffer:
                self.index_buffer[term].discard(doc_id)
                self.modified_keys.add(term)
                if not self.index_buffer[term]:
                    del self.index_buffer[term]
            
            # 从词索引中删除
            if term in self.word_index:
                self.word_index[term].discard(doc_id)
                self.modified_keys.add(term)
                if not self.word_index[term]:
                    del self.word_index[term]
            
            # 从分片文件中删除
            if self.index_dir:
                shard_id = self._get_shard_id(term)
                shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.apex'
                if shard_path.exists():
                    try:
                        with open(shard_path, 'rb') as f:
                            meta_size = int.from_bytes(f.read(4), 'big')
                            meta_data = f.read(meta_size)
                            meta = msgpack.unpackb(meta_data, raw=False)
                            
                            if term in meta:
                                offset, size = meta[term]
                                f.seek(4 + meta_size + offset)
                                bitmap_data = f.read(size)
                                bitmap = BitMap.deserialize(bitmap_data)
                                bitmap.discard(doc_id)
                                self.modified_keys.add(term)
                                
                                # 更新分片文件
                                if bitmap:
                                    new_data = {term: bitmap.serialize()}
                                    new_meta = {term: (0, len(new_data[term]))}
                                    with open(shard_path, 'wb') as f:
                                        meta_data = msgpack.packb(new_meta, use_bin_type=True)
                                        f.write(len(meta_data).to_bytes(4, 'big'))
                                        f.write(meta_data)
                                        f.write(new_data[term])
                                else:
                                    # 如果位图为空，删除文件
                                    shard_path.unlink()
                    except Exception as e:
                        print(f"Error updating shard {shard_path}: {e}")
                        continue
        
        # 添加新词条
        for term in terms_to_add:
            self.index_buffer[term].add(doc_id)
            self.modified_keys.add(term)
            
            # 如果是词条，也添加到词索引中
            if ' ' in term:
                words = term.split()
                for word in words:
                    if len(word) >= self.min_term_length and not self.chinese_pattern.search(word):
                        self.word_index[word].add(doc_id)
                        self.modified_keys.add(word)
        
        # 合并缓冲区（如果需要）
        if len(self.index_buffer) >= self.buffer_size:
            self.merge_buffer()
        
        # 清理缓存
        self._bitmap_cache.cache_clear()

    def batch_update_terms(self, doc_ids: List[int], docs_terms: List[Dict[str, Union[str, int, float]]]) -> None:
        """批量更新多个文档的词条
        
        Args:
            doc_ids: 文档ID列表
            docs_terms: 文档词条列表，与doc_ids一一对应
        """
        if len(doc_ids) != len(docs_terms):
            raise ValueError("文档ID列表和文档词条列表长度不匹配")
            
        if not doc_ids:
            return
            
        # 收集所有文档的当前词条
        all_current_terms = defaultdict(set)  # term -> set(doc_ids)
        all_new_terms = defaultdict(set)      # term -> set(doc_ids)
        
        # 1. 收集所有文档的当前词条
        # 从缓冲区收集
        for term, bitmap in self.index_buffer.items():
            for doc_id in doc_ids:
                if doc_id in bitmap:
                    all_current_terms[term].add(doc_id)
        
        # 从分片文件收集
        if self.index_dir:
            shards_dir = self.index_dir / 'shards'
            if shards_dir.exists():
                for shard_path in shards_dir.glob('*.apex'):
                    try:
                        with open(shard_path, 'rb') as f:
                            meta_size = int.from_bytes(f.read(4), 'big')
                            meta_data = f.read(meta_size)
                            meta = msgpack.unpackb(meta_data, raw=False)
                            
                            for term, (offset, size) in meta.items():
                                f.seek(4 + meta_size + offset)
                                bitmap_data = f.read(size)
                                bitmap = BitMap.deserialize(bitmap_data)
                                
                                for doc_id in doc_ids:
                                    if doc_id in bitmap:
                                        all_current_terms[term].add(doc_id)
                    except Exception as e:
                        print(f"Error reading shard {shard_path}: {e}")
                        continue
        
        # 从词索引收集
        for term, bitmap in self.word_index.items():
            for doc_id in doc_ids:
                if doc_id in bitmap:
                    all_current_terms[term].add(doc_id)
        
        # 2. 生成所有文档的新词条
        for i, (doc_id, terms) in enumerate(zip(doc_ids, docs_terms)):
            # 处理每个文档的词条
            for field_value in terms.values():
                field_str = str(field_value).lower()
                
                # 处理完整字段
                if len(field_str) >= self.min_term_length:
                    all_new_terms[field_str].add(doc_id)
                
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
                        all_new_terms[substr].add(doc_id)
                
                # 处理短语
                if ' ' in field_str:
                    all_new_terms[field_str].add(doc_id)
                    words = field_str.split()
                    for word in words:
                        if len(word) >= self.min_term_length and not self.chinese_pattern.search(word):
                            all_new_terms[word].add(doc_id)
        
        # 3. 计算需要更新的词条
        # 对于每个词条，找出需要删除和添加的文档ID
        terms_to_update = set(all_current_terms.keys()) | set(all_new_terms.keys())
        
        # 按分片分组，减少文件I/O
        sharded_updates = defaultdict(lambda: defaultdict(lambda: {'add': set(), 'remove': set()}))
        
        for term in terms_to_update:
            current_docs = all_current_terms.get(term, set())
            new_docs = all_new_terms.get(term, set())
            
            # 需要删除的文档ID
            docs_to_remove = current_docs - new_docs
            # 需要添加的文档ID
            docs_to_add = new_docs - current_docs
            
            if not docs_to_remove and not docs_to_add:
                continue  # 没有变化，跳过
                
            # 更新缓冲区
            if term in self.index_buffer:
                for doc_id in docs_to_remove:
                    self.index_buffer[term].discard(doc_id)
                for doc_id in docs_to_add:
                    self.index_buffer[term].add(doc_id)
                if not self.index_buffer[term]:
                    del self.index_buffer[term]
            else:
                for doc_id in docs_to_add:
                    self.index_buffer[term].add(doc_id)
            
            # 更新词索引
            if ' ' in term:
                words = term.split()
                for word in words:
                    if len(word) >= self.min_term_length and not self.chinese_pattern.search(word):
                        if word in self.word_index:
                            for doc_id in docs_to_remove:
                                self.word_index[word].discard(doc_id)
                            for doc_id in docs_to_add:
                                self.word_index[word].add(doc_id)
                            if not self.word_index[word]:
                                del self.word_index[word]
                        else:
                            for doc_id in docs_to_add:
                                self.word_index[word].add(doc_id)
            
            # 按分片分组
            if self.index_dir:
                shard_id = self._get_shard_id(term)
                sharded_updates[shard_id][term]['remove'].update(docs_to_remove)
                sharded_updates[shard_id][term]['add'].update(docs_to_add)
            
            # 标记为已修改
            self.modified_keys.add(term)
        
        # 4. 更新分片文件
        if self.index_dir:
            for shard_id, terms in sharded_updates.items():
                shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.apex'
                if not shard_path.exists() and not any(terms[term]['add'] for term in terms):
                    continue  # 没有需要添加的文档，且分片不存在，跳过
                
                # 读取现有分片数据
                existing_meta = {}
                existing_data = {}
                
                if shard_path.exists():
                    try:
                        with open(shard_path, 'rb') as f:
                            meta_size = int.from_bytes(f.read(4), 'big')
                            meta_data = f.read(meta_size)
                            existing_meta = msgpack.unpackb(meta_data, raw=False)
                            
                            for term, (offset, size) in existing_meta.items():
                                if term in terms:  # 只读取需要更新的词条
                                    f.seek(4 + meta_size + offset)
                                    bitmap_data = f.read(size)
                                    bitmap = BitMap.deserialize(bitmap_data)
                                    
                                    # 更新位图
                                    for doc_id in terms[term]['remove']:
                                        bitmap.discard(doc_id)
                                    for doc_id in terms[term]['add']:
                                        bitmap.add(doc_id)
                                    
                                    if bitmap:
                                        existing_data[term] = bitmap.serialize()
                                else:
                                    f.seek(4 + meta_size + offset)
                                    existing_data[term] = f.read(size)
                    except Exception as e:
                        print(f"Error reading shard {shard_path}: {e}")
                        continue
                
                # 添加新词条
                for term in terms:
                    if term not in existing_data and terms[term]['add']:
                        bitmap = BitMap(terms[term]['add'])
                        existing_data[term] = bitmap.serialize()
                
                # 删除空位图
                for term in list(existing_data.keys()):
                    if term in terms and not terms[term]['add'] and term in existing_meta:
                        bitmap = BitMap.deserialize(existing_data[term])
                        for doc_id in terms[term]['remove']:
                            bitmap.discard(doc_id)
                        if not bitmap:
                            del existing_data[term]
                
                # 保存更新后的分片
                if existing_data:
                    # 重建元数据
                    new_meta = {}
                    offset = 0
                    for term, data in existing_data.items():
                        new_meta[term] = (offset, len(data))
                        offset += len(data)
                    
                    # 写入文件
                    shard_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(shard_path, 'wb') as f:
                        meta_data = msgpack.packb(new_meta, use_bin_type=True)
                        f.write(len(meta_data).to_bytes(4, 'big'))
                        f.write(meta_data)
                        for data in existing_data.values():
                            f.write(data)
                elif shard_path.exists():
                    # 如果没有数据，删除文件
                    shard_path.unlink()
        
        # 5. 合并缓冲区（如果需要）
        if len(self.index_buffer) >= self.buffer_size:
            self.merge_buffer()
        
        # 6. 清理缓存
        self._bitmap_cache.cache_clear()

    def merge_buffer(self) -> None:
        """Merge the buffer"""
        if not self.index_buffer:
            return
            
        # Group by shard
        sharded_buffer = defaultdict(lambda: defaultdict(BitMap))
        for term, bitmap in self.index_buffer.items():
            shard_id = self._get_shard_id(term)
            sharded_buffer[shard_id][term] = bitmap
        
        # Write to shards
        for shard_id, terms in sharded_buffer.items():
            self._merge_shard(shard_id, terms)
        
        # Clear the buffer
        self.index_buffer.clear()
        # Clear the cache
        self._bitmap_cache.cache_clear()

    def _merge_shard(self, shard_id: int, terms: Dict[str, BitMap]) -> None:
        """Merge the data of a single shard"""
        shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.apex'
        
        # Read the existing data
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
        
        # Merge the data
        new_data = {}
        new_meta = {}
        offset = 0
        
        for term, bitmap_data in existing_data.items():
            if term in terms:  # Need to update
                bitmap = BitMap.deserialize(bitmap_data)
                bitmap |= terms[term]
                bitmap_data = bitmap.serialize()
                del terms[term]
            
            new_data[term] = bitmap_data
            new_meta[term] = (offset, len(bitmap_data))
            offset += len(bitmap_data)
        
        # Add new terms
        for term, bitmap in terms.items():
            bitmap_data = bitmap.serialize()
            new_data[term] = bitmap_data
            new_meta[term] = (offset, len(bitmap_data))
            offset += len(bitmap_data)
        
        # Save
        if new_data:
            shard_path.parent.mkdir(exist_ok=True)
            with open(shard_path, 'wb') as f:
                meta_data = msgpack.packb(new_meta, use_bin_type=True)
                f.write(len(meta_data).to_bytes(4, 'big'))
                f.write(meta_data)
                for bitmap_data in new_data.values():
                    f.write(bitmap_data)

    def save(self, incremental: bool = True) -> None:
        """Save the index"""
        if not self.index_dir:
            return
            
        # Save the buffer data by shard
        if self.index_buffer:
            self.merge_buffer()
        
        # Save the word index
        if self.word_index:
            word_dir = self.index_dir / 'word'
            word_dir.mkdir(exist_ok=True)
            self._save_shard(self.word_index, word_dir / "index.apex", incremental)
        
        if incremental:
            self.modified_keys.clear()

    def _save_shard(self, shard_data: Dict[str, BitMap], shard_path: Path, incremental: bool) -> None:
        """Save a single shard
        
        Args:
            shard_data: The shard data
            shard_path: The shard path
            incremental: Whether to save incrementally
        """
        if not shard_data:
            if shard_path.exists():
                shard_path.unlink()
            return
            
        # Process the incremental update
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
        
        # Prepare new data
        data = {}
        meta = {}
        offset = 0
        
        # Process the existing data
        for key, bitmap_data in existing_data.items():
            meta[key] = (offset, len(bitmap_data))
            data[key] = bitmap_data
            offset += len(bitmap_data)
        
        # Process the new data
        for key, bitmap in shard_data.items():
            if not bitmap:
                continue
            if not incremental or key in self.modified_keys:
                bitmap_data = bitmap.serialize()
                data[key] = bitmap_data
                meta[key] = (offset, len(bitmap_data))
                offset += len(bitmap_data)
        
        # Save
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
        """Load the index"""
        if not self.index_dir:
            return False
            
        try:
            # Load shards
            shards_dir = self.index_dir / 'shards'
            if not shards_dir.exists():
                return False
                
            for shard_path in shards_dir.glob('*.apex'):
                self._load_shard(shard_path)
            
            # Load the word index
            word_dir = self.index_dir / 'word'
            if word_dir.exists():
                word_index_path = word_dir / "index.apex"
                if word_index_path.exists():
                    self._load_shard(word_index_path, is_word_index=True)
            
            return True
        except Exception as e:
            print(f"Failed to load the index: {e}")
            return False

    def _load_shard(self, shard_path: Path, is_word_index: bool = False) -> None:
        """Load a single shard
        
        Args:
            shard_path: The shard path
            is_word_index: Whether it is the word index
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
                            # Add the data to the buffer
                            self.index_buffer[key] |= bitmap
            
                # If it is not the word index, merge to the disk
                if not is_word_index and self.index_buffer:
                    self.merge_buffer()
                
        except Exception as e:
            print(f"Failed to load the shard {shard_path}: {e}")

    def build_word_index(self) -> None:
        """Build the word index"""
        if not self.index_dir:
            return
        
        temp_word_index = defaultdict(set)
        shards_dir = self.index_dir / 'shards'
        
        # Read the data from all shards
        if shards_dir.exists():
            for shard_path in shards_dir.glob('*.apex'):
                try:
                    with open(shard_path, 'rb') as f:
                        meta_size = int.from_bytes(f.read(4), 'big')
                        meta_data = f.read(meta_size)
                        meta = msgpack.unpackb(meta_data, raw=False)
                        
                        for term, (offset, size) in meta.items():
                            if ' ' in term:  # Only process the word with space
                                f.seek(4 + meta_size + offset)
                                bitmap_data = f.read(size)
                                bitmap = BitMap.deserialize(bitmap_data)
                                
                                # Process the words in the phrase
                                words = term.split()
                                doc_ids = list(bitmap)
                                for word in words:
                                    if not self.chinese_pattern.search(word) and len(word) >= self.min_term_length:
                                        temp_word_index[word].update(doc_ids)
                except Exception as e:
                    print(f"Failed to process the shard {shard_path}: {e}")
                    continue
        
        # Process the data in the buffer
        for term, bitmap in self.index_buffer.items():
            if ' ' in term:
                words = term.split()
                doc_ids = list(bitmap)
                for word in words:
                    if not self.chinese_pattern.search(word) and len(word) >= self.min_term_length:
                        temp_word_index[word].update(doc_ids)
        
        # Convert to BitMap and save
        self.word_index.clear()
        for word, doc_ids in temp_word_index.items():
            if doc_ids:
                self.word_index[word] = BitMap(doc_ids)
        
        # Save the word index
        if self.index_dir:
            word_dir = self.index_dir / 'word'
            word_dir.mkdir(exist_ok=True)
            self._save_shard(self.word_index, word_dir / "index.apex", False) 
import re
import msgpack
from pathlib import Path
from collections import defaultdict
from typing import Dict, Union, Optional, List, Tuple

from pyroaring import BitMap
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
                 cache_size: int = 1000,
                 fuzzy_threshold: float = 0.7,
                 fuzzy_max_distance: int = 2):
        """
        Initialize the inverted index
        
        Args:
            index_dir: The directory to store the index files
            max_chinese_length: The maximum length of Chinese substrings
            min_term_length: The minimum length of terms
            buffer_size: The size of the memory buffer
            shard_bits: The number of bits for the shard
            cache_size: The size of the cache
            fuzzy_threshold: The similarity threshold for fuzzy search (0.0-1.0)
            fuzzy_max_distance: The maximum edit distance for fuzzy search
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
        
        # Fuzzy search parameters
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_max_distance = fuzzy_max_distance
        self._fuzzy_cache = {}  # Cache for fuzzy search results
        self._term_cache = None  # Cache for all terms, built on demand
        
        self._init_cache()

    def _init_cache(self) -> None:
        """Initialize the optimized cache"""
        # 使用普通字典代替LRU缓存，减少查找开销
        self._fast_cache = {}
        self._cache_hits = 0
        self._cache_max_size = self.cache_size
        
        def get_bitmap_fast(term: str) -> Optional[BitMap]:
            # 快速缓存查找
            if term in self._fast_cache:
                self._cache_hits += 1
                return self._fast_cache[term]
            
            # 首先检查缓冲区 - 避免不必要的拷贝
            if term in self.index_buffer:
                result = self.index_buffer[term]
                # 缓存结果（不拷贝，直接引用）
                self._cache_bitmap(term, result)
                return result
            
            # 检查内存索引（内存模式）
            if term in self.word_index:
                result = self.word_index[term] 
                self._cache_bitmap(term, result)
                return result
            
            # 如果是磁盘模式，从磁盘加载
            if self.index_dir:
                shard_id = self._get_shard_id(term)
                result = self._load_term_bitmap(term, shard_id)
                if result is not None:
                    self._cache_bitmap(term, result)
                return result
            
            return None
        
        self._bitmap_cache = get_bitmap_fast
    
    def _cache_bitmap(self, term: str, bitmap: BitMap) -> None:
        """缓存bitmap，管理缓存大小"""
        if len(self._fast_cache) >= self._cache_max_size:
            # 清理一半缓存（简单策略）
            keys_to_remove = list(self._fast_cache.keys())[:self._cache_max_size // 2]
            for key in keys_to_remove:
                del self._fast_cache[key]
        
        self._fast_cache[term] = bitmap

    def _get_shard_id(self, term: str) -> int:
        """Calculate the shard ID of the term"""
        return xxhash.xxh32(term.encode()).intdigest() & (self.shard_count - 1)

    def _load_term_bitmap(self, term: str, shard_id: int) -> Optional[BitMap]:
        """Load the bitmap of the term from the disk"""
        if not self.index_dir:
            return None
            
        shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.nfts'
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
            
            # Process English parts in mixed Chinese-English text
            english_parts = re.findall(r'[a-zA-Z]+', field_str)
            for eng_part in english_parts:
                if len(eng_part) >= self.min_term_length:
                    self.index_buffer[eng_part.lower()].add(doc_id)
            
            # Process the phrase
            if ' ' in field_str:
                self.index_buffer[field_str].add(doc_id)
                words = field_str.split()
                for word in words:
                    # 处理纯英文单词
                    if len(word) >= self.min_term_length and not self.chinese_pattern.search(word):
                        self.index_buffer[word].add(doc_id)
                    # 处理中英文混合单词，提取英文部分
                    elif self.chinese_pattern.search(word):
                        # 提取英文字母序列
                        english_parts = re.findall(r'[a-zA-Z]+', word)
                        for eng_part in english_parts:
                            if len(eng_part) >= self.min_term_length:
                                self.index_buffer[eng_part.lower()].add(doc_id)

        # Merge the buffer when it reaches the threshold
        if len(self.index_buffer) >= self.buffer_size:
            self.merge_buffer()
        
        # 无效化词条缓存，因为添加了新词条
        self._invalidate_term_cache()

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """计算两个字符串的编辑距离（Levenshtein距离）"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    @staticmethod
    def _similarity_score(s1: str, s2: str) -> float:
        """计算两个字符串的相似度分数 (0.0-1.0)，针对中文优化"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # 基于编辑距离的相似度
        distance = InvertedIndex._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        edit_similarity = 1.0 - (distance / max_len)
        
        # 基于字符重叠的相似度 (特别适合中文)
        set1, set2 = set(s1), set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        char_overlap = intersection / union if union > 0 else 0.0
        
        # 基于最长公共子序列的相似度
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        lcs_len = lcs_length(s1, s2)
        lcs_similarity = lcs_len / max_len if max_len > 0 else 0.0
        
        # 混合相似度：取三种算法的最大值，更宽松地匹配中文
        # 这样"支持"和"一支"就能有合理的相似度了
        final_similarity = max(edit_similarity, char_overlap, lcs_similarity)
        
        return final_similarity

    def _get_all_terms(self) -> set:
        """获取所有索引中的词条，用于模糊搜索"""
        if self._term_cache is not None:
            return self._term_cache
        
        terms = set()
        
        # 从缓冲区获取词条
        terms.update(self.index_buffer.keys())
        
        # 从内存索引获取词条（内存模式）
        terms.update(self.word_index.keys())
        
        # 从磁盘分片获取词条（如果存在）
        if self.index_dir and (self.index_dir / 'shards').exists():
            shards_dir = self.index_dir / 'shards'
            for shard_path in shards_dir.glob('*.nfts'):
                try:
                    with open(shard_path, 'rb') as f:
                        meta_size = int.from_bytes(f.read(4), 'big')
                        meta_data = f.read(meta_size)
                        meta = msgpack.unpackb(meta_data, raw=False)
                        terms.update(meta.keys())
                except:
                    continue
        
        # 缓存结果
        self._term_cache = terms
        return terms

    def _invalidate_term_cache(self):
        """无效化词条缓存，在索引更新时调用"""
        self._term_cache = None

    def _fuzzy_search_terms(self, query: str, max_results: int = 20) -> List[Tuple[str, float]]:
        """在所有词条中进行模糊搜索，返回相似词条和相似度分数"""
        cache_key = f"{query}:{max_results}"
        if cache_key in self._fuzzy_cache:
            return self._fuzzy_cache[cache_key]
        
        all_terms = self._get_all_terms()
        candidates = []
        
        # 过滤长度相近的词条以提高效率
        query_len = len(query)
        max_len_diff = self.fuzzy_max_distance
        
        for term in all_terms:
            # 跳过长度差异过大的词条
            if abs(len(term) - query_len) > max_len_diff:
                continue
            
            # 快速预过滤：如果完全没有共同字符，跳过
            if not set(query) & set(term):
                continue
            
            similarity = self._similarity_score(query, term)
            if similarity >= self.fuzzy_threshold:
                candidates.append((term, similarity))
        
        # 按相似度降序排序，取前max_results个
        candidates.sort(key=lambda x: x[1], reverse=True)
        result = candidates[:max_results]
        
        # 缓存结果（限制缓存大小）
        if len(self._fuzzy_cache) > 1000:
            # 清理最旧的一半缓存
            oldest_keys = list(self._fuzzy_cache.keys())[:500]
            for key in oldest_keys:
                del self._fuzzy_cache[key]
        
        self._fuzzy_cache[cache_key] = result
        return result

    def search(self, query: str, enable_fuzzy: bool = False, min_results: int = 5) -> BitMap:
        """Search for a query - optimized version"""
        # 预处理优化：减少字符串操作
        if not query:
            return BitMap()
        
        # 只在必要时进行字符串转换
        if query != query.strip().lower():
            query = query.strip().lower()
            if not query:
                return BitMap()
        
        # 快速路径：单个英文单词（最常见的查询）
        if ' ' not in query and not self.chinese_pattern.search(query):
            result = self._bitmap_cache(query)
            if result is not None:
                # 对于读取操作，直接返回引用而不是拷贝
                return result
            else:
                return BitMap()  # 快速返回空结果
        
        # 直接匹配优化：避免不必要的拷贝
        result = self._bitmap_cache(query)
        if result is not None:
            # 对于读取操作，直接返回引用而不是拷贝
            return result
        
        # 2. Phrase query optimization - 支持中英文混合搜索
        if ' ' in query:
            words = query.split()
            if not words:
                return BitMap()
            
            # Pre-allocate the list to avoid dynamic growth
            results = []
            min_size = float('inf')
            min_idx = 0
            
            # 优化：预分配结果列表，提前检查词长度
            valid_words = [word for word in words if len(word) >= self.min_term_length]
            if not valid_words:
                return BitMap()
            
            # Get all the document sets of all the words at once
            for i, word in enumerate(valid_words):
                # 首先尝试直接匹配完整单词
                docs = self._bitmap_cache(word)
                
                # 如果直接匹配失败且单词包含中文，则尝试中文子串匹配
                if docs is None and self.chinese_pattern.search(word):
                    docs = self._search_chinese_word(word)
                
                if docs is None or len(docs) == 0:
                    return BitMap()  # Quick failure
                
                size = len(docs)
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
        exact_result = None
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
                        exact_result = result
                        break
                    # Save the first match result
                    first_match = result
                    
                    # 3.2 Try to intersect with the adjacent substring
                    if i > 0:
                        prev_substr = query[i-1:i-1+max_len]
                        prev_docs = self._bitmap_cache(prev_substr)
                        if prev_docs is not None:
                            temp = result & prev_docs
                            if temp:
                                exact_result = temp
                                break
                    
                    if i < n - max_len:
                        next_substr = query[i+1:i+1+max_len]
                        next_docs = self._bitmap_cache(next_substr)
                        if next_docs is not None:
                            temp = result & next_docs
                            if temp:
                                exact_result = temp
                                break
                    
                    exact_result = first_match
                    break
            
            # 3.3 Fall back to the minimum length match
            if exact_result is None:
                for i in range(n - self.min_term_length + 1):
                    substr = query[i:i + self.min_term_length]
                    result = self._bitmap_cache(substr)
                    if result is not None:
                        exact_result = result
                        break
        
        # 4. 模糊搜索（仅在启用且精确搜索结果不足时）
        if enable_fuzzy and (exact_result is None or len(exact_result) < min_results):
            fuzzy_result = self._perform_fuzzy_search(query)
            
            if exact_result is None:
                return fuzzy_result
            elif len(exact_result) < min_results and len(fuzzy_result) > 0:
                # 合并精确搜索和模糊搜索的结果
                return exact_result | fuzzy_result
        
        return exact_result if exact_result is not None else BitMap()

    def _search_chinese_word(self, word: str) -> Optional[BitMap]:
        """对包含中文的单词进行子串搜索"""
        n = len(word)
        if n < self.min_term_length:
            return None
        
        # 尝试不同长度的子串，从最长开始
        max_len = min(n, self.max_chinese_length)
        
        # 首先尝试最长匹配
        for i in range(n - max_len + 1):
            substr = word[i:i + max_len]
            result = self._bitmap_cache(substr)
            if result is not None:
                if len(result) < 1000:  # 如果结果较少，直接返回
                    return result
                # 保存第一个匹配结果
                first_match = result
                
                # 尝试与相邻子串交集以获得更精确的结果
                if i > 0:
                    prev_substr = word[i-1:i-1+max_len]
                    prev_docs = self._bitmap_cache(prev_substr)
                    if prev_docs is not None:
                        temp = result & prev_docs
                        if temp:
                            return temp
                
                if i < n - max_len:
                    next_substr = word[i+1:i+1+max_len]
                    next_docs = self._bitmap_cache(next_substr)
                    if next_docs is not None:
                        temp = result & next_docs
                        if temp:
                            return temp
                
                return first_match
        
        # 如果最长匹配失败，尝试最小长度匹配
        for i in range(n - self.min_term_length + 1):
            substr = word[i:i + self.min_term_length]
            result = self._bitmap_cache(substr)
            if result is not None:
                return result
        
        return None

    def _perform_fuzzy_search(self, query: str) -> BitMap:
        """执行模糊搜索并返回合并的结果"""
        similar_terms = self._fuzzy_search_terms(query)
        if not similar_terms:
            return BitMap()
        
        # 合并所有相似词条的结果
        result = BitMap()
        for term, similarity in similar_terms:
            term_result = self._bitmap_cache(term)
            if term_result is not None:
                # 可以根据相似度加权，但为了性能简单起见，直接合并
                result |= term_result
        
        return result

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
            for shard_path in shards_dir.glob('*.nfts'):
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
        self._fast_cache.clear()
        
        # Ensure buffer changes are immediately available for search
        # This is especially important on Windows platforms
        self.merge_buffer()

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
                for shard_path in shards_dir.glob('*.nfts'):
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
                shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.nfts'
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
        
        # 强制合并缓冲区以确保更新立即生效
        # 这对于Windows平台特别重要
        self.merge_buffer()
        
        # 清理缓存
        self._fast_cache.clear()

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
                for shard_path in shards_dir.glob('*.nfts'):
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
                shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.nfts'
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
        
        # 5. 强制合并缓冲区以确保批量更新立即生效
        # 这对于Windows平台特别重要
        self.merge_buffer()
        
        # 6. 清理缓存
        self._fast_cache.clear()

    def merge_buffer(self) -> None:
        """Merge the buffer"""
        if not self.index_buffer:
            return
        
        # 如果是内存模式（index_dir为None），直接合并到内存索引
        if not self.index_dir:
            for term, bitmap in self.index_buffer.items():
                if term in self.word_index:
                    self.word_index[term] |= bitmap
                else:
                    self.word_index[term] = bitmap.copy()
            
            # Clear the buffer
            self.index_buffer.clear()
            # Clear the cache
            self._fast_cache.clear()
            # 无效化词条缓存
            self._invalidate_term_cache()
            return
            
        # 磁盘模式：分片处理
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
        self._fast_cache.clear()
        # 无效化词条缓存
        self._invalidate_term_cache()

    def _merge_shard(self, shard_id: int, terms: Dict[str, BitMap]) -> None:
        """Merge the data of a single shard"""
        shard_path = self.index_dir / 'shards' / f'shard_{shard_id}.nfts'
        
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
            self._save_shard(self.word_index, word_dir / "index.nfts", incremental)
        
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
                
            for shard_path in shards_dir.glob('*.nfts'):
                self._load_shard(shard_path)
            
            # Load the word index
            word_dir = self.index_dir / 'word'
            if word_dir.exists():
                word_index_path = word_dir / "index.nfts"
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
            for shard_path in shards_dir.glob('*.nfts'):
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
            self._save_shard(self.word_index, word_dir / "index.nfts", False) 

    def batch_remove_document(self, doc_ids: List[int]) -> None:
        """批量删除多个文档
        
        Args:
            doc_ids: 要删除的文档ID列表
        """
        if not doc_ids:
            return
            
        if not self.index_dir:
            return
            
        # 从缓冲区中删除
        keys_to_remove = []
        for key, bitmap in self.index_buffer.items():
            modified = False
            for doc_id in doc_ids:
                if doc_id in bitmap:
                    bitmap.discard(doc_id)
                    modified = True
            
            if modified:
                self.modified_keys.add(key)
                if not bitmap:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.index_buffer[key]
        
        # 从分片文件中删除
        shards_dir = self.index_dir / 'shards'
        if shards_dir.exists():
            # 按分片分组，减少文件I/O
            sharded_updates = defaultdict(set)
            
            # 收集所有需要更新的分片
            for shard_path in shards_dir.glob('*.nfts'):
                try:
                    with open(shard_path, 'rb') as f:
                        meta_size = int.from_bytes(f.read(4), 'big')
                        meta_data = f.read(meta_size)
                        meta = msgpack.unpackb(meta_data, raw=False)
                        
                        modified = False
                        new_data = {}
                        new_meta = {}
                        offset = 0
                        
                        # 处理每个词条
                        for term, (term_offset, size) in meta.items():
                            f.seek(4 + meta_size + term_offset)
                            bitmap_data = f.read(size)
                            bitmap = BitMap.deserialize(bitmap_data)
                            
                            term_modified = False
                            for doc_id in doc_ids:
                                if doc_id in bitmap:
                                    bitmap.discard(doc_id)
                                    term_modified = True
                            
                            if term_modified:
                                modified = True
                                self.modified_keys.add(term)
                                
                            if bitmap:  # 只保存非空位图
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
                    print(f"Error processing shard {shard_path}: {e}")
                    continue
        
        # 从词索引中删除
        keys_to_remove = []
        for key, bitmap in self.word_index.items():
            modified = False
            for doc_id in doc_ids:
                if doc_id in bitmap:
                    bitmap.discard(doc_id)
                    modified = True
            
            if modified:
                self.modified_keys.add(key)
                if not bitmap:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.word_index[key]
        
        # 清理缓存
        self._fast_cache.clear()
        
        # 强制合并缓冲区以确保批量删除立即生效
        # 这对于Windows平台特别重要
        self.merge_buffer() 
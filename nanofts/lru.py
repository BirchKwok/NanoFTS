from collections import defaultdict

from pyroaring import BitMap

class LRUCache:
    """LRU缓存，用于管理内存中的活跃索引"""
    def __init__(self, maxsize: int = 10000):
        self.cache = {}
        self.maxsize = maxsize
        self.hits = defaultdict(int)
    
    def get(self, key: str) -> BitMap:
        if key in self.cache:
            self.hits[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key: str, value: BitMap):
        if len(self.cache) >= self.maxsize:
            # 移除最少使用的项
            lru_key = min(self.hits.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.hits[lru_key]
        self.cache[key] = value
        self.hits[key] = 1
        
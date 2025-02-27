from abc import ABC, abstractmethod
from typing import Dict, Union, List, Optional
from pathlib import Path
from pyroaring import BitMap

class BaseIndex(ABC):
    """索引的抽象基类"""
    
    @abstractmethod
    def __init__(self, 
                 index_dir: Optional[Path] = None,
                 max_chinese_length: int = 4,
                 min_term_length: int = 2,
                 buffer_size: int = 100000,
                 shard_bits: int = 8,
                 cache_size: int = 1000):
        """
        初始化索引
        
        Args:
            index_dir: 索引文件存储目录
            max_chinese_length: 中文子串最大长度
            min_term_length: 最小词条长度
            buffer_size: 内存缓冲区大小
            shard_bits: 分片位数
            cache_size: 缓存大小
        """
        pass

    @abstractmethod
    def add_terms(self, doc_id: int, terms: Dict[str, Union[str, int, float]]) -> None:
        """添加文档词条到索引"""
        pass

    @abstractmethod
    def search(self, query: str, score_threshold: Optional[float] = None) -> Union[BitMap, List[tuple[int, float]]]:
        """搜索查询"""
        pass

    @abstractmethod
    def remove_document(self, doc_id: int) -> None:
        """从索引中移除文档"""
        pass

    @abstractmethod
    def merge_buffer(self) -> None:
        """合并缓冲区"""
        pass

    @abstractmethod
    def save(self, incremental: bool = True) -> None:
        """保存索引"""
        pass

    @abstractmethod
    def load(self) -> bool:
        """加载索引"""
        pass

    @abstractmethod
    def build_word_index(self) -> None:
        """构建词组索引"""
        pass 
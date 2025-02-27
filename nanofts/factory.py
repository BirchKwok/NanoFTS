from enum import Enum
from pathlib import Path
from typing import Optional

from .inverted import InvertedIndex


class IndexType(Enum):
    """索引类型枚举"""
    INVERTED = "inverted"  # 倒排索引


class IndexFactory:
    """索引工厂类"""
    
    @staticmethod
    def create_index(index_type: IndexType,
                    index_dir: Optional[str] = None,
                    max_chinese_length: int = 4,
                    min_term_length: int = 2,
                    buffer_size: int = 100000,
                    shard_bits: int = 8,
                    cache_size: int = 1000,
                    **kwargs) -> InvertedIndex:
        """
        创建索引实例
        
        Args:
            index_type: 索引类型
            index_dir: 索引文件存储目录
            max_chinese_length: 中文子串最大长度
            min_term_length: 最小词条长度
            buffer_size: 内存缓冲区大小
            shard_bits: 分片位数
            cache_size: 缓存大小
            ngram_size: n-gram大小，仅用于NGram索引
            **kwargs: 其他参数
        """
        index_dir_path = Path(index_dir) if index_dir else None
        
        if index_type == IndexType.INVERTED:
            return InvertedIndex(
                index_dir=index_dir_path,
                max_chinese_length=max_chinese_length,
                min_term_length=min_term_length,
                buffer_size=buffer_size,
                shard_bits=shard_bits,
                cache_size=cache_size
            )
        else:
            raise ValueError(f"不支持的索引类型: {index_type}") 
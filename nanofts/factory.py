from enum import Enum
from pathlib import Path
from typing import Optional

from .inverted import InvertedIndex


class IndexType(Enum):
    """Index type enum"""
    INVERTED = "inverted"  # Inverted index


class IndexFactory:
    """Index factory class"""
    
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
        Create an index instance
        
        Args:
            index_type: The index type
            index_dir: The directory to store the index files
            max_chinese_length: The maximum length of Chinese substrings
            min_term_length: The minimum length of terms
            buffer_size: The size of the memory buffer
            shard_bits: The number of bits for the shard
            cache_size: The size of the cache
            ngram_size: The size of the n-gram, only used for NGram index
            **kwargs: Other parameters
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
            raise ValueError(f"Unsupported index type: {index_type}") 
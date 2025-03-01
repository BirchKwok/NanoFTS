from abc import ABC, abstractmethod
from typing import Dict, Union, List, Optional
from pathlib import Path
from pyroaring import BitMap

class BaseIndex(ABC):
    """Base class for all indexes"""
    
    @abstractmethod
    def __init__(self, 
                 index_dir: Optional[Path] = None,
                 max_chinese_length: int = 4,
                 min_term_length: int = 2,
                 buffer_size: int = 100000,
                 shard_bits: int = 8,
                 cache_size: int = 1000):
        """
        Initialize the index
        
        Args:
            index_dir: The directory to store the index files
            max_chinese_length: The maximum length of Chinese substrings
            min_term_length: The minimum length of terms
            buffer_size: The size of the memory buffer
            shard_bits: The number of bits for the shard
            cache_size: The size of the cache
        """
        pass

    @abstractmethod
    def add_terms(self, doc_id: int, terms: Dict[str, Union[str, int, float]]) -> None:
        """Add terms to the index"""
        pass

    @abstractmethod
    def search(self, query: str, score_threshold: Optional[float] = None) -> Union[BitMap, List[tuple[int, float]]]:
        """Search for a query"""
        pass

    @abstractmethod
    def remove_document(self, doc_id: int) -> None:
        """Remove a document from the index"""
        pass

    @abstractmethod
    def merge_buffer(self) -> None:
        """Merge the buffer"""
        pass

    @abstractmethod
    def save(self, incremental: bool = True) -> None:
        """Save the index"""
        pass

    @abstractmethod
    def load(self) -> bool:
        """Load the index"""
        pass

    @abstractmethod
    def build_word_index(self) -> None:
        """Build the word index"""
        pass 
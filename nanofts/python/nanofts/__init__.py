"""
NanoFTS - Ultra High-Performance Full-Text Search Engine

Optimized for billion-scale data with sub-millisecond search response.

Unified API:
- create_engine() - Create search engine
- SearchEngine - Search engine class (alias)
- SearchResult - Search result class (alias)

Main Features:
- LSM-Tree Architecture: No scale limits
- Incremental Writes: Real-time updates
- Fuzzy Search
- Zero-copy Result Handle
- Result Set Operations (AND/OR/NOT)
- NumPy Return Support
- Data Import Support (pandas, polars, arrow, parquet, csv)
"""

try:
    # Import from Rust extension
    from .nanofts import (
        create_engine as _create_engine,
        UnifiedEngine as _UnifiedEngine,
        ResultHandle,
        FuzzyConfig,
    )
except ImportError:
    # Development environment (direct import)
    from nanofts import (
        create_engine as _create_engine,
        UnifiedEngine as _UnifiedEngine,
        ResultHandle,
        FuzzyConfig,
    )


__version__ = "0.3.0"


class UnifiedEngine(_UnifiedEngine):
    """
    Unified Search Engine - Extended Version
    
    Adds data import capabilities on top of the Rust core engine:
    - from_pandas() - Import from pandas DataFrame
    - from_polars() - Import from Polars DataFrame
    - from_arrow() - Import from PyArrow Table
    - from_parquet() - Import from Parquet file
    - from_csv() - Import from CSV file
    """
    
    def from_pandas(self, df, id_column='id', text_columns=None):
        """
        Import data from a pandas DataFrame
        
        Args:
            df: pandas DataFrame
            id_column: ID column name, default 'id'
            text_columns: List of text column names to index, defaults to all columns except id_column
        
        Returns:
            int: Number of documents imported
        
        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'id': [1, 2, 3],
            ...     'title': ['Hello', 'World', 'Test'],
            ...     'content': ['Content 1', 'Content 2', 'Content 3']
            ... })
            >>> engine.from_pandas(df, id_column='id')
            3
        """
        if text_columns is None:
            text_columns = [col for col in df.columns if col != id_column]
        
        count = 0
        for _, row in df.iterrows():
            doc_id = int(row[id_column])
            fields = {col: str(row[col]) for col in text_columns if col in row.index}
            self.add_document(doc_id, fields)
            count += 1
        
        self.flush()
        return count
    
    def from_polars(self, df, id_column='id', text_columns=None):
        """
        Import data from a Polars DataFrame
        
        Args:
            df: polars DataFrame
            id_column: ID column name, default 'id'
            text_columns: List of text column names to index, defaults to all columns except id_column
        
        Returns:
            int: Number of documents imported
        
        Example:
            >>> import polars as pl
            >>> df = pl.DataFrame({
            ...     'id': [1, 2, 3],
            ...     'title': ['Hello', 'World', 'Test'],
            ...     'content': ['Content 1', 'Content 2', 'Content 3']
            ... })
            >>> engine.from_polars(df, id_column='id')
            3
        """
        if text_columns is None:
            text_columns = [col for col in df.columns if col != id_column]
        
        count = 0
        for row in df.iter_rows(named=True):
            doc_id = int(row[id_column])
            fields = {col: str(row[col]) for col in text_columns if col in row}
            self.add_document(doc_id, fields)
            count += 1
        
        self.flush()
        return count
    
    def from_arrow(self, table, id_column='id', text_columns=None):
        """
        Import data from a PyArrow Table
        
        Args:
            table: pyarrow Table
            id_column: ID column name, default 'id'
            text_columns: List of text column names to index, defaults to all columns except id_column
        
        Returns:
            int: Number of documents imported
        
        Example:
            >>> import pyarrow as pa
            >>> table = pa.Table.from_pydict({
            ...     'id': [1, 2, 3],
            ...     'title': ['Hello', 'World', 'Test']
            ... })
            >>> engine.from_arrow(table, id_column='id')
            3
        """
        if text_columns is None:
            text_columns = [col for col in table.column_names if col != id_column]
        
        # Convert to Python dict list
        df_dict = table.to_pydict()
        num_rows = len(df_dict[id_column])
        
        count = 0
        for i in range(num_rows):
            doc_id = int(df_dict[id_column][i])
            fields = {col: str(df_dict[col][i]) for col in text_columns if col in df_dict}
            self.add_document(doc_id, fields)
            count += 1
        
        self.flush()
        return count
    
    def from_parquet(self, path, id_column='id', text_columns=None):
        """
        Import data from a Parquet file
        
        Args:
            path: Parquet file path
            id_column: ID column name, default 'id'
            text_columns: List of text column names to index, defaults to all columns except id_column
        
        Returns:
            int: Number of documents imported
        
        Example:
            >>> engine.from_parquet("documents.parquet", id_column='id')
            1000
        """
        import pyarrow.parquet as pq
        
        table = pq.read_table(str(path))
        return self.from_arrow(table, id_column=id_column, text_columns=text_columns)
    
    def from_csv(self, path, id_column='id', text_columns=None, **csv_options):
        """
        Import data from a CSV file
        
        Args:
            path: CSV file path
            id_column: ID column name, default 'id'
            text_columns: List of text column names to index, defaults to all columns except id_column
            **csv_options: Additional arguments passed to pandas.read_csv
        
        Returns:
            int: Number of documents imported
        
        Example:
            >>> engine.from_csv("documents.csv", id_column='id')
            1000
            
            # With custom CSV options
            >>> engine.from_csv("documents.csv", id_column='id', encoding='utf-8', sep=';')
            1000
        """
        import pandas as pd
        
        df = pd.read_csv(str(path), **csv_options)
        return self.from_pandas(df, id_column=id_column, text_columns=text_columns)
    
    def from_json(self, path, id_column='id', text_columns=None, **json_options):
        """
        Import data from a JSON file
        
        Args:
            path: JSON file path (supports JSON Lines format)
            id_column: ID column name, default 'id'
            text_columns: List of text column names to index, defaults to all columns except id_column
            **json_options: Additional arguments passed to pandas.read_json
        
        Returns:
            int: Number of documents imported
        
        Example:
            >>> engine.from_json("documents.json", id_column='id')
            1000
            
            # JSON Lines format
            >>> engine.from_json("documents.jsonl", id_column='id', lines=True)
            1000
        """
        import pandas as pd
        
        df = pd.read_json(str(path), **json_options)
        return self.from_pandas(df, id_column=id_column, text_columns=text_columns)
    
    def from_dict(self, data, id_column='id', text_columns=None):
        """
        Import data from a list of dictionaries
        
        Args:
            data: List of dictionaries, each representing a document
            id_column: ID field name, default 'id'
            text_columns: List of text field names to index, defaults to all fields except id_column
        
        Returns:
            int: Number of documents imported
        
        Example:
            >>> data = [
            ...     {'id': 1, 'title': 'Hello', 'content': 'World'},
            ...     {'id': 2, 'title': 'Test', 'content': 'Document'}
            ... ]
            >>> engine.from_dict(data, id_column='id')
            2
        """
        if not data:
            return 0
        
        if text_columns is None:
            # Infer text columns from first document
            text_columns = [k for k in data[0].keys() if k != id_column]
        
        count = 0
        for row in data:
            doc_id = int(row[id_column])
            fields = {col: str(row[col]) for col in text_columns if col in row}
            self.add_document(doc_id, fields)
            count += 1
        
        self.flush()
        return count


def create_engine(
    index_file="",
    max_chinese_length=4,
    min_term_length=2,
    fuzzy_threshold=0.7,
    fuzzy_max_distance=2,
    track_doc_terms=False,
    drop_if_exists=False,
    lazy_load=True,
    cache_size=10000
):
    """
    Create a unified search engine
    
    Args:
        index_file: Index file path, empty string for memory-only mode
        max_chinese_length: Maximum Chinese n-gram length (default 4)
        min_term_length: Minimum term length (default 2)
        fuzzy_threshold: Fuzzy search similarity threshold (default 0.7)
        fuzzy_max_distance: Fuzzy search maximum edit distance (default 2)
        track_doc_terms: Whether to track document terms (default False)
        drop_if_exists: Whether to delete existing index file (default False)
        lazy_load: Whether to enable lazy loading mode (default True)
        cache_size: LRU cache size in lazy load mode (default 10000)
    
    Returns:
        UnifiedEngine: Search engine instance
    
    Example:
        # Default mode (lazy load, recommended)
        engine = create_engine("index.nfts")
        
        # Full load mode (suitable for small indexes)
        engine = create_engine("index.nfts", lazy_load=False)
        
        # Memory-only mode
        engine = create_engine("")
    """
    return UnifiedEngine(
        index_file,
        max_chinese_length,
        min_term_length,
        fuzzy_threshold,
        fuzzy_max_distance,
        track_doc_terms,
        drop_if_exists,
        lazy_load,
        cache_size
    )


# Aliases
SearchEngine = UnifiedEngine
SearchResult = ResultHandle


__all__ = [
    # Main API
    'create_engine',
    'SearchEngine',
    'SearchResult',
    # Actual class names
    'UnifiedEngine',
    'ResultHandle',
    'FuzzyConfig',
    # Version
    '__version__',
]

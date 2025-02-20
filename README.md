# NanoFTS

A lightweight full-text search engine implementation in Python, featuring efficient indexing and searching capabilities for both English and Chinese text.

## Features

- Lightweight and efficient full-text search implementation
- Pure Python with minimal dependencies (only requires `pyroaring` and `msgpack`)
- Support for both English and Chinese text
- Memory-efficient disk-based index storage with sharding
- Incremental indexing and real-time updates
- Case-insensitive search
- Phrase matching support
- Built-in LRU caching for frequently accessed terms
- Data import support from popular formats:
  - Pandas DataFrame
  - Polars DataFrame
  - Apache Arrow Table
  - Parquet files
  - CSV files

## Installation

```bash
# Basic installation
pip install nanofts

# With pandas support
pip install nanofts[pandas]

# With polars support
pip install nanofts[polars]

# With Apache Arrow/Parquet support
pip install nanofts[pyarrow]
```

## Usage

### Basic Example
```python
from nanofts import FullTextSearch

# Create a new search instance with disk storage
fts = FullTextSearch(index_dir="./index")

# Add single document
fts.add_document(1, {
    "title": "Hello World",
    "content": "Python full-text search engine"
})

# Add multiple documents at once
docs = [
    {"title": "全文搜索", "content": "支持中文搜索功能"},
    {"title": "Mixed Text", "content": "Support both English and 中文"}
]
fts.add_document([2, 3], docs)

# Don't forget to flush after adding documents
fts.flush()

# Search for documents
results = fts.search("python search")  # Case-insensitive search
print(results)  # Returns list of matching document IDs

# Chinese text search
results = fts.search("全文搜索")
print(results)
```

### Data Import from Different Sources
```python
# Import from pandas DataFrame
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3],
    'title': ['Hello World', '全文搜索', 'Test Document'],
    'content': ['This is a test', '支持多语言', 'Another test']
})

fts = FullTextSearch(index_dir="./index")
fts.from_pandas(df, id_column='id')

# Import from Polars DataFrame
import polars as pl
df = pl.DataFrame(...)
fts.from_polars(df, id_column='id')

# Import from Arrow Table
import pyarrow as pa
table = pa.Table.from_pandas(df)
fts.from_arrow(table, id_column='id')

# Import from Parquet file
fts.from_parquet("documents.parquet", id_column='id')

# Import from CSV file
fts.from_csv("documents.csv", id_column='id')
```

### Advanced Configuration
```python
fts = FullTextSearch(
    index_dir="./index",           # Index storage directory
    max_chinese_length=4,          # Maximum length for Chinese substrings
    num_workers=4,                 # Number of parallel workers
    shard_size=100_000,           # Documents per shard
    min_term_length=2,            # Minimum term length to index
    auto_save=True,               # Auto-save to disk
    batch_size=1000,              # Batch processing size
    buffer_size=10000,            # Memory buffer size
    drop_if_exists=False          # Whether to drop existing index
)
```

## Implementation Details

- Uses `pyroaring` for efficient bitmap operations
- Implements sharding for large-scale indexes
- LRU caching for frequently accessed terms
- Parallel processing for batch indexing
- Incremental updates with memory buffer
- Disk-based storage with msgpack serialization
- Support for both exact and phrase matching
- Efficient Chinese text substring indexing

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
# NanoFTS

A lightweight full-text search engine implementation in Python, featuring efficient indexing and searching capabilities for both English and Chinese text.

## Features

- Lightweight and efficient full-text search implementation
- Pure Python with minimal dependencies (only requires `pyroaring` and `msgpack` and `xxhash`)
- Support for both English and Chinese text
- Memory-efficient disk-based index storage with sharding
- Incremental indexing and real-time updates
- Case-insensitive search
- Phrase matching support
- **🔍 Fuzzy Search Support**: Intelligent fuzzy matching with configurable similarity thresholds
- **📝 Document Management**: Full CRUD operations (Create, Read, Update, Delete)
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

# Install all optional dependencies
pip install nanofts[all]

# Development dependencies (for contributors)
pip install nanofts[dev]
```

## Usage

### Quick Start
```python
from nanofts import FullTextSearch

# Initialize with fuzzy search support
fts = FullTextSearch(index_dir="./index", fuzzy_threshold=0.6)

# Add documents
fts.add_document(1, {"title": "Python教程", "content": "学习Python编程"})
fts.add_document(2, {"title": "数据分析", "content": "使用pandas进行数据处理"})
fts.flush()

# Search with typo handling
results = fts.fuzzy_search("Pytho教成")  # Finds "Python教程" despite typos
print(f"Found {len(results)} documents")

# Update and delete documents
fts.update_document(1, {"title": "高级Python教程"})
fts.remove_document(2)
```

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

### Fuzzy Search
```python
# Enable fuzzy search for typos and similar words
fts = FullTextSearch(
    index_dir="./index",
    fuzzy_threshold=0.6,      # Similarity threshold (0.0-1.0)
    fuzzy_max_distance=2      # Maximum edit distance
)

# Add some documents
fts.add_document(1, {"title": "苹果手机", "content": "最新的iPhone产品"})
fts.add_document(2, {"title": "编程教程", "content": "Python开发指南"})
fts.flush()

# Exact search
exact_results = fts.search("苹果", enable_fuzzy=False)
print(f"Exact search: {len(exact_results)} results")

# Fuzzy search for typos (苹檎 instead of 苹果)
fuzzy_results = fts.search("苹檎", enable_fuzzy=True, min_results=1)
print(f"Fuzzy search: {len(fuzzy_results)} results")

# Convenient fuzzy search method
results = fts.fuzzy_search("编成")  # 编成 -> 编程
print(f"Fuzzy search results: {results}")

# Configure fuzzy search parameters
fts.set_fuzzy_config(fuzzy_threshold=0.8, fuzzy_max_distance=1)
config = fts.get_fuzzy_config()
print(f"Current config: {config}")
```

### Document Management (CRUD Operations)
```python
# Create: Add documents (already shown above)
fts.add_document(1, {"title": "Document 1", "content": "Content 1"})

# Read: Search documents (already shown above)
results = fts.search("Document")

# Update: Modify existing documents
fts.update_document(1, {"title": "Updated Document", "content": "Updated Content"})

# Batch update multiple documents
fts.update_document([1, 2], [
    {"title": "New Title 1", "content": "New Content 1"},
    {"title": "New Title 2", "content": "New Content 2"}
])

# Delete: Remove documents
fts.remove_document(1)  # Remove single document

# Batch delete multiple documents
fts.remove_document([2, 3, 4])  # Remove multiple documents
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
    drop_if_exists=False,         # Whether to drop existing index
    fuzzy_threshold=0.4,          # Fuzzy search similarity threshold (0.0-1.0)
    fuzzy_max_distance=2          # Maximum edit distance for fuzzy search
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
- **Fuzzy Search Features**:
  - Zero I/O overhead: completely in-memory fuzzy matching
  - Intelligent activation: automatically enabled when exact results are insufficient
  - Configurable similarity thresholds and edit distance
  - Support for both Chinese and English fuzzy matching
  - Built-in caching for repeated fuzzy queries
- **Document Management**:
  - Full CRUD operations with atomic updates
  - Batch operations for high-performance updates
  - Incremental saving for modified documents

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
# NanoFTS

A high-performance full-text search engine with Rust core, featuring efficient indexing and searching capabilities for both English and Chinese text.

[![Crates.io](https://img.shields.io/crates/v/nanofts.svg)](https://crates.io/crates/nanofts)
[![Documentation](https://docs.rs/nanofts/badge.svg)](https://docs.rs/nanofts)
[![PyPI](https://img.shields.io/pypi/v/nanofts.svg)](https://pypi.org/project/nanofts/)

## Features

- **High Performance**: Rust-powered core with sub-millisecond search latency
- **LSM-Tree Architecture**: Scalable to billions of documents
- **Incremental Updates**: Real-time document add/update/delete
- **Fuzzy Search**: Intelligent fuzzy matching with configurable thresholds
- **Full CRUD**: Complete document management operations
- **Result Handle**: Zero-copy result with set operations (AND/OR/NOT)
- **NumPy Support**: Direct numpy array output (Python only)
- **Multilingual**: Support for both English and Chinese text
- **Persistence**: Disk-based storage with WAL recovery
- **LRU Cache**: Built-in caching for frequently accessed terms
- **Data Import**: Import from pandas, polars, arrow, parquet, CSV, JSON (Python only)
- **Dual API**: Available as both Rust crate and Python package

## Installation

### Rust (Cargo)

Add to your `Cargo.toml`:

```toml
[dependencies]
nanofts = "0.3"
```

### Python (pip)

```bash
pip install nanofts
```

## Rust Quick Start

```rust
use nanofts::{UnifiedEngine, EngineConfig, EngineResult};
use std::collections::HashMap;

fn main() -> EngineResult<()> {
    // Create an in-memory search engine
    let engine = UnifiedEngine::new(EngineConfig::memory_only())?;

    // Add a document
    let mut fields = HashMap::new();
    fields.insert("title".to_string(), "Hello World".to_string());
    fields.insert("content".to_string(), "This is a test document".to_string());
    engine.add_document(1, fields)?;

    // Search
    let result = engine.search("hello")?;
    println!("Found {} documents", result.total_hits());

    // Get document IDs
    for doc_id in result.iter() {
        println!("Document ID: {}", doc_id);
    }

    Ok(())
}
```

### Persistent Storage (Rust)

```rust
use nanofts::{UnifiedEngine, EngineConfig};

// Create a persistent search engine
let config = EngineConfig::persistent("my_index.nfts")
    .with_lazy_load(true)       // Enable lazy loading for large indexes
    .with_cache_size(10000);    // LRU cache size

let engine = UnifiedEngine::new(config)?;

// ... add documents and search ...

// Flush to disk
engine.flush()?;
```

### Boolean Search Operations (Rust)

```rust
// AND search
let result = engine.search_and(vec!["rust".to_string(), "programming".to_string()])?;

// OR search  
let result = engine.search_or(vec!["rust".to_string(), "python".to_string()])?;

// Result set operations
let result1 = engine.search("rust")?;
let result2 = engine.search("python")?;
let intersection = result1.intersect(&result2);
let union = result1.union(&result2);
let difference = result1.difference(&result2);
```

## Python Quick Start

```python
from nanofts import create_engine

# Create a search engine
engine = create_engine(
    index_file="./index.nfts",
    track_doc_terms=True,  # Enable update/delete operations
)

# Add documents (field values must be strings)
engine.add_document(1, {"title": "Python教程", "content": "学习Python编程"})
engine.add_document(2, {"title": "数据分析", "content": "使用pandas进行数据处理"})
engine.flush()

# Search - returns ResultHandle object
result = engine.search("Python")
print(f"Found {result.total_hits} documents")
print(f"Document IDs: {result.to_list()}")

# Update document
engine.update_document(1, {"title": "高级Python教程", "content": "深入学习Python"})

# Delete document
engine.remove_document(2)

# Compact to persist deletions
engine.compact()
```

## API Reference

### Creating Engine

```python
from nanofts import create_engine

engine = create_engine(
    index_file="./index.nfts",     # Index file path (empty string for memory-only)
    max_chinese_length=4,          # Max Chinese n-gram length
    min_term_length=2,             # Minimum term length to index
    fuzzy_threshold=0.7,           # Fuzzy search similarity threshold (0.0-1.0)
    fuzzy_max_distance=2,          # Maximum edit distance for fuzzy search
    track_doc_terms=False,         # Enable for update/delete support
    drop_if_exists=False,          # Drop existing index on creation
    lazy_load=False,               # Lazy load mode (memory efficient)
    cache_size=10000,              # LRU cache size for lazy load mode
)
```

### Document Operations

```python
# Add single document
engine.add_document(doc_id=1, fields={"title": "Hello", "content": "World"})

# Add multiple documents
docs = [
    (1, {"title": "Doc 1", "content": "Content 1"}),
    (2, {"title": "Doc 2", "content": "Content 2"}),
]
engine.add_documents(docs)

# Update document (requires track_doc_terms=True)
engine.update_document(1, {"title": "Updated", "content": "New content"})

# Delete single document
engine.remove_document(1)

# Delete multiple documents
engine.remove_documents([1, 2, 3])

# Flush buffer to disk
engine.flush()

# Compact index (applies deletions permanently)
engine.compact()
```

### Search Operations

```python
# Basic search - returns ResultHandle
result = engine.search("python programming")

# Get results
doc_ids = result.to_list()           # List[int]
doc_ids = result.to_numpy()          # numpy array
top_10 = result.top(10)              # Top N results
page_2 = result.page(page=2, size=10)  # Pagination

# Result properties
print(result.total_hits)             # Total match count
print(result.is_empty)               # Check if empty
print(1 in result)                   # Check if doc_id in results

# Fuzzy search (for typo tolerance)
result = engine.fuzzy_search("pythn", min_results=5)
print(result.fuzzy_used)             # True if fuzzy matching was applied

# Batch search
results = engine.search_batch(["python", "rust", "java"])

# AND search (intersection)
result = engine.search_and(["python", "tutorial"])

# OR search (union)
result = engine.search_or(["python", "rust"])

# Filter by document IDs
result = engine.filter_by_ids([1, 2, 3, 4, 5])

# Exclude specific IDs
result = engine.exclude_ids([1, 2])
```

### Result Set Operations

```python
# Search for different terms
python_docs = engine.search("python")
rust_docs = engine.search("rust")

# Intersection (AND)
both = python_docs.intersect(rust_docs)

# Union (OR)
either = python_docs.union(rust_docs)

# Difference (NOT)
python_only = python_docs.difference(rust_docs)

# Chained operations
result = engine.search("python").intersect(
    engine.search("tutorial")
).difference(
    engine.search("beginner")
)
```

### Statistics

```python
stats = engine.stats()
print(stats)
# {
#     'term_count': 1234,
#     'search_count': 100,
#     'fuzzy_search_count': 10,
#     'total_search_ns': 1234567,
#     ...
# }
```

### Data Import

NanoFTS supports importing data from various sources:

```python
from nanofts import create_engine

engine = create_engine("./index.nfts")

# Import from pandas DataFrame
import pandas as pd
df = pd.DataFrame({
    'id': [1, 2, 3],
    'title': ['Hello World', '全文搜索', 'Test Document'],
    'content': ['This is a test', '支持多语言', 'Another test']
})
engine.from_pandas(df, id_column='id')

# Import from Polars DataFrame
import polars as pl
df = pl.DataFrame({
    'id': [1, 2, 3],
    'title': ['Doc 1', 'Doc 2', 'Doc 3']
})
engine.from_polars(df, id_column='id')

# Import from PyArrow Table
import pyarrow as pa
table = pa.Table.from_pydict({
    'id': [1, 2, 3],
    'title': ['Arrow 1', 'Arrow 2', 'Arrow 3']
})
engine.from_arrow(table, id_column='id')

# Import from Parquet file
engine.from_parquet("documents.parquet", id_column='id')

# Import from CSV file
engine.from_csv("documents.csv", id_column='id')

# Import from JSON file
engine.from_json("documents.json", id_column='id')

# Import from JSON Lines file
engine.from_json("documents.jsonl", id_column='id', lines=True)

# Import from Python dict list
data = [
    {'id': 1, 'title': 'Hello', 'content': 'World'},
    {'id': 2, 'title': 'Test', 'content': 'Document'}
]
engine.from_dict(data, id_column='id')
```

#### Specifying Text Columns

By default, all columns except the ID column are indexed. You can specify which columns to index:

```python
# Only index 'title' and 'content' columns, ignore 'metadata'
engine.from_pandas(df, id_column='id', text_columns=['title', 'content'])

# Same for other import methods
engine.from_csv("data.csv", id_column='id', text_columns=['title', 'content'])
```

#### CSV and JSON Options

You can pass additional options to the underlying pandas readers:

```python
# CSV with custom delimiter
engine.from_csv("data.csv", id_column='id', sep=';', encoding='utf-8')

# JSON Lines format
engine.from_json("data.jsonl", id_column='id', lines=True)
```

## Chinese Text Support

NanoFTS handles Chinese text using n-gram tokenization:

```python
engine = create_engine(
    index_file="./chinese_index.nfts",
    max_chinese_length=4,  # Generate 2,3,4-gram for Chinese
)

engine.add_document(1, {"content": "全文搜索引擎"})
engine.flush()

# Search Chinese text
result = engine.search("搜索")
print(result.to_list())  # [1]
```

## Persistence and Recovery

```python
# Create persistent index
engine = create_engine(index_file="./data.nfts")
engine.add_document(1, {"title": "Test"})
engine.flush()

# Close and reopen
del engine
engine = create_engine(index_file="./data.nfts")

# Data is automatically recovered
result = engine.search("Test")
print(result.to_list())  # [1]

# Important: Use compact() to persist deletions
engine.remove_document(1)
engine.compact()  # Deletions are now permanent
```

## Memory-Only Mode

```python
# Create in-memory engine (no persistence)
engine = create_engine(index_file="")

engine.add_document(1, {"content": "temporary data"})
# No flush needed for in-memory mode

result = engine.search("temporary")
```

## Best Practices

### For Production Use

1. **Always call `compact()` after bulk deletions** - Deletions are only persisted after compaction
2. **Use `track_doc_terms=True`** if you need update/delete operations
3. **Call `flush()` periodically** to persist new documents
4. **Use `lazy_load=True`** for large indexes that don't fit in memory

### Performance Tips

```python
# Batch operations are faster
docs = [(i, {"content": f"doc {i}"}) for i in range(10000)]
engine.add_documents(docs)  # Much faster than individual add_document calls
engine.flush()

# Use batch search for multiple queries
results = engine.search_batch(["query1", "query2", "query3"])

# Use result set operations instead of multiple searches
# Good:
result = engine.search_and(["python", "tutorial"])
# Instead of:
# result = engine.search("python").intersect(engine.search("tutorial"))
```

## Migration from Old API

If you're upgrading from the old `FullTextSearch` API:

```python
# Old API (deprecated)
# from nanofts import FullTextSearch
# fts = FullTextSearch(index_dir="./index")
# fts.add_document(1, {"title": "Test"})
# results = fts.search("Test")  # Returns List[int]

# New API
from nanofts import create_engine
engine = create_engine(index_file="./index.nfts")
engine.add_document(1, {"title": "Test"})
result = engine.search("Test")
results = result.to_list()  # Returns List[int]
```

Key differences:
- `FullTextSearch` → `create_engine()` function
- `index_dir` → `index_file` (file path, not directory)
- Search returns `ResultHandle` instead of `List[int]`
- Call `.to_list()` to get document IDs
- Use `compact()` to persist deletions

## Cargo Features

| Feature | Description | Default |
|---------|-------------|---------|
| `python` | Enable Python bindings via PyO3 | No |
| `simd` | Enable SIMD acceleration (requires nightly) | No |
| `mimalloc` | Use mimalloc allocator | Yes |

### Building with Python Support

```bash
# Build with Python bindings
cargo build --features python

# Build Python wheel with maturin
maturin build --release --features python
```

## Publishing to crates.io

```bash
# Login to crates.io (first time only)
cargo login

# Publish the crate
cd nanofts
cargo publish
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# NanoFTS Complete Tutorial

This tutorial will guide you through all features of NanoFTS.

## Table of Contents

- [Quick Start](#quick-start)
- [Creating Search Engine](#creating-search-engine)
- [Document Operations](#document-operations)
- [Search Operations](#search-operations)
- [Result Handling](#result-handling)
- [Fuzzy Search](#fuzzy-search)
- [Data Import](#data-import)
- [Persistence and Recovery](#persistence-and-recovery)
- [Performance Optimization](#performance-optimization)
- [Chinese Search](#chinese-search)
- [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
pip install nanofts
```

### First Example

```python
from nanofts import create_engine

# 1. Create engine
engine = create_engine("./my_index.nfts")

# 2. Add documents
engine.add_document(1, {"title": "Hello World", "content": "Welcome to NanoFTS"})
engine.add_document(2, {"title": "Python Guide", "content": "Learn Python programming"})
engine.flush()  # Persist to disk

# 3. Search
result = engine.search("Python")

# 4. Get results
print(f"Total hits: {result.total_hits}")
print(f"Document IDs: {result.to_list()}")
```

---

## Creating Search Engine

Use `create_engine()` function to create a search engine instance.

### Basic Usage

```python
from nanofts import create_engine

# Mode 1: Memory-only (not persisted)
engine = create_engine("")

# Mode 2: Persistent (recommended)
engine = create_engine("./index.nfts")

# Mode 3: Overwrite existing index
engine = create_engine("./index.nfts", drop_if_exists=True)
```

### All Configuration Options

```python
engine = create_engine(
    index_file="./index.nfts",   # Index file path, "" for memory-only
    max_chinese_length=4,         # Max Chinese n-gram length
    min_term_length=2,            # Min term length
    fuzzy_threshold=0.7,          # Fuzzy match threshold (0.0-1.0)
    fuzzy_max_distance=2,         # Max edit distance
    track_doc_terms=False,        # Track document terms (for efficient deletion)
    drop_if_exists=False,         # Delete existing index
    lazy_load=True,               # Lazy load mode (low memory usage)
    cache_size=10000              # LRU cache size
)
```

### Mode Selection Guide

| Mode | Use Case | Memory Usage | Startup Speed |
|------|----------|--------------|---------------|
| `lazy_load=True` | Large indexes, billions of documents | Low | Fast |
| `lazy_load=False` | Small indexes, frequent queries | High | Slower |
| Memory-only `""` | Testing, temporary data | Medium | Fastest |

---

## Document Operations

### Add Single Document

```python
# Basic usage
engine.add_document(1, {"title": "Document Title", "content": "Document content"})

# Multiple fields
engine.add_document(2, {
    "title": "Python Tutorial",
    "summary": "Learn Python basics",
    "content": "Python is a programming language...",
    "tags": "python programming tutorial"
})
```

### Batch Add Documents

```python
docs = [
    (1, {"title": "Doc 1", "content": "Content 1"}),
    (2, {"title": "Doc 2", "content": "Content 2"}),
    (3, {"title": "Doc 3", "content": "Content 3"}),
]
engine.add_documents(docs)
```

### Update Document

```python
# Update document 1's content
engine.update_document(1, {
    "title": "Updated Title",
    "content": "Updated content"
})
```

### Delete Document

```python
# Delete single document
engine.remove_document(1)

# Batch delete
engine.remove_documents([1, 2, 3])
```

---

## Search Operations

### Basic Search

```python
# Single word search
result = engine.search("python")

# Phrase search (AND logic)
result = engine.search("python tutorial")
```

### Batch Search

```python
queries = ["python", "java", "rust"]
results = engine.search_batch(queries)

for query, result in zip(queries, results):
    print(f"{query}: {result.total_hits} hits")
```

### AND Search

```python
# All keywords must appear
result = engine.search_and(["python", "tutorial"])
```

### OR Search

```python
# Any keyword can appear
result = engine.search_or(["python", "java", "rust"])
```

### Result Filtering

```python
# Filter by allowed IDs
result = engine.search("python")
filtered = engine.filter_by_ids(result, allowed_ids=[1, 2, 3, 4, 5])

# Exclude specific IDs
filtered = engine.exclude_ids(result, excluded_ids=[10, 20])
```

---

## Result Handling

### ResultHandle Class

Search returns `ResultHandle` object providing various access methods:

```python
result = engine.search("python")

# Basic properties
print(f"Total hits: {result.total_hits}")
print(f"Search time: {result.elapsed_ms():.3f}ms")
print(f"Fuzzy search used: {result.fuzzy_used}")

# Get results
all_ids = result.to_list()              # All results
top_10 = result.top(10)                  # Top 10
page_2 = result.page(offset=10, limit=10)  # Pagination

# NumPy output
import numpy as np
ids_array = result.to_numpy()            # Returns numpy.ndarray

# Check if document matches
if result.contains(doc_id=42):
    print("Document 42 matches!")
```

### Result Set Operations

```python
result1 = engine.search("python")
result2 = engine.search("tutorial")

# Intersection (AND)
combined = result1.intersect(result2)

# Union (OR)
combined = result1.union(result2)

# Difference (NOT)
combined = result1.difference(result2)  # In result1 but not result2
```

---

## Fuzzy Search

Fuzzy search automatically finds similar terms when exact match fails.

### Basic Usage

```python
# Enable fuzzy search, minimum 5 results
result = engine.fuzzy_search("pythn", min_results=5)  # Typo
print(f"Found {result.total_hits} results (fuzzy={result.fuzzy_used})")
```

### Configure Fuzzy Parameters

```python
# Set fuzzy configuration
engine.set_fuzzy_config(
    threshold=0.8,        # Similarity threshold (0.0-1.0)
    max_distance=2,       # Max edit distance
    max_candidates=30     # Max candidate terms
)

# Get current configuration
config = engine.get_fuzzy_config()
print(config)  # {'threshold': 0.8, 'max_distance': 2, 'max_candidates': 30}
```

### Fuzzy Search Tips

1. **threshold**: Higher value = stricter matching, more accurate but fewer results
2. **max_distance**: Edit distance limit, 2 is good for most cases
3. **max_candidates**: Trade-off between speed and recall

---

## Data Import

NanoFTS supports importing data from various formats.

### From pandas DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3],
    'title': ['Hello', 'World', 'Test'],
    'content': ['Content 1', 'Content 2', 'Content 3']
})

count = engine.from_pandas(df, id_column='id')
print(f"Imported {count} documents")
```

### From Polars DataFrame

```python
import polars as pl

df = pl.DataFrame({
    'id': [1, 2, 3],
    'title': ['Hello', 'World', 'Test']
})

count = engine.from_polars(df, id_column='id')
```

### From PyArrow Table

```python
import pyarrow as pa

table = pa.Table.from_pydict({
    'id': [1, 2, 3],
    'title': ['Hello', 'World', 'Test']
})

count = engine.from_arrow(table, id_column='id')
```

### From Parquet File

```python
count = engine.from_parquet("documents.parquet", id_column='id')
```

### From CSV File

```python
# Basic usage
count = engine.from_csv("documents.csv", id_column='id')

# With custom options
count = engine.from_csv(
    "documents.csv",
    id_column='id',
    encoding='utf-8',
    sep=','
)
```

### From JSON File

```python
# Standard JSON
count = engine.from_json("documents.json", id_column='id')

# JSON Lines format
count = engine.from_json("documents.jsonl", id_column='id', lines=True)
```

### From Dict List

```python
data = [
    {'id': 1, 'title': 'Hello', 'content': 'World'},
    {'id': 2, 'title': 'Test', 'content': 'Document'}
]

count = engine.from_dict(data, id_column='id')
```

---

## Persistence and Recovery

### Manual Persistence

```python
# Flush buffer to disk
engine.flush()

# Same as flush()
engine.save()
```

### Automatic Recovery

NanoFTS uses WAL (Write-Ahead Log) for crash recovery:

```python
# Data is automatically recovered on open
engine = create_engine("./index.nfts")
# Data is automatically recovered if previous session crashed
```

### Compact Operation

```python
# Apply deletions to disk and optimize storage
engine.compact()
```

**Note**: `compact()` should be called periodically:
- After batch deletions
- To reclaim disk space
- To improve query performance

---

## Performance Optimization

### Preload Index

```python
# Load all data into memory (for non-lazy mode)
term_count = engine.preload()
print(f"Preloaded {term_count} terms")
```

### Cache Warmup (Lazy Load Mode)

```python
# Preload frequently used terms
hot_terms = ["python", "java", "rust", "programming"]
loaded = engine.warmup_terms(hot_terms)
print(f"Warmed up {loaded} terms")
```

### Cache Management

```python
# Clear search result cache
engine.clear_cache()

# Clear document term tracking
engine.clear_doc_terms()

# Clear LRU cache (lazy load mode)
engine.clear_lru_cache()
```

### Statistics Monitoring

```python
stats = engine.stats()
print(f"Search count: {stats['search_count']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Avg search time: {stats['avg_search_us']:.2f}μs")
```

---

## Chinese Search

NanoFTS has native Chinese support with n-gram indexing.

### Chinese Text Indexing

```python
engine.add_document(1, {"content": "Full-text search engine"})
engine.add_document(2, {"content": "High-performance search"})

# Chinese search
result = engine.search("search")
print(f"Found: {result.total_hits}")  # 2

# Search for Chinese substrings
result = engine.search("high-performance")
print(f"Found: {result.total_hits}")  # 1
```

### Chinese Configuration

```python
engine = create_engine(
    "./index.nfts",
    max_chinese_length=4,  # Max n-gram length (2-4 chars recommended)
    min_term_length=2      # Min term length
)
```

---

## Best Practices

### 1. Choose Appropriate Mode

```python
# Large data (millions of documents): Use lazy load
engine = create_engine("./large_index.nfts", lazy_load=True)

# Small data, high-frequency queries: Disable lazy load
engine = create_engine("./small_index.nfts", lazy_load=False)

# Testing: Use memory-only
engine = create_engine("")
```

### 2. Batch Operations

```python
# Good: Batch add
docs = [(i, {"content": f"Document {i}"}) for i in range(1000)]
engine.add_documents(docs)
engine.flush()

# Bad: Add one by one with flush
for i in range(1000):
    engine.add_document(i, {"content": f"Document {i}"})
    engine.flush()  # Too many disk I/O!
```

### 3. Use Result Handle Operations

```python
# Good: Use result handle operations
result1 = engine.search("python")
result2 = engine.search("tutorial")
combined = result1.intersect(result2)

# Alternative: Use AND search
result = engine.search_and(["python", "tutorial"])
```

### 4. Regular Maintenance

```python
# Periodic compact to reclaim space
if engine.deleted_count() > 10000:
    engine.compact()

# Monitor performance
stats = engine.stats()
if stats['cache_hit_rate'] < 0.5:
    # Consider enabling track_doc_terms or adjusting cache size
    pass
```

### 5. Error Handling

```python
from nanofts import create_engine

try:
    engine = create_engine("./index.nfts")
    result = engine.search("query")
except Exception as e:
    print(f"Search error: {e}")
```

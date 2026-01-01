# NanoFTS API Reference

This document provides complete API reference for NanoFTS, **100% coverage of all public APIs**.

## Table of Contents

- [Module Exports](#module-exports)
- [create_engine Function](#create_engine-function)
- [UnifiedEngine Class](#unifiedengine-class)
- [ResultHandle Class](#resulthandle-class)
- [FuzzyConfig Class](#fuzzyconfig-class)

---

## Module Exports

```python
from nanofts import (
    # Main API
    create_engine,      # Factory function to create search engine
    
    # Classes
    SearchEngine,       # Alias for UnifiedEngine
    SearchResult,       # Alias for ResultHandle
    UnifiedEngine,      # Search engine class
    ResultHandle,       # Search result handle class
    FuzzyConfig,        # Fuzzy search configuration class
    
    # Version
    __version__,        # Current version "0.3.2"
)
```

---

## create_engine Function

Factory function to create search engine instance.

### Signature

```python
def create_engine(
    index_file: str = "",
    max_chinese_length: int = 4,
    min_term_length: int = 2,
    fuzzy_threshold: float = 0.7,
    fuzzy_max_distance: int = 2,
    track_doc_terms: bool = False,
    drop_if_exists: bool = False,
    lazy_load: bool = True,
    cache_size: int = 10000
) -> UnifiedEngine
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index_file` | `str` | `""` | Index file path, empty string for memory-only mode |
| `max_chinese_length` | `int` | `4` | Maximum Chinese n-gram length |
| `min_term_length` | `int` | `2` | Minimum term length |
| `fuzzy_threshold` | `float` | `0.7` | Fuzzy search similarity threshold (0.0-1.0) |
| `fuzzy_max_distance` | `int` | `2` | Fuzzy search maximum edit distance |
| `track_doc_terms` | `bool` | `False` | Track document terms for efficient deletion |
| `drop_if_exists` | `bool` | `False` | Delete existing index file |
| `lazy_load` | `bool` | `True` | Enable lazy load mode |
| `cache_size` | `int` | `10000` | LRU cache size in lazy load mode |

### Return Value

Returns `UnifiedEngine` instance.

### Examples

```python
# Default mode (lazy load, recommended)
engine = create_engine("./index.nfts")

# Memory-only mode
engine = create_engine("")

# Full load mode (for small indexes)
engine = create_engine("./index.nfts", lazy_load=False)

# Overwrite existing index
engine = create_engine("./index.nfts", drop_if_exists=True)
```

---

## UnifiedEngine Class

Main search engine class, provides all search functionality.

### Constructor

```python
UnifiedEngine(
    index_file: str = "",
    max_chinese_length: int = 4,
    min_term_length: int = 2,
    fuzzy_threshold: float = 0.7,
    fuzzy_max_distance: int = 2,
    track_doc_terms: bool = False,
    drop_if_exists: bool = False,
    lazy_load: bool = False,
    cache_size: int = 10000
)
```

### Document Operations

#### add_document

```python
def add_document(self, doc_id: int, fields: Dict[str, str]) -> None
```

Add single document.

**Parameters:**
- `doc_id`: Document ID (unsigned 32-bit integer)
- `fields`: Dictionary of field name -> field value

**Example:**
```python
engine.add_document(1, {"title": "Hello", "content": "World"})
```

#### add_documents

```python
def add_documents(self, docs: List[Tuple[int, Dict[str, str]]]) -> int
```

Batch add documents.

**Parameters:**
- `docs`: List of (doc_id, fields) tuples

**Returns:** Number of documents added

**Example:**
```python
docs = [
    (1, {"title": "Doc 1"}),
    (2, {"title": "Doc 2"}),
]
count = engine.add_documents(docs)
```

#### update_document

```python
def update_document(self, doc_id: int, fields: Dict[str, str]) -> None
```

Update existing document.

**Parameters:**
- `doc_id`: Document ID to update
- `fields`: New field values

**Example:**
```python
engine.update_document(1, {"title": "Updated Title"})
```

#### remove_document

```python
def remove_document(self, doc_id: int) -> None
```

Remove document.

**Parameters:**
- `doc_id`: Document ID to remove

**Example:**
```python
engine.remove_document(1)
```

#### remove_documents

```python
def remove_documents(self, doc_ids: List[int]) -> None
```

Batch remove documents.

**Parameters:**
- `doc_ids`: List of document IDs to remove

**Example:**
```python
engine.remove_documents([1, 2, 3])
```

### Search Operations

#### search

```python
def search(self, query: str) -> ResultHandle
```

Basic search.

**Parameters:**
- `query`: Search query string

**Returns:** `ResultHandle` object

**Example:**
```python
result = engine.search("python tutorial")
```

#### fuzzy_search

```python
def fuzzy_search(self, query: str, min_results: int = 5) -> ResultHandle
```

Fuzzy search, returns approximate matches when exact match fails.

**Parameters:**
- `query`: Search query string
- `min_results`: Minimum results before triggering fuzzy search

**Returns:** `ResultHandle` object

**Example:**
```python
result = engine.fuzzy_search("pythn", min_results=10)  # Typo
```

#### search_batch

```python
def search_batch(self, queries: List[str]) -> List[ResultHandle]
```

Batch search (parallel execution).

**Parameters:**
- `queries`: List of query strings

**Returns:** List of `ResultHandle` objects

**Example:**
```python
results = engine.search_batch(["python", "java", "rust"])
```

#### search_and

```python
def search_and(self, queries: List[str]) -> ResultHandle
```

AND search (intersection).

**Parameters:**
- `queries`: List of query strings

**Returns:** `ResultHandle` object (documents matching all queries)

**Example:**
```python
result = engine.search_and(["python", "tutorial"])
```

#### search_or

```python
def search_or(self, queries: List[str]) -> ResultHandle
```

OR search (union).

**Parameters:**
- `queries`: List of query strings

**Returns:** `ResultHandle` object (documents matching any query)

**Example:**
```python
result = engine.search_or(["python", "java"])
```

#### filter_by_ids

```python
def filter_by_ids(self, result: ResultHandle, allowed_ids: List[int]) -> ResultHandle
```

Filter results to allowed IDs only.

**Parameters:**
- `result`: Search result to filter
- `allowed_ids`: Allowed document IDs

**Returns:** Filtered `ResultHandle`

#### exclude_ids

```python
def exclude_ids(self, result: ResultHandle, excluded_ids: List[int]) -> ResultHandle
```

Exclude specific IDs from results.

**Parameters:**
- `result`: Search result to filter
- `excluded_ids`: Document IDs to exclude

**Returns:** Filtered `ResultHandle`

### Configuration Operations

#### set_fuzzy_config

```python
def set_fuzzy_config(
    self,
    threshold: float = 0.7,
    max_distance: int = 2,
    max_candidates: int = 20
) -> None
```

Set fuzzy search configuration.

**Parameters:**
- `threshold`: Similarity threshold (0.0-1.0)
- `max_distance`: Maximum edit distance
- `max_candidates`: Maximum candidate terms

#### get_fuzzy_config

```python
def get_fuzzy_config(self) -> Dict[str, float]
```

Get current fuzzy search configuration.

**Returns:** Dictionary with `threshold`, `max_distance`, `max_candidates`

### Persistence Operations

#### flush

```python
def flush(self) -> int
```

Flush buffer to disk.

**Returns:** Number of terms flushed

**Example:**
```python
count = engine.flush()
print(f"Flushed {count} terms")
```

#### save

```python
def save(self) -> int
```

Save index (same as `flush()`).

**Returns:** Number of terms saved

#### load

```python
def load(self) -> None
```

Load index (no-op, data is loaded on creation).

#### preload

```python
def preload(self) -> int
```

Preload all index data into memory.

**Returns:** Number of terms preloaded

#### compact

```python
def compact(self) -> None
```

Compact index, apply pending deletions, and optimize storage.

**Example:**
```python
engine.compact()  # Should be called periodically
```

### Cache Operations

#### clear_cache

```python
def clear_cache(self) -> None
```

Clear search result cache.

#### clear_doc_terms

```python
def clear_doc_terms(self) -> None
```

Clear document term tracking data.

#### clear_lru_cache

```python
def clear_lru_cache(self) -> None
```

Clear LRU cache (lazy load mode).

#### warmup_terms

```python
def warmup_terms(self, terms: List[str]) -> int
```

Warmup cache with specified terms (lazy load mode).

**Parameters:**
- `terms`: List of terms to preload

**Returns:** Number of terms loaded

### Statistics Operations

#### stats

```python
def stats(self) -> Dict[str, float]
```

Get engine statistics.

**Returns:** Dictionary containing:
- `search_count`: Total searches
- `fuzzy_search_count`: Fuzzy searches
- `cache_hits`: Cache hits
- `cache_hit_rate`: Cache hit rate
- `avg_search_us`: Average search time (microseconds)
- `result_cache_size`: Result cache size
- `buffer_size`: Buffer size
- `term_count`: Total terms
- `deleted_count`: Deleted document count
- `memory_only`: Whether memory-only mode
- `lazy_load`: Whether lazy load enabled
- `track_doc_terms`: Whether tracking document terms
- `wal_enabled`: Whether WAL enabled
- `wal_size`: WAL file size
- `wal_pending_batches`: WAL pending batches
- `lru_cache_hits`: LRU cache hits (lazy mode)
- `lru_cache_misses`: LRU cache misses (lazy mode)
- `lru_cache_size`: LRU cache size (lazy mode)
- `lru_cache_hit_rate`: LRU cache hit rate (lazy mode)

#### term_count

```python
def term_count(self) -> int
```

Get total term count.

#### buffer_size

```python
def buffer_size(self) -> int
```

Get buffer size.

#### doc_terms_size

```python
def doc_terms_size(self) -> int
```

Get document terms tracking size.

#### segment_count

```python
def segment_count(self) -> int
```

Get segment/page count.

#### memtable_size

```python
def memtable_size(self) -> int
```

Get memtable size.

#### deleted_count

```python
def deleted_count(self) -> int
```

Get deleted document count.

#### is_memory_only

```python
def is_memory_only(self) -> bool
```

Check if memory-only mode.

#### is_lazy_load

```python
def is_lazy_load(self) -> bool
```

Check if lazy load enabled.

### Data Import Methods

#### from_pandas

```python
def from_pandas(
    self,
    df,
    id_column: str = 'id',
    text_columns: Optional[List[str]] = None
) -> int
```

Import from pandas DataFrame.

#### from_polars

```python
def from_polars(
    self,
    df,
    id_column: str = 'id',
    text_columns: Optional[List[str]] = None
) -> int
```

Import from Polars DataFrame.

#### from_arrow

```python
def from_arrow(
    self,
    table,
    id_column: str = 'id',
    text_columns: Optional[List[str]] = None
) -> int
```

Import from PyArrow Table.

#### from_parquet

```python
def from_parquet(
    self,
    path: str,
    id_column: str = 'id',
    text_columns: Optional[List[str]] = None
) -> int
```

Import from Parquet file.

#### from_csv

```python
def from_csv(
    self,
    path: str,
    id_column: str = 'id',
    text_columns: Optional[List[str]] = None,
    **csv_options
) -> int
```

Import from CSV file.

#### from_json

```python
def from_json(
    self,
    path: str,
    id_column: str = 'id',
    text_columns: Optional[List[str]] = None,
    **json_options
) -> int
```

Import from JSON file.

#### from_dict

```python
def from_dict(
    self,
    data: List[Dict],
    id_column: str = 'id',
    text_columns: Optional[List[str]] = None
) -> int
```

Import from list of dictionaries.

---

## ResultHandle Class

Search result handle class, provides zero-copy access to results.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `total_hits` | `int` | Total matching documents |
| `elapsed_ns` | `int` | Search time (nanoseconds) |
| `fuzzy_used` | `bool` | Whether fuzzy search was used |

### Methods

#### elapsed_ms

```python
def elapsed_ms(self) -> float
```

Get search time in milliseconds.

#### elapsed_us

```python
def elapsed_us(self) -> float
```

Get search time in microseconds.

#### contains

```python
def contains(self, doc_id: int) -> bool
```

Check if result contains document.

#### is_empty

```python
def is_empty(self) -> bool
```

Check if result is empty.

#### top

```python
def top(self, n: int = 100) -> List[int]
```

Get top N document IDs.

#### to_list

```python
def to_list(self) -> List[int]
```

Get all document IDs as list.

#### to_numpy

```python
def to_numpy(self) -> numpy.ndarray
```

Get document IDs as NumPy array.

#### page

```python
def page(self, offset: int, limit: int) -> List[int]
```

Get paginated results.

#### intersect

```python
def intersect(self, other: ResultHandle) -> ResultHandle
```

Intersection (AND) with another result.

#### union

```python
def union(self, other: ResultHandle) -> ResultHandle
```

Union (OR) with another result.

#### difference

```python
def difference(self, other: ResultHandle) -> ResultHandle
```

Difference (NOT) with another result.

### Magic Methods

#### \_\_len\_\_

```python
def __len__(self) -> int
```

Get result count (for `len(result)`).

#### \_\_repr\_\_

```python
def __repr__(self) -> str
```

String representation.

---

## FuzzyConfig Class

Fuzzy search configuration class.

### Constructor

```python
FuzzyConfig(
    threshold: float = 0.7,
    max_distance: int = 2,
    max_candidates: int = 20
)
```

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `threshold` | `float` | `0.7` | Similarity threshold (0.0-1.0) |
| `max_distance` | `int` | `2` | Maximum edit distance |
| `max_candidates` | `int` | `20` | Maximum candidate terms |

### Example

```python
from nanofts import FuzzyConfig

config = FuzzyConfig(
    threshold=0.8,
    max_distance=3,
    max_candidates=50
)
```

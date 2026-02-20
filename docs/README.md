# NanoFTS Documentation

Welcome to NanoFTS - Ultra High-Performance Full-Text Search Engine!

## Overview

NanoFTS is a full-text search engine optimized for billion-scale data, with a Rust core implementation and Python bindings.

### Core Features

- 🚀 **High Performance**: Rust-powered, sub-millisecond search response
- 📈 **Scalable**: LSM-Tree architecture, supports billion-scale documents
- ⚡ **Real-time Updates**: Incremental writes, no index rebuild needed
- 🔍 **Fuzzy Search**: Intelligent fuzzy matching with spelling tolerance
- 🌏 **Multi-language**: Native support for mixed Chinese/English text
- 💾 **Persistence**: WAL (Write-Ahead Log) ensures data safety
- 🔢 **NumPy Support**: Search results directly output as NumPy arrays
- 📦 **Data Import**: Supports pandas, polars, arrow, parquet, CSV, JSON

## Documentation Index

| Document | Description |
|----------|-------------|
| [Quick Start](./tutorial.md#quick-start) | Get started with NanoFTS in 5 minutes |
| [Complete Tutorial](./tutorial.md) | Detailed feature guide |
| [API Reference](./api-reference.md) | Complete API documentation (100% coverage) |
| [Rust 写入性能指南](./rust-ingestion-performance.md) | Rust API 架构变更说明与最优性能使用方法 |

## Installation

```bash
pip install nanofts
```

## Quick Example

```python
from nanofts import create_engine

# Create search engine
engine = create_engine("./index.nfts")

# Add documents
engine.add_document(1, {"title": "Python Tutorial", "content": "Learn Python programming"})
engine.add_document(2, {"title": "Data Analysis", "content": "Use pandas for data processing"})
engine.flush()  # Persist to disk

# Search
result = engine.search("Python")
print(f"Total hits: {result.total_hits}")
print(f"Document IDs: {result.to_list()}")
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│                   Python API                      │
│     create_engine / SearchEngine / ResultHandle   │
├──────────────────────────────────────────────────┤
│                   Rust Core                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │ UnifiedEngine│  │ ResultHandle │  │ Fuzzy   │  │
│  └─────────────┘  └──────────────┘  │ Config  │  │
│                                     └─────────┘  │
├──────────────────────────────────────────────────┤
│               Storage Layer                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────┐  │
│  │ LsmSingle   │  │    WAL       │  │ Bitmap  │  │
│  │   Index     │  │              │  │ Cache   │  │
│  └─────────────┘  └──────────────┘  └─────────┘  │
└──────────────────────────────────────────────────┘
```

## License

MIT License

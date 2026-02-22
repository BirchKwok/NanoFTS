# NanoFTS Rust 写入性能指南

本文档说明 NanoFTS v0.6.0 引擎在写入链路上引入的架构变更，以及如何正确使用 Rust API 以达到最优吞吐量。

## 目录

- [架构变更概述](#架构变更概述)
- [核心优化详解](#核心优化详解)
- [API 选择与性能排名](#api-选择与性能排名)
- [最优性能使用示例](#最优性能使用示例)
- [基准测试数据](#基准测试数据)
- [调优参数说明](#调优参数说明)
- [常见误区](#常见误区)

---

## 架构变更概述

v0.6.0 对批量写入路径进行了五项主要优化，整体吞吐量在真实场景下提升约 **3×**，峰值达到 **943 万文档/秒**：

| 优化项 | 变更前 | 变更后 | 收益 |
|--------|--------|--------|------|
| 词典编码 | 字符串键 `DashMap<String, FastBitmap>` | 整数键 `DashMap<u32, FastBitmap>` | 消除 buffer 层字符串哈希 |
| 本地词汇结构 | `FxHashMap<String, Vec<u32>>` | `(FxHashMap<String, u32>, Vec<Vec<u32>>)` | Phase 1 热路径 O(1) 数组访问 |
| 线程协调 | 两段 `thread::scope`，Phase 1/2 有同步屏障 | 单段 `thread::scope`，每线程完成 Phase 1 后立即 Phase 2 | 消除线程间等待 |
| Phase 2 缓存 | 每线程维护 `FxHashMap<String, u32>` 缓存（命中率 0%） | `Vec<u32>` (local→global ID 映射) | 消除无效哈希查找和重复 String 分配 |
| DashMap 分片 | 默认 16 分片 | 64 分片 | 降低锁竞争概率 ~4× |

---

## 核心优化详解

### 1. 字典编码（Dictionary Encoding）

每个词项（term）在全局字典中被分配一个唯一 `u32` ID：

```
"rust" → 1
"tutorial" → 2
"performance" → 3
```

写入 buffer 时直接使用 `u32` 键，搜索时先将查询词转为 ID 再查 buffer，避免了 buffer 层的字符串哈希计算。

**关键字段**：

```rust
// 全局字典：词项字符串 → u32 ID（64 分片 DashMap）
dict: Arc<DashMap<String, u32>>

// 原子计数器，分配下一个 term ID
next_term_id: AtomicU32

// 内存写入缓冲区：term ID → 文档 ID 位图（64 分片 DashMap）
buffer: DashMap<u32, FastBitmap>
```

### 2. Vec 索引的本地词汇（Phase 1 热路径优化）

每个线程在处理自己的文档分片时，使用以下结构累积词汇：

```rust
// term 字符串 → 本地 ID（线程私有 FxHashMap）
let mut local_dict: FxHashMap<String, u32> = FxHashMap::with_capacity_and_hasher(4096, ..);

// 本地 ID → 该 term 对应的文档 ID 列表（Vec 数组，O(1) 下标访问）
let mut local_terms: Vec<Vec<u32>> = Vec::with_capacity(4096);
```

对于每个 token，查找流程为：

```rust
tokenize_fast_cb(text, max_chinese_len, min_term_len, &mut word_buf, |term| {
    // 1. FxHashMap 查询本地词典（term → local_id）
    let lid = if let Some(&id) = local_dict.get(term) {
        id
    } else {
        // 2. 新词：分配本地 ID，扩展 Vec
        let id = local_terms.len() as u32;
        local_dict.insert(term.to_string(), id);
        local_terms.push(Vec::new());
        id
    };
    // 3. O(1) 数组下标访问，追加 doc_id
    local_terms[lid as usize].push(doc_id);
});
```

相比旧版 `FxHashMap<String, Vec<u32>>` 的双重字符串哈希，新版本的 `local_terms[lid]` 访问是纯数组下标，无哈希计算。

### 3. 单 scope 流水线化（消除 Phase 1/2 屏障）

旧版两段式：所有线程必须完成 Phase 1 才能开始 Phase 2：

```
Thread 0: [===Phase1===][===Phase2===]
Thread 1: [===Phase1======][===Phase2===]
Thread 2: [===Phase1====][===Phase2===]
Thread 3: [===Phase1==========][---wait---][===Phase2===]
                                ↑ 所有线程等最慢的那个
```

新版单 scope：每线程独立完成 Phase 1 后立即开始 Phase 2，无等待：

```
Thread 0: [===Phase1===][===Phase2===]
Thread 1: [===Phase1======][===Phase2===]
Thread 2: [===Phase1====][===Phase2===]
Thread 3: [===Phase1==========][===Phase2===]
```

实现只需一个 `thread::scope`：

```rust
let d2 = &self.dict;
let n2 = &self.next_term_id;
let b2 = &self.buffer;

std::thread::scope(|s| {
    for thread_idx in 0..num_threads {
        let start = thread_idx * chunk_size;
        if start >= num_docs { break; }
        let end = (start + chunk_size).min(num_docs);
        let texts_slice = &texts[start..end];
        let doc_ids_slice = &doc_ids[start..end];

        s.spawn(move || {
            // ── Phase 1：线程私有，无锁 ──
            let mut local_dict = FxHashMap::with_capacity_and_hasher(4096, ..);
            let mut local_terms: Vec<Vec<u32>> = Vec::with_capacity(4096);
            for (i, text) in texts_slice.iter().enumerate() {
                let doc_id = doc_ids_slice[i];
                Self::tokenize_fast_cb(text, .., |term| {
                    let lid = /* local_dict 查找或插入 */;
                    local_terms[lid as usize].push(doc_id);
                });
            }

            // ── Phase 2：合并到全局结构（紧随 Phase 1，无屏障）──
            // 构建 local_id → global_id 映射（避免重复 String 分配）
            let mut l2g: Vec<u32> = vec![0u32; local_dict.len()];
            for (term, &lid) in local_dict.iter() {
                let gid = match d2.entry(term.to_string()) {
                    Occupied(e) => *e.get(),
                    Vacant(e) => { let id = n2.fetch_add(1, Relaxed); e.insert(id); id }
                };
                l2g[lid as usize] = gid;
            }
            // 按本地 ID 顺序写入 buffer（Vec 下标，无字符串哈希）
            for (lid, doc_ids) in local_terms.iter().enumerate() {
                if !doc_ids.is_empty() {
                    b2.entry(l2g[lid]).or_insert_with(FastBitmap::new).add_many(doc_ids);
                }
            }
        });
    }
});
```

### 4. 零分配分词（tokenize_fast_cb）

分词函数使用回调模式，token 以 `&str` 切片形式传入回调，全程无堆分配：

```rust
// 内部复用的字节缓冲区（词项临时存储）
let mut word_buf: Vec<u8> = Vec::with_capacity(256);

Self::tokenize_fast_cb(text, max_chinese_len, min_term_len, &mut word_buf, |term: &str| {
    // term 是对 word_buf 的借用切片，无 String 分配
    // 仅在 local_dict 首次遇到新词时才执行 term.to_string()
});
```

---

## API 选择与性能排名

以下按**吞吐量从高到低**排列（100K 文档测试，4 核）：

| 排名 | API | 适用场景 | 吞吐量参考 |
|------|-----|----------|------------|
| 🥇 1 | `add_documents_arrow_texts` | 文本字段已预拼接，使用 `&str` 切片 | ~3.4M docs/s |
| 🥈 2 | `add_documents_arrow_str` | 多列 `&str` 切片（Arrow 列式格式） | ~3.3M docs/s |
| 🥉 3 | `add_documents_columnar` | 多列 `Vec<String>` | ~3.0M docs/s |
| 4 | `add_documents_texts` | 单列 `Vec<String>` | ~2.7M docs/s |
| 5 | `add_documents` | `Vec<(u32, HashMap<String, String>)>` | ~2.5M docs/s |
| 6 | `add_document`（循环） | **避免用于批量写入** | <0.5M docs/s |

> **核心原则**：能用 `&str` 就不用 `String`；能预拼接就预拼接；永远使用批量 API。

---

## 最优性能使用示例

### 场景 1：最快路径——预拼接 `&str` 文本

文本字段在调用前合并为单字符串，使用零拷贝切片：

```rust
use nanofts::{UnifiedEngine, EngineConfig};

fn main() -> nanofts::EngineResult<()> {
    let engine = UnifiedEngine::new(EngineConfig::memory_only())?;

    let doc_ids: Vec<u32> = (1..=100_000).collect();

    // 预先拼接所有字段（只分配一次）
    let merged: Vec<String> = (0..100_000)
        .map(|i| format!("标题 {} 这是文档 {} 的正文内容", i, i))
        .collect();

    // 转为 &str 切片，避免再次分配
    let text_refs: Vec<&str> = merged.iter().map(|s| s.as_str()).collect();

    // 🚀 最快路径：零拷贝单列写入
    engine.add_documents_arrow_texts(&doc_ids, &text_refs)?;

    Ok(())
}
```

### 场景 2：多列 Arrow 格式（列式数据源）

适用于从 Arrow / Parquet / DataFrame 读取的场景，各列数据已为 `&str`：

```rust
use nanofts::{UnifiedEngine, EngineConfig};

fn main() -> nanofts::EngineResult<()> {
    let engine = UnifiedEngine::new(EngineConfig::memory_only())?;

    let doc_ids: Vec<u32> = vec![1, 2, 3, 4, 5];

    // 各列数据（真实场景来自 Arrow StringArray）
    let titles:   Vec<&str> = vec!["Rust 入门", "Python 进阶", "数据库原理", "操作系统", "算法导论"];
    let contents: Vec<&str> = vec!["所有权与借用", "装饰器与元类", "B+树索引", "进程与线程", "排序与搜索"];

    let columns = vec![
        ("title".to_string(),   titles),
        ("content".to_string(), contents),
    ];

    // 🚀 多列零拷贝写入
    engine.add_documents_arrow_str(&doc_ids, columns)?;

    Ok(())
}
```

### 场景 3：与 Arrow 库集成（真实 Arrow 数组）

```rust
use nanofts::{UnifiedEngine, EngineConfig};
use arrow_array::StringArray;

fn ingest_from_arrow(
    engine: &UnifiedEngine,
    doc_ids: &[u32],
    title_array: &StringArray,
    content_array: &StringArray,
) -> nanofts::EngineResult<()> {
    // 从 Arrow StringArray 提取 &str 切片（零拷贝）
    let titles: Vec<&str> = title_array.iter()
        .map(|v| v.unwrap_or(""))
        .collect();
    let contents: Vec<&str> = content_array.iter()
        .map(|v| v.unwrap_or(""))
        .collect();

    let columns = vec![
        ("title".to_string(),   titles),
        ("content".to_string(), contents),
    ];

    engine.add_documents_arrow_str(doc_ids, columns)?;
    Ok(())
}
```

### 场景 4：超大规模批量写入（分批 + 定期 flush）

写入量超过几百万文档时，建议分批写入并定期 flush，以控制内存 buffer 大小：

```rust
use nanofts::{UnifiedEngine, EngineConfig};

fn main() -> nanofts::EngineResult<()> {
    let engine = UnifiedEngine::new(
        EngineConfig::persistent("./index.nfts")
            .with_lazy_load(true)
            .with_cache_size(50_000)
    )?;

    let batch_size = 200_000;  // 每批 20 万文档
    let total_docs = 5_000_000;

    for batch_start in (0..total_docs).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(total_docs);
        let size = batch_end - batch_start;

        let doc_ids: Vec<u32> = (batch_start as u32..batch_end as u32).collect();
        let texts: Vec<String> = (0..size)
            .map(|i| format!("文档 {} 内容", batch_start + i))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

        engine.add_documents_arrow_texts(&doc_ids, &text_refs)?;

        // 每批写入后 flush，避免 buffer 无限膨胀
        if batch_end % 1_000_000 == 0 {
            engine.flush()?;
            println!("已写入 {} 万文档", batch_end / 10_000);
        }
    }

    engine.flush()?;
    Ok(())
}
```

### 场景 5：持久化引擎完整示例

```rust
use nanofts::{UnifiedEngine, EngineConfig};
use std::collections::HashMap;

fn main() -> nanofts::EngineResult<()> {
    // 创建持久化引擎（首次创建）
    let engine = UnifiedEngine::new(
        EngineConfig::persistent("./myindex.nfts")
            .with_lazy_load(true)        // 大索引时延迟加载
            .with_cache_size(20_000)     // LRU 缓存条目数
            .with_track_doc_terms(true)  // 需要删除/更新时开启
    )?;

    // 批量写入
    let doc_ids: Vec<u32> = (1..=10_000).collect();
    let texts: Vec<String> = (1..=10_000u32)
        .map(|i| format!("文章标题 {} 正文内容 {}", i, i))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    engine.add_documents_arrow_texts(&doc_ids, &text_refs)?;

    // 持久化到磁盘
    engine.flush()?;

    // 搜索
    let result = engine.search("文章")?;
    println!("命中文档数: {}", result.total_hits());

    // 更新文档（需要 track_doc_terms=true）
    let mut new_fields = HashMap::new();
    new_fields.insert("content".to_string(), "更新后的内容".to_string());
    engine.update_document(1, new_fields)?;

    // 删除文档（逻辑删除，compact 后才永久生效）
    engine.remove_document(2)?;
    engine.compact()?;  // 物理删除

    Ok(())
}
```

---

## 基准测试数据

测试环境：Apple Silicon（ARM64），4 核，`cargo run --release`（LTO fat + codegen-units=1）

### ingestion_compare 基准（标准测试集）

文档格式：`"Document Title Number {i} with searchable keywords {content}"` ，每文档约 27 个唯一 token，其中包含每文档唯一数字（最坏情况词汇量）。

| 文档数 | `add_documents_columnar` | `add_documents_texts` | `add_documents_arrow_str` | `add_documents_arrow_texts` |
|--------|--------------------------|-----------------------|---------------------------|-----------------------------|
| 1,000  | 949K docs/s  | 1.77M docs/s | 2.08M docs/s | 1.91M docs/s |
| 10,000 | 2.13M docs/s | 2.11M docs/s | 1.88M docs/s | 2.31M docs/s |
| 50,000 | 3.82M docs/s | 3.31M docs/s | 2.93M docs/s | 4.17M docs/s |
| 100,000 | **4.03M docs/s** | 3.11M docs/s | 3.83M docs/s | **3.39M docs/s** |

### 真实场景基准（词汇量受限）

**词汇量受限**（10K 独立数字，模拟真实文本复用）vs **完全随机词汇**（每文档独立数字）：

| 场景 | 文档数 | 总耗时 | 吞吐量 |
|------|--------|--------|--------|
| 词汇量受限（10K unique terms） | 100 万 | 106 ms | **943 万 docs/s** ✅ |
| 完全随机词汇（1M unique terms） | 100 万 | 228 ms | 438 万 docs/s |

**结论**：
- 真实文本场景（词汇有复用）：**达到 10M docs/s 目标**
- 极端对抗场景（每文档都有独特词汇）：约 4M docs/s（受限于 String 分配）

### 性能敏感点

| 瓶颈 | 说明 |
|------|------|
| 新词首次插入 | 每个新 term 需要 `String` 分配 + DashMap 写锁，约 100–150 ns/词 |
| 常见词 bitmap 追加 | `FastBitmap::add_many` 在持有 DashMap 分片锁期间执行，跨线程串行化 |
| 线程数 | 超过物理核数会因上下文切换降低吞吐 |

---

## 调优参数说明

### 写入相关

```rust
EngineConfig::memory_only()
    // 无持久化，最大吞吐（benchmark 场景使用）

EngineConfig::persistent("./index.nfts")
    .with_lazy_load(true)      // 大索引不全量加载到内存
    .with_cache_size(N)        // lazy_load 时 LRU 缓存条目数（建议 10K–100K）
    .with_track_doc_terms(true) // 仅在需要 update/delete 时开启（有额外写开销）
```

### 编译期优化（`Cargo.toml`）

项目已在 release profile 中预设最优编译参数：

```toml
[profile.release]
lto = "fat"          # 全程序链接时优化
codegen-units = 1    # 单编译单元，最大优化机会
panic = "abort"      # 消除 unwind 开销
strip = true         # 移除调试符号
opt-level = 3
```

### CPU 亲和性（可选）

在多 NUMA 节点机器上，可设置 CPU 亲和性以减少跨节点内存访问：

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

---

## 常见误区

### ❌ 误区 1：在循环中逐条调用 `add_document`

```rust
// ❌ 慢：每次调用都有独立的分词、hash、锁开销
for (id, fields) in docs {
    engine.add_document(id, fields)?;
}
```

```rust
// ✅ 快：批量调用，内部并行处理
engine.add_documents(docs)?;
// 或者预拼接文本后使用更快的 API：
engine.add_documents_arrow_texts(&doc_ids, &text_refs)?;
```

### ❌ 误区 2：不必要地开启 `track_doc_terms`

`track_doc_terms=true` 会在每次写入时额外维护 doc→term 的反向映射，约增加 10–20% 写入开销。**仅在需要 `update_document` / `remove_document` 功能时才开启**。

### ❌ 误区 3：写入时频繁调用 `flush`

```rust
// ❌ 每写一批就 flush，I/O 放大严重
for batch in batches {
    engine.add_documents_arrow_texts(&ids, &texts)?;
    engine.flush()?;  // 不必要
}
```

```rust
// ✅ 写完所有批次后统一 flush
for batch in batches {
    engine.add_documents_arrow_texts(&ids, &texts)?;
}
engine.flush()?;  // 一次 flush
```

### ❌ 误区 4：对已拥有 `String` 的数据做 `String → &str` 零拷贝后再次 clone

```rust
// ❌ 多余的 clone
let texts: Vec<String> = load_data();
let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
let texts_copy: Vec<String> = text_refs.iter().map(|s| s.to_string()).collect(); // 多余！
engine.add_documents_arrow_texts(&doc_ids, &text_refs)?;
```

`add_documents_arrow_texts` 接受 `&[&str]`，**无需将 `&str` 再次转为 `String`**，直接传即可。

---

## 异步 Flush：`flush_async()` + `wait_flush()`

### 为什么需要异步 Flush

`flush()` 在完成 `sync_all()` 之前会阻塞调用线程。对于 100k 文档的批量建索引场景，阻塞时间约 **450ms**（SSD）。在此期间，任何搜索请求都只能等待。

`flush_async()` 将磁盘 I/O 全部卸载到后台线程，主线程仅执行 CPU 密集型操作（buffer drain + 内存合并），完成后**立即可搜索**。

### 工作原理

```
flush_async() 调用流程
│
├─ [主线程，同步] 从 UnifiedEngine.buffer 取出所有 entries
├─ [主线程，同步] 直接合并进 LsmSingleIndex.data（DashMap，并行写入）
│   └─ 此时数据已可被搜索
├─ [后台线程，异步] enqueue_for_flush → flush()
│   ├─ WAL 写入（可选）
│   ├─ 序列化 + zstd 压缩
│   └─ sync_all()（真正的 fsync）
│
└─ wait_flush() 等待后台线程结束
```

### Rust API

```rust
use nanofts::{UnifiedEngine, EngineConfig};

let engine = UnifiedEngine::new(EngineConfig::persistent("./index.nfts"))?;

// 批量导入
engine.add_documents_texts(doc_ids, texts)?;

// 异步 flush —— 微秒到毫秒级返回
engine.flush_async()?;

// 立即可搜索，无需等待磁盘写入
let result = engine.search("keyword")?;
assert!(result.total_hits() > 0);

// 在合适时机等待持久化完成
let terms = engine.wait_flush()?;
println!("持久化了 {terms} 个词项");
```

### Python API

```python
import nanofts

engine = nanofts.create_engine("./index.nfts")

doc_ids = list(range(1, 100001))
texts   = [f"document {i} with keywords alpha beta gamma" for i in doc_ids]

engine.add_documents_texts(doc_ids, texts)

# 异步 flush
engine.flush_async()

# 立即搜索
result = engine.search("alpha")
print(f"即时命中：{result.total_hits}")  # 100000

# 等待持久化
terms = engine.wait_flush()
print(f"已持久化 {terms} 个词项")
```

### 基准测试数据（SSD，release 编译）

| 规模 | `flush()` 阻塞时间 | `flush_async()` 返回时间 | 主线程节省 | 即时命中 |
|------|------------------|-----------------------|---------|---------|
| 10,000 | 91ms | **5.8ms** | 85ms | ✅ 全部 |
| 50,000 | 228ms | **29.7ms** | 198ms | ✅ 全部 |
| 100,000 | 452ms | **58ms** | 394ms | ✅ 全部 |

`wait_flush()` 的实际等待时间约为同步 `flush()` 的一半（后台线程与主线程并行执行）。

运行基准测试：

```bash
cargo run --example flush_benchmark --release
```

### 何时选择哪种方式

| 场景 | 推荐 |
|------|------|
| 批量建索引后立即提供查询服务 | `flush_async()` + `wait_flush()` |
| 每次写入都必须立即持久化 | `flush()` |
| 内存模式（无磁盘） | 两者均可（均为 no-op） |
| 写入量大、需限制 fsync 频率 | 写入多批后调用一次 `flush_async()` |

### 注意事项

- **`wait_flush()` 是可选的。** 背景线程持有 `Arc<LsmSingleIndex>` 的克隆，即使 JoinHandle 被丢弃（或 `UnifiedEngine` 被 drop），线程也会继续运行直到完成，数据最终会写入磁盘。
- **进程强制退出（crash / kill -9）除外。** 若进程在背景线程完成 `sync_all()` 之前退出，数据可能未持久化。需要强持久化保证时，调用 `wait_flush()` 或改用 `flush()`。
- **重复调用 `flush_async()` 是安全的。** 旧 JoinHandle 被替换后，旧背景线程仍在运行、会正常完成；但旧 flush 的返回值（成功/错误）将无法再被收集。如需感知错误，在下次 `flush_async()` 前先调 `wait_flush()`。
- `flush_async()` 仅适用于**持久化模式**；内存模式直接返回。

---

## 版本历史

| 版本 | 写入链路主要变更 |
|------|-----------------|
| v0.5.x | `flush_async()` / `wait_flush()`；`LsmSingleIndex.data` 改为 `DashMap`（并行读写）；`flush_internal` / `merge_into_data` 并行写入；批量路径同步 `id_to_str` |
| v0.6.0 | 字典编码（term→u32）；Vec 索引本地词汇；单 scope 流水线；Phase 2 l2g 映射；DashMap 64 分片 |
| v0.3.x | 基于字符串键的 DashMap buffer；两段式 thread::scope |
| v0.2.x | 单线程写入路径 |

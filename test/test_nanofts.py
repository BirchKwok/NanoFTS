import pytest
from nanofts import create_engine, UnifiedEngine


@pytest.fixture
def test_data():
    """Test data fixture"""
    return [
        {"title": "Hello World", "content": "Python 全文搜索器"},  # id: 0 - 包含 hello, world, 全文, 搜索
        {"title": "GitHub Copilot", "content": "代码自动生成"},   # id: 1 - 包含 github, copilot
        {"title": "全文搜索", "content": "支持多语言", "tags": "测试数据"},  # id: 2 - 包含 全文, 搜索, 测试
        {"title": "hello", "content": "WORLD", "number": "123"},  # id: 3 - 包含 hello, world
        {"title": "数据处理", "content": "搜索引擎"},  # id: 4 - 包含 搜索
        {"title": "hello world", "content": "示例文本"},  # id: 5 - 包含 hello, world
        {"title": "混合文本", "content": "Mixed 全文内容测试"},  # id: 6 - 包含 mixed, 全文, 测试
        {"title": "hello world 你好", "content": "示例文本"},  # id: 7 - 包含 hello, world, 你好
    ]


@pytest.fixture
def engine(tmp_path):
    """Create an engine instance fixture"""
    index_file = tmp_path / "test_index.nfts"
    return create_engine(
        str(index_file),
        drop_if_exists=True,
        track_doc_terms=True
    )


def test_basic_search(engine, test_data):
    """Test the basic search functionality"""
    # Add test data
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()

    # Test word-based matching (AND semantics for multiple words)
    # "hello world" 会匹配所有同时包含 hello 和 world 的文档
    test_cases = [
        ("hello world", [0, 3, 5, 7]),  # 所有包含 hello AND world 的文档
        ("hello", [0, 3, 5, 7]),  # 所有包含 hello 的文档
        ("mixed", [6]),  # 只有文档 6 包含 mixed
        ("github", [1]),  # 只有文档 1 包含 github
    ]

    for query, expected in test_cases:
        result = engine.search(query)
        assert sorted(result.to_list()) == expected, f"Query '{query}' failed"


def test_chinese_search(engine, test_data):
    """Test the Chinese search functionality"""
    # Add test data
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()

    test_cases = [
        ("全文", [0, 2, 6]),
        ("搜索", [0, 2, 4]),
        ("测试", [2, 6])
    ]

    for query, expected in test_cases:
        result = engine.search(query)
        assert sorted(result.to_list()) == expected, f"Query '{query}' failed"


def test_incremental_update(engine, test_data):
    """Test the incremental update functionality"""
    # Add initial test data
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()

    # Add new document
    new_doc = {"title": "新增文档", "content": "测试全文搜索引擎"}
    new_doc_id = len(test_data)
    engine.add_document(new_doc_id, new_doc)
    engine.flush()

    # Test if the new document is searchable
    result = engine.search("新增")
    assert new_doc_id in result.to_list()
    
    result = engine.search("测试")
    assert new_doc_id in result.to_list()  # 新文档也包含测试

    # Test deleting documents
    engine.remove_document(new_doc_id)
    engine.flush()
    
    result = engine.search("新增")
    assert new_doc_id not in result.to_list()


def test_index_persistence(engine, test_data, tmp_path):
    """Test the index persistence functionality"""
    # Add test data
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()

    # Create a new engine instance to load the existing index
    index_file = tmp_path / "test_index.nfts"
    engine_reload = create_engine(str(index_file))

    test_cases = [
        ("hello", [0, 3, 5, 7]),  # 所有包含 hello 的文档
        ("全文", [0, 2, 6]),
        ("搜索", [0, 2, 4])
    ]

    for query, expected in test_cases:
        result = engine_reload.search(query)
        assert sorted(result.to_list()) == expected, f"Query '{query}' failed after reload"


def test_empty_search(engine):
    """Test the empty index search"""
    result = engine.search("任意查询")
    assert len(result.to_list()) == 0


def test_case_insensitive_search(engine, test_data):
    """Test the case insensitive search"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()

    # Test that queries with different cases return the same results
    lower_case = sorted(engine.search("hello").to_list())
    upper_case = sorted(engine.search("HELLO").to_list())
    mixed_case = sorted(engine.search("Hello").to_list())

    assert lower_case == upper_case == mixed_case 


def test_update_document(engine, test_data):
    """Test updating document terms"""
    # Add initial test data
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()

    # Initial search to verify setup
    hello_world_docs = sorted(engine.search("hello world").to_list())
    assert hello_world_docs == [0, 3, 5, 7]  # 所有同时包含 hello 和 world 的文档

    # Update document 0 - 移除 hello 和 world
    updated_doc = {
        "title": "Updated Title",
        "content": "新的内容 New Content",
        "tags": "updated"
    }
    engine.update_document(0, updated_doc)
    engine.flush()

    # Verify old terms are removed - 文档 0 不再包含 hello/world
    result = engine.search("hello world")
    assert 0 not in result.to_list()
    
    # 剩余的文档应该仍然能被搜索到
    assert sorted(result.to_list()) == [3, 5, 7]

    # Verify new terms are searchable
    assert 0 in engine.search("updated").to_list()
    assert 0 in engine.search("新的内容").to_list()


def test_update_nonexistent_document(engine):
    """Test updating a document that doesn't exist (adds as new)"""
    doc = {
        "title": "Brand New Document",
        "content": "Some unique content here"
    }
    # Should not raise an error - will add as new document
    engine.update_document(999, doc)
    engine.flush()
    
    # Verify the document is added and searchable
    result = engine.search("brand")
    assert 999 in result.to_list()
    
    result = engine.search("unique")
    assert 999 in result.to_list()


def test_batch_add_documents(engine, test_data):
    """测试批量添加文档功能"""
    # 使用 add_documents 批量添加
    docs = [(doc_id, doc) for doc_id, doc in enumerate(test_data)]
    count = engine.add_documents(docs)
    
    assert count == len(test_data)
    
    engine.flush()
    
    # 验证文档已添加
    assert len(engine.search("hello").to_list()) == 4  # 文档 0, 3, 5, 7
    assert len(engine.search("全文").to_list()) == 3  # 文档 0, 2, 6


def test_batch_remove_document(engine, test_data):
    """测试批量删除文档功能"""
    # 添加初始测试数据
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    # 初始搜索验证
    assert sorted(engine.search("hello").to_list()) == [0, 3, 5, 7]
    assert sorted(engine.search("github").to_list()) == [1]
    
    # 批量删除文档 0, 1, 5
    engine.remove_documents([0, 1, 5])
    engine.flush()

    # 验证文档已被删除
    hello_docs = engine.search("hello")
    assert 0 not in hello_docs.to_list()
    assert 5 not in hello_docs.to_list()
    # 文档 3 和 7 还在
    assert sorted(hello_docs.to_list()) == [3, 7]
    
    # github 的文档 1 已删除
    assert engine.search("github").total_hits == 0


def test_search_result_handle(engine, test_data):
    """测试搜索结果句柄"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    result = engine.search("hello")
    
    # 测试结果属性
    assert result.total_hits > 0
    assert result.elapsed_ns >= 0
    assert result.elapsed_ms() >= 0
    assert result.elapsed_us() >= 0
    
    # 测试结果方法
    assert not result.is_empty()
    assert len(result) > 0
    assert len(result.to_list()) > 0
    assert len(result.top(10)) > 0
    
    # 测试 contains
    for doc_id in result.to_list():
        assert result.contains(doc_id)


def test_search_and_operation(engine, test_data):
    """测试 AND 搜索操作"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    # 使用 search_and 进行 AND 搜索
    result = engine.search_and(["hello", "world"])
    
    # 验证结果包含同时含有 hello 和 world 的文档
    assert result.total_hits == 4  # 文档 0, 3, 5, 7
    assert sorted(result.to_list()) == [0, 3, 5, 7]


def test_search_or_operation(engine, test_data):
    """测试 OR 搜索操作"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    # 使用 search_or 进行 OR 搜索
    result_or = engine.search_or(["github", "mixed"])
    
    # github: 文档 1, mixed: 文档 6
    assert result_or.total_hits == 2
    assert sorted(result_or.to_list()) == [1, 6]


def test_result_set_operations(engine, test_data):
    """测试结果集操作"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    r1 = engine.search("hello")  # [0, 3, 5, 7]
    r2 = engine.search("world")  # [0, 3, 5, 7]
    r3 = engine.search("github")  # [1]
    
    # 交集 - hello AND world = [0, 3, 5, 7]
    intersection = r1.intersect(r2)
    assert sorted(intersection.to_list()) == [0, 3, 5, 7]
    
    # 并集 - hello OR github
    union = r1.union(r3)
    assert sorted(union.to_list()) == [0, 1, 3, 5, 7]
    
    # 差集 - hello NOT github = hello 的所有文档（因为 github 和 hello 没有交集）
    diff = r1.difference(r3)
    assert sorted(diff.to_list()) == [0, 3, 5, 7]


def test_filter_by_ids(engine, test_data):
    """测试按 ID 过滤结果"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    result = engine.search("hello")  # [0, 3, 5, 7]
    
    # 按 ID 过滤，只保留 [0, 1, 2, 3]
    filtered = engine.filter_by_ids(result, [0, 1, 2, 3])
    
    # 过滤后的结果应该只包含 [0, 3]（hello 结果与过滤列表的交集）
    assert sorted(filtered.to_list()) == [0, 3]


def test_exclude_ids(engine, test_data):
    """测试排除指定 ID"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    result = engine.search("hello")  # [0, 3, 5, 7]
    
    # 排除 ID 0 和 3
    excluded = engine.exclude_ids(result, [0, 3])
    
    # 排除后的结果应该是 [5, 7]
    assert sorted(excluded.to_list()) == [5, 7]


def test_search_batch(engine, test_data):
    """测试批量搜索"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    queries = ["hello", "github", "全文", "不存在的词"]
    results = engine.search_batch(queries)
    
    assert len(results) == 4
    assert results[0].total_hits == 4  # hello: [0, 3, 5, 7]
    assert results[1].total_hits == 1  # github: [1]
    assert results[2].total_hits == 3  # 全文: [0, 2, 6]
    assert results[3].total_hits == 0  # 不存在的词


def test_fuzzy_search(engine, test_data):
    """测试模糊搜索"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    # 测试模糊搜索
    result = engine.fuzzy_search("helo", min_results=0)  # 拼写错误的 hello
    
    # 结果应该是 ResultHandle 类型
    assert hasattr(result, 'total_hits')
    assert hasattr(result, 'to_list')


def test_numpy_support(engine, test_data):
    """测试 numpy 支持"""
    np = pytest.importorskip("numpy")
    
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    result = engine.search("hello")
    
    # 转换为 numpy 数组
    arr = result.to_numpy()
    
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint32


def test_stats(engine, test_data):
    """测试统计信息"""
    for doc_id, doc in enumerate(test_data):
        engine.add_document(doc_id, doc)
    engine.flush()
    
    # 执行一些搜索
    engine.search("hello")
    engine.search("world")
    
    stats = engine.stats()
    
    assert isinstance(stats, dict)
    assert "search_count" in stats
    assert stats["search_count"] >= 2


def test_memory_mode():
    """测试纯内存模式"""
    # 空路径表示纯内存模式
    engine = create_engine("")
    
    assert engine.is_memory_only()
    
    engine.add_document(1, {"content": "memory test"})
    
    result = engine.search("memory")
    assert 1 in result.to_list()


def test_large_batch_operations(engine):
    """测试大批量操作"""
    # 批量添加 100 个文档
    large_batch = [(i, {"content": f"batch document {i}", "tag": f"tag{i}"}) for i in range(100)]
    count = engine.add_documents(large_batch)
    assert count == 100
    engine.flush()
    
    # 验证所有文档可搜索
    result = engine.search("batch")
    assert result.total_hits == 100
    
    # 批量删除前 50 个
    engine.remove_documents(list(range(50)))
    engine.flush()
    
    # 验证删除后只剩 50 个
    result = engine.search("batch")
    assert result.total_hits == 50
    
    # 验证特定 tag 搜索
    assert engine.search("tag0").total_hits == 0  # 已删除
    assert engine.search("tag50").total_hits == 1  # 仍存在


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

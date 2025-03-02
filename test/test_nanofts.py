import pytest

from nanofts import FullTextSearch

@pytest.fixture
def test_data():
    """Test data fixture"""
    return [
        {"title": "Hello World", "content": "Python 全文搜索器"},  # id: 0
        {"title": "GitHub Copilot", "content": "代码自动生成"},   # id: 1
        {"title": "全文搜索", "content": "支持多语言", "tags": "测试数据"},  # id: 2
        {"title": "hello", "content": "WORLD", "number": 123},  # id: 3
        {"title": "数据处理", "content": "搜索引擎"},  # id: 4
        {"title": "hello world", "content": "示例文本"},  # id: 5
        {"title": "混合文本", "content": "Mixed 全文内容测试"},  # id: 6
        {"title": "hello world 你好", "content": "示例文本"},  # id: 7
    ]

@pytest.fixture
def fts(tmp_path):
    """Create a FTS instance fixture"""
    index_dir = tmp_path / "fts_index"
    return FullTextSearch(
        index_dir=str(index_dir),
        batch_size=1000,
        buffer_size=5000,
        drop_if_exists=True
    )

def test_basic_search(fts, test_data):
    """Test the basic search functionality"""
    # Add test data
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # Test exact matching
    test_cases = [
        ("Hello World", [0, 5]),  # Only match the complete "Hello World"
        ("hello world", [0, 5]),  # Case insensitive
        ("全文搜索", [0, 2]),  # Modified: Include all documents containing these words
        ("mixed", [6])
    ]

    for query, expected in test_cases:
        assert sorted(fts.search(query)) == expected

def test_chinese_search(fts, test_data):
    """Test the Chinese search functionality"""
    # Add test data
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    test_cases = [
        ("全文", [0, 2, 6]),
        ("搜索", [0, 2, 4]),
        ("测试", [2, 6])
    ]

    for query, expected in test_cases:
        assert sorted(fts.search(query)) == expected

def test_phrase_search(fts, test_data):
    """Test the phrase search functionality"""
    # Add test data
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    test_cases = [
        ("hello world", [0, 5]),  # Only match the complete phrase
        ("全文 搜索", [0, 2]),
        ("python 搜索", [0])
    ]

    for query, expected in test_cases:
        assert sorted(fts.search(query)) == expected

def test_incremental_update(fts, test_data):
    """Test the incremental update functionality"""
    # Add initial test data
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # Add new document
    new_doc = {"title": "新增文档", "content": "测试全文搜索", "tags": "hello world test"}
    new_doc_id = len(test_data)
    fts.add_document(new_doc_id, new_doc)
    fts.flush()

    # Test if the new document is searchable
    test_cases = [
        ("新增", [new_doc_id]),
        ("测试", [2, 6, new_doc_id]),
        ("hello world", [0, 5])  # Only match the complete phrase
    ]

    for query, expected in test_cases:
        assert sorted(fts.search(query)) == expected

    # Test deleting documents
    fts.remove_document(new_doc_id)
    
    for query, expected in test_cases:
        result = sorted(fts.search(query))
        # Remove the deleted document ID from the expected result
        expected = [id for id in expected if id != new_doc_id]
        assert result == expected

def test_index_persistence(fts, test_data, tmp_path):
    """Test the index persistence functionality"""
    # Add test data
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # Create a new FTS instance to load the existing index
    index_dir = tmp_path / "fts_index"
    fts_reload = FullTextSearch(index_dir=str(index_dir))

    test_cases = [
        ("hello world", [0, 5]),  # Only match the complete phrase
        ("全文", [0, 2, 6]),
        ("搜索", [0, 2, 4])
    ]

    for query, expected in test_cases:
        assert sorted(fts_reload.search(query)) == expected

def test_empty_search(fts):
    """Test the empty index search"""
    assert len(fts.search("任意查询")) == 0

def test_invalid_document_input(fts):
    """Test the invalid document input"""
    with pytest.raises(ValueError):
        fts.add_document([1, 2], [{"title": "doc1"}])  # The document ID and document number do not match

def test_case_insensitive_search(fts, test_data):
    """Test the case insensitive search"""
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # Test that queries with different cases return the same results
    lower_case = sorted(fts.search("hello world"))
    upper_case = sorted(fts.search("HELLO WORLD"))
    mixed_case = sorted(fts.search("Hello World"))

    assert lower_case == upper_case == mixed_case 

def test_from_pandas(tmp_path):
    """Test importing data from a pandas DataFrame"""
    pd = pytest.importorskip("pandas")
    
    # Create test data
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'title': ['Hello World', '全文搜索', 'Test Document'],
        'content': ['This is a test', '支持多语言', 'Another test'],
        'tags': ['test, hello', '搜索, 测试', 'doc, test']
    })
    
    # Create an instance and import data
    fts = FullTextSearch(index_dir=str(tmp_path / "fts_index"))
    fts.from_pandas(df, id_column='id')
    
    # Test the search results
    assert sorted(fts.search("test")) == [1, 3]
    assert sorted(fts.search("搜索")) == [2]
    assert sorted(fts.search("hello")) == [1]

def test_from_polars(tmp_path):
    """Test importing data from a polars DataFrame"""
    pl = pytest.importorskip("polars")
    
    # Create test data
    df = pl.DataFrame({
        'id': [1, 2, 3],
        'title': ['Hello World', '全文搜索', 'Test Document'],
        'content': ['This is a test', '支持多语言', 'Another test'],
        'tags': ['test, hello', '搜索, 测试', 'doc, test']
    })
    
    # Create an instance and import data
    fts = FullTextSearch(index_dir=str(tmp_path / "fts_index"))
    fts.from_polars(df, id_column='id')
    
    # Test the search results
    assert sorted(fts.search("test")) == [1, 3]
    assert sorted(fts.search("搜索")) == [2]
    assert sorted(fts.search("hello")) == [1]

def test_from_arrow(tmp_path):
    """Test importing data from a pyarrow Table"""
    pa = pytest.importorskip("pyarrow")
    
    # Create test data
    data = {
        'id': [1, 2, 3],
        'title': ['Hello World', '全文搜索', 'Test Document'],
        'content': ['This is a test', '支持多语言', 'Another test'],
        'tags': ['test, hello', '搜索, 测试', 'doc, test']
    }
    table = pa.Table.from_pydict(data)
    
    # Create an instance and import data
    fts = FullTextSearch(index_dir=str(tmp_path / "fts_index"))
    fts.from_arrow(table, id_column='id')
    
    # Test the search results
    assert sorted(fts.search("test")) == [1, 3]
    assert sorted(fts.search("搜索")) == [2]
    assert sorted(fts.search("hello")) == [1]

def test_from_parquet(tmp_path):
    """Test importing data from a parquet file"""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    
    # Create test data
    data = {
        'id': [1, 2, 3],
        'title': ['Hello World', '全文搜索', 'Test Document'],
        'content': ['This is a test', '支持多语言', 'Another test'],
        'tags': ['test, hello', '搜索, 测试', 'doc, test']
    }
    table = pa.Table.from_pydict(data)
    
    # Save as a parquet file
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(table, parquet_path)
    
    # Create an instance and import data
    fts = FullTextSearch(index_dir=str(tmp_path / "fts_index"))
    fts.from_parquet(parquet_path, id_column='id')
    
    # Test the search results
    assert sorted(fts.search("test")) == [1, 3]
    assert sorted(fts.search("搜索")) == [2]
    assert sorted(fts.search("hello")) == [1] 

def test_from_csv(tmp_path):
    """Test importing data from a csv file"""
    pd = pytest.importorskip("pandas")
    
    # Create test data
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'title': ['Hello World', '全文搜索', 'Test Document'],
        'content': ['This is a test', '支持多语言', 'Another test'],
        'tags': ['test, hello', '搜索, 测试', 'doc, test']
    })
    
    # Save as a csv file
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)
    
    # Create an instance and import data
    fts = FullTextSearch(index_dir=str(tmp_path / "fts_index"))
    fts.from_csv(csv_path, id_column='id')

    # Test the search results
    assert sorted(fts.search("test")) == [1, 3]
    assert sorted(fts.search("搜索")) == [2]
    assert sorted(fts.search("hello")) == [1]   

def test_update_document(fts, test_data):
    """Test updating document terms"""
    # Add initial test data
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # Initial search to verify setup
    assert sorted(fts.search("hello world")) == [0, 5]
    assert sorted(fts.search("全文搜索")) == [0, 2]

    # Update document 0
    updated_doc = {
        "title": "Updated Title",
        "content": "新的内容 New Content",
        "tags": "updated"
    }
    fts.update_document(0, updated_doc)
    fts.flush()

    # Verify old terms are removed
    assert sorted(fts.search("hello world")) == [5]  # Document 0 no longer matches
    assert sorted(fts.search("全文搜索")) == [2]  # Document 0 no longer matches

    # Verify new terms are searchable
    assert sorted(fts.search("updated")) == [0]
    assert sorted(fts.search("新的内容")) == [0]

    # Update document with mixed content
    mixed_doc = {
        "title": "Mixed 混合",
        "content": "Content 内容",
        "tags": "test"
    }
    fts.update_document(5, mixed_doc)
    fts.flush()

    # Verify the updates
    assert set(fts.search("mixed")) == {5, 6}  # Both doc 5 and 6 contain "mixed"
    assert set(fts.search("混合")) == {5, 6}  # Both doc 5 and 6 contain "混合"
    assert set(fts.search("hello world")) == {3, 7}  # Documents 3 and 7 still contain "hello" and "world"

def test_update_nonexistent_document(fts):
    """Test updating a document that doesn't exist"""
    doc = {
        "title": "New Doc",
        "content": "Some content"
    }
    # Should not raise an error
    fts.update_document(999, doc)
    fts.flush()
    
    # Verify the document is added
    assert sorted(fts.search("new doc")) == [999]

def test_batch_update_document(fts, test_data):
    """测试批量更新文档功能"""
    # 添加初始测试数据
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()
    
    # 初始搜索验证
    assert sorted(fts.search("hello world")) == [0, 5]
    assert sorted(fts.search("全文搜索")) == [0, 2]
    assert sorted(fts.search("github")) == [1]
    
    # 批量更新文档
    updated_docs = [
        {
            "title": "Updated Title 1",
            "content": "新的内容 New Content 1",
            "tags": "tag1"
        },
        {
            "title": "Updated GitHub",
            "content": "更新的代码生成器",
            "tags": "ai"
        },
        {
            "title": "Mixed 混合",
            "content": "Content 内容",
            "tags": "test"
        }
    ]
    
    # 更新文档 0, 1, 5
    fts.batch_update_document([0, 1, 5], updated_docs)
    fts.flush()
    
    # 验证旧词条已被删除
    assert sorted(fts.search("hello world")) == [3, 7]  # 文档 0 和 5 已更新，但文档 3 和 7 仍然匹配
    assert sorted(fts.search("全文搜索")) == [2]  # 文档 0 不再匹配
    assert sorted(fts.search("github copilot")) == []  # 文档 1 不再匹配
    
    # 验证新词条可搜索
    assert sorted(fts.search("tag1")) == [0]  # 只有文档 0 包含 tag1
    assert sorted(fts.search("新的内容")) == [0]
    assert sorted(fts.search("更新的代码")) == [1]
    assert sorted(fts.search("混合")) == [5, 6]  # 文档 5 和 6 都包含"混合"
    
    # 验证未更新的文档仍然可搜索
    assert sorted(fts.search("测试数据")) == [2]
    assert sorted(fts.search("world")) == [3, 7]  # 文档 3 和 7 仍然包含 "world"
    
    # 测试批量更新的性能优势 - 同时更新多个文档
    large_batch = []
    large_ids = list(range(100, 150))
    
    for i in range(50):
        large_batch.append({
            "title": f"Batch Doc {i}",
            "content": f"批量文档内容 {i}",
            "tags": f"tag{i}"
        })
    
    # 批量添加
    fts.add_document(large_ids, large_batch)
    fts.flush()
    
    # 验证添加成功
    assert len(fts.search("批量文档内容")) == 50
    
    # 批量更新
    updated_batch = []
    for i in range(50):
        updated_batch.append({
            "title": f"Updated Batch {i}",
            "content": f"更新的批量内容 {i}",
            "tags": f"updated{i}"
        })
    
    fts.batch_update_document(large_ids, updated_batch)
    fts.flush()
    
    # 验证更新成功 - 使用更具体的搜索词
    # 验证旧内容已被删除 - 使用更具体的词组而不是单个词
    assert len(fts.search("批量文档内容 0")) == 0  # 特定文档内容应该被删除
    assert len(fts.search("批量文档内容 10")) == 0  # 特定文档内容应该被删除
    assert len(fts.search("批量文档内容 49")) == 0  # 特定文档内容应该被删除
    
    # 验证新内容已添加
    assert len(fts.search("更新的批量内容")) == 50  # 新内容已添加

def test_batch_remove_document(fts, test_data):
    """测试批量删除文档功能"""
    # 添加初始测试数据
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()
    
    # 初始搜索验证
    assert sorted(fts.search("hello world")) == [0, 5]
    assert sorted(fts.search("全文搜索")) == [0, 2]
    assert sorted(fts.search("github")) == [1]
    
    # 批量删除文档 0, 1, 5
    fts.batch_remove_document([0, 1, 5])
    fts.flush()
    
    # 验证文档已被删除
    assert sorted(fts.search("hello world")) == [3, 7]  # 文档 0 和 5 已删除，但文档 3 和 7 仍然匹配
    assert sorted(fts.search("全文搜索")) == [2]  # 文档 0 已删除
    assert sorted(fts.search("github")) == []  # 文档 1 已删除
    
    # 验证未删除的文档仍然可搜索
    assert sorted(fts.search("测试数据")) == [2]
    assert sorted(fts.search("world")) == [3, 7]  # 文档 3 和 7 仍然包含 "world"
    
    # 测试批量删除的性能优势 - 同时删除多个文档
    large_batch = []
    large_ids = list(range(100, 150))
    
    for i in range(50):
        large_batch.append({
            "title": f"Batch Doc {i}",
            "content": f"批量文档内容 {i}",
            "tags": f"tag{i}"
        })
    
    # 批量添加
    fts.add_document(large_ids, large_batch)
    fts.flush()
    
    # 验证添加成功
    assert len(fts.search("批量文档内容")) == 50
    
    # 批量删除一半文档
    fts.batch_remove_document(large_ids[:25])
    fts.flush()
    
    # 验证删除成功
    assert len(fts.search("批量文档内容")) == 25  # 只剩下一半文档
    
    # 验证特定文档已删除
    for i in range(25):
        assert len(fts.search(f"批量文档内容 {i}")) == 0  # 前25个文档应该已被删除
    
    # 验证剩余文档仍可搜索
    for i in range(25, 50):
        assert len(fts.search(f"批量文档内容 {i}")) == 1  # 后25个文档应该仍然存在

import pytest

from nanofts import FullTextSearch

@pytest.fixture
def test_data():
    """测试数据fixture"""
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
    """创建FTS实例的fixture"""
    index_dir = tmp_path / "fts_index"
    return FullTextSearch(
        index_dir=str(index_dir),
        batch_size=1000,
        buffer_size=5000,
        drop_if_exists=True
    )

def test_basic_search(fts, test_data):
    """测试基本搜索功能"""
    # 添加测试数据
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # 测试精确匹配
    test_cases = [
        ("Hello World", [0, 5]),  # 只匹配完整的"Hello World"
        ("hello world", [0, 5]),  # 大小写不敏感
        ("全文搜索", [0, 2]),  # 修改：包含所有包含这些词的文档
        ("mixed", [6])
    ]

    for query, expected in test_cases:
        assert sorted(fts.search(query)) == expected

def test_chinese_search(fts, test_data):
    """测试中文搜索功能"""
    # 添加测试数据
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
    """测试词组搜索功能"""
    # 添加测试数据
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    test_cases = [
        ("hello world", [0, 5]),  # 只匹配完整短语
        ("全文 搜索", [0, 2, 6]),  # 修改：包含所有包含这些词的文档
        ("python 搜索", [0])
    ]

    for query, expected in test_cases:
        assert sorted(fts.search(query)) == expected

def test_incremental_update(fts, test_data):
    """测试增量更新功能"""
    # 添加初始测试数据
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # 添加新文档
    new_doc = {"title": "新增文档", "content": "测试全文搜索", "tags": "hello world test"}
    new_doc_id = len(test_data)
    fts.add_document(new_doc_id, new_doc)
    fts.flush()

    # 测试新增文档是否可搜索
    test_cases = [
        ("新增", [new_doc_id]),
        ("测试", [2, 6, new_doc_id]),
        ("hello world", [0, 5])  # 只匹配完整短语
    ]

    for query, expected in test_cases:
        assert sorted(fts.search(query)) == expected

    # 测试删除文档
    fts.remove_document(new_doc_id)
    
    for query, expected in test_cases:
        result = sorted(fts.search(query))
        # 从预期结果中移除已删除的文档ID
        expected = [id for id in expected if id != new_doc_id]
        assert result == expected

def test_index_persistence(fts, test_data, tmp_path):
    """测试索引持久化功能"""
    # 添加测试数据
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # 创建新的FTS实例加载已有索引
    index_dir = tmp_path / "fts_index"
    fts_reload = FullTextSearch(index_dir=str(index_dir))

    test_cases = [
        ("hello world", [0, 5]),  # 只匹配完整短语
        ("全文", [0, 2, 6]),
        ("搜索", [0, 2, 4])
    ]

    for query, expected in test_cases:
        assert sorted(fts_reload.search(query)) == expected

def test_empty_search(fts):
    """测试空索引搜索"""
    assert fts.search("任意查询") == []

def test_invalid_document_input(fts):
    """测试无效的文档输入"""
    with pytest.raises(ValueError):
        fts.add_document([1, 2], [{"title": "doc1"}])  # 文档ID和文档数量不匹配

def test_case_insensitive_search(fts, test_data):
    """测试大小写不敏感搜索"""
    doc_ids = list(range(len(test_data)))
    fts.add_document(doc_ids, test_data)
    fts.flush()

    # 测试不同大小写的查询返回相同结果
    lower_case = sorted(fts.search("hello world"))
    upper_case = sorted(fts.search("HELLO WORLD"))
    mixed_case = sorted(fts.search("Hello World"))

    assert lower_case == upper_case == mixed_case 
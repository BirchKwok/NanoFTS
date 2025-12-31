"""
NanoFTS ResultHandle 测试

测试搜索结果句柄的各种操作：
- 基本属性
- 集合操作（交集、并集、差集）
- 分页和转换
- numpy 支持
"""

import pytest
from nanofts import create_engine


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_result_handle.nfts")


@pytest.fixture
def engine_with_data(tmp_index_file):
    """创建带有测试数据的引擎"""
    engine = create_engine(tmp_index_file, drop_if_exists=True)
    
    # 添加测试数据
    test_data = [
        (0, {"content": "apple banana cherry"}),
        (1, {"content": "apple banana"}),
        (2, {"content": "apple"}),
        (3, {"content": "banana cherry"}),
        (4, {"content": "cherry date"}),
        (5, {"content": "date elderberry"}),
        (6, {"content": "apple date fig"}),
        (7, {"content": "banana fig grape"}),
        (8, {"content": "cherry grape"}),
        (9, {"content": "date fig grape"}),
    ]
    
    for doc_id, fields in test_data:
        engine.add_document(doc_id, fields)
    engine.flush()
    
    return engine


class TestResultHandleBasics:
    """ResultHandle 基本属性测试"""
    
    def test_total_hits(self, engine_with_data):
        """测试总命中数"""
        result = engine_with_data.search("apple")
        assert result.total_hits == 4  # docs 0, 1, 2, 6
    
    def test_is_empty(self, engine_with_data):
        """测试空结果检查"""
        result = engine_with_data.search("nonexistent")
        assert result.is_empty()
        
        result = engine_with_data.search("apple")
        assert not result.is_empty()
    
    def test_contains(self, engine_with_data):
        """测试包含检查"""
        result = engine_with_data.search("apple")
        
        assert result.contains(0)
        assert result.contains(1)
        assert result.contains(2)
        assert result.contains(6)
        assert not result.contains(3)
        assert not result.contains(5)
    
    def test_elapsed_time(self, engine_with_data):
        """测试耗时统计"""
        result = engine_with_data.search("apple")
        
        assert result.elapsed_ns > 0
        assert result.elapsed_ms() >= 0
        assert result.elapsed_us() >= 0
        
        # elapsed_ns 应该等于 elapsed_us * 1000 约等于 elapsed_ms * 1000000
        assert abs(result.elapsed_ns - result.elapsed_us() * 1000) < 100
    
    def test_fuzzy_used_flag(self, engine_with_data):
        """测试模糊搜索标志"""
        exact_result = engine_with_data.search("apple")
        assert not exact_result.fuzzy_used
        
        fuzzy_result = engine_with_data.fuzzy_search("appl", min_results=0)
        # 模糊搜索可能使用也可能不使用，取决于是否找到足够结果
    
    def test_len_method(self, engine_with_data):
        """测试 __len__ 方法"""
        result = engine_with_data.search("apple")
        assert len(result) == 4
    
    def test_repr_method(self, engine_with_data):
        """测试 __repr__ 方法"""
        result = engine_with_data.search("apple")
        repr_str = repr(result)
        
        assert "ResultHandle" in repr_str
        assert "hits=" in repr_str
        assert "query=" in repr_str
        assert "apple" in repr_str


class TestResultHandleConversion:
    """ResultHandle 转换方法测试"""
    
    def test_to_list(self, engine_with_data):
        """测试转换为列表"""
        result = engine_with_data.search("apple")
        doc_list = result.to_list()
        
        assert isinstance(doc_list, list)
        assert set(doc_list) == {0, 1, 2, 6}
    
    def test_top_n(self, engine_with_data):
        """测试获取前 N 个结果"""
        result = engine_with_data.search("apple")
        
        top_2 = result.top(2)
        assert len(top_2) == 2
        
        top_10 = result.top(10)
        assert len(top_10) == 4  # 只有 4 个匹配
    
    def test_page(self, engine_with_data):
        """测试分页"""
        result = engine_with_data.search("apple")
        
        page_1 = result.page(0, 2)  # 第一页，每页 2 个
        assert len(page_1) == 2
        
        page_2 = result.page(2, 2)  # 第二页
        assert len(page_2) == 2
        
        # 超出范围
        page_3 = result.page(4, 2)
        assert len(page_3) == 0
    
    def test_to_numpy(self, engine_with_data):
        """测试转换为 numpy 数组"""
        np = pytest.importorskip("numpy")
        
        result = engine_with_data.search("apple")
        arr = result.to_numpy()
        
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint32
        assert len(arr) == 4
        assert set(arr) == {0, 1, 2, 6}


class TestSetOperations:
    """集合操作测试"""
    
    def test_intersect(self, engine_with_data):
        """测试交集操作"""
        result_apple = engine_with_data.search("apple")
        result_banana = engine_with_data.search("banana")
        
        # apple: {0, 1, 2, 6}
        # banana: {0, 1, 3, 7}
        # 交集: {0, 1}
        
        intersection = result_apple.intersect(result_banana)
        
        assert intersection.total_hits == 2
        assert set(intersection.to_list()) == {0, 1}
    
    def test_union(self, engine_with_data):
        """测试并集操作"""
        result_apple = engine_with_data.search("apple")
        result_cherry = engine_with_data.search("cherry")
        
        # apple: {0, 1, 2, 6}
        # cherry: {0, 3, 4, 8}
        # 并集: {0, 1, 2, 3, 4, 6, 8}
        
        union = result_apple.union(result_cherry)
        
        assert union.total_hits == 7
        assert set(union.to_list()) == {0, 1, 2, 3, 4, 6, 8}
    
    def test_difference(self, engine_with_data):
        """测试差集操作"""
        result_apple = engine_with_data.search("apple")
        result_banana = engine_with_data.search("banana")
        
        # apple: {0, 1, 2, 6}
        # banana: {0, 1, 3, 7}
        # apple - banana: {2, 6}
        
        difference = result_apple.difference(result_banana)
        
        assert difference.total_hits == 2
        assert set(difference.to_list()) == {2, 6}
    
    def test_chained_operations(self, engine_with_data):
        """测试链式操作"""
        r1 = engine_with_data.search("apple")
        r2 = engine_with_data.search("banana")
        r3 = engine_with_data.search("cherry")
        
        # (apple AND banana) OR cherry
        result = r1.intersect(r2).union(r3)
        
        # apple AND banana: {0, 1}
        # cherry: {0, 3, 4, 8}
        # union: {0, 1, 3, 4, 8}
        
        assert result.total_hits == 5
        assert set(result.to_list()) == {0, 1, 3, 4, 8}
    
    def test_operations_on_empty_results(self, engine_with_data):
        """测试空结果的集合操作"""
        empty_result = engine_with_data.search("nonexistent")
        apple_result = engine_with_data.search("apple")
        
        # 空集与任意集的交集为空
        assert empty_result.intersect(apple_result).is_empty()
        
        # 空集与任意集的并集为任意集
        assert empty_result.union(apple_result).total_hits == apple_result.total_hits
        
        # 空集与任意集的差集为空
        assert empty_result.difference(apple_result).is_empty()
        
        # 任意集减空集等于任意集
        assert apple_result.difference(empty_result).total_hits == apple_result.total_hits
    
    def test_self_operations(self, engine_with_data):
        """测试自身集合操作"""
        result = engine_with_data.search("apple")
        
        # A AND A = A
        assert result.intersect(result).total_hits == result.total_hits
        
        # A OR A = A
        assert result.union(result).total_hits == result.total_hits
        
        # A - A = 空
        assert result.difference(result).is_empty()


class TestSearchAndOperations:
    """搜索 AND 操作测试"""
    
    def test_search_and(self, engine_with_data):
        """测试 AND 搜索"""
        result = engine_with_data.search_and(["apple", "banana"])
        
        # 同时包含 apple 和 banana 的文档
        assert result.total_hits == 2
        assert set(result.to_list()) == {0, 1}
    
    def test_search_and_multiple(self, engine_with_data):
        """测试多个条件的 AND 搜索"""
        result = engine_with_data.search_and(["apple", "banana", "cherry"])
        
        # 同时包含所有三个词的文档
        assert result.total_hits == 1
        assert 0 in result.to_list()
    
    def test_search_and_no_match(self, engine_with_data):
        """测试没有匹配的 AND 搜索"""
        result = engine_with_data.search_and(["apple", "elderberry"])
        
        # 没有文档同时包含这两个词
        assert result.is_empty()


class TestSearchOrOperations:
    """搜索 OR 操作测试"""
    
    def test_search_or(self, engine_with_data):
        """测试 OR 搜索"""
        result = engine_with_data.search_or(["apple", "elderberry"])
        
        # 包含 apple 或 elderberry 的文档
        # apple: {0, 1, 2, 6}, elderberry: {5}
        assert result.total_hits == 5
        assert set(result.to_list()) == {0, 1, 2, 5, 6}
    
    def test_search_or_multiple(self, engine_with_data):
        """测试多个条件的 OR 搜索"""
        result = engine_with_data.search_or(["apple", "banana", "cherry"])
        
        # 包含任意一个词的文档
        assert result.total_hits >= 4


class TestFilterOperations:
    """过滤操作测试"""
    
    def test_filter_by_ids(self, engine_with_data):
        """测试按 ID 过滤"""
        result = engine_with_data.search("apple")
        
        # 只保留 ID 为 0, 2 的结果
        filtered = engine_with_data.filter_by_ids(result, [0, 2, 3, 4, 5])
        
        assert filtered.total_hits == 2
        assert set(filtered.to_list()) == {0, 2}
    
    def test_exclude_ids(self, engine_with_data):
        """测试排除 ID"""
        result = engine_with_data.search("apple")
        
        # 排除 ID 0 和 1
        excluded = engine_with_data.exclude_ids(result, [0, 1])
        
        assert excluded.total_hits == 2
        assert set(excluded.to_list()) == {2, 6}
    
    def test_filter_empty_list(self, engine_with_data):
        """测试空过滤列表"""
        result = engine_with_data.search("apple")
        
        # 用空列表过滤应该返回空结果
        filtered = engine_with_data.filter_by_ids(result, [])
        assert filtered.is_empty()
        
        # 排除空列表应该返回原结果
        excluded = engine_with_data.exclude_ids(result, [])
        assert excluded.total_hits == result.total_hits


class TestBatchSearch:
    """批量搜索测试"""
    
    def test_search_batch(self, engine_with_data):
        """测试批量搜索"""
        queries = ["apple", "banana", "cherry"]
        results = engine_with_data.search_batch(queries)
        
        assert len(results) == 3
        
        # 验证每个结果
        assert results[0].total_hits == 4  # apple
        assert results[1].total_hits == 4  # banana
        assert results[2].total_hits == 4  # cherry
    
    def test_search_batch_empty(self, engine_with_data):
        """测试空批量搜索"""
        results = engine_with_data.search_batch([])
        assert len(results) == 0
    
    def test_search_batch_mixed(self, engine_with_data):
        """测试混合批量搜索"""
        queries = ["apple", "nonexistent", "banana"]
        results = engine_with_data.search_batch(queries)
        
        assert len(results) == 3
        assert results[0].total_hits > 0
        assert results[1].is_empty()
        assert results[2].total_hits > 0


class TestResultHandleConsistency:
    """结果一致性测试"""
    
    def test_multiple_searches_same_query(self, engine_with_data):
        """测试同一查询多次搜索结果一致"""
        results = [engine_with_data.search("apple") for _ in range(5)]
        
        # 所有结果应该相同
        first_set = set(results[0].to_list())
        for r in results[1:]:
            assert set(r.to_list()) == first_set
    
    def test_result_immutability(self, engine_with_data):
        """测试结果不可变性"""
        result = engine_with_data.search("apple")
        original_hits = result.total_hits
        original_list = result.to_list()
        
        # 操作创建新对象，不修改原对象
        result.intersect(engine_with_data.search("banana"))
        result.union(engine_with_data.search("cherry"))
        result.difference(engine_with_data.search("date"))
        
        assert result.total_hits == original_hits
        assert result.to_list() == original_list


class TestResultHandleEdgeCases:
    """边缘情况测试"""
    
    def test_very_large_result(self, tmp_index_file):
        """测试大量结果"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 创建 1000 个文档
        for i in range(1000):
            engine.add_document(i, {"content": f"large result test {i}"})
        engine.flush()
        
        result = engine.search("large")
        
        assert result.total_hits == 1000
        assert len(result.to_list()) == 1000
        
        # 测试分页
        for page in range(10):
            page_result = result.page(page * 100, 100)
            assert len(page_result) == 100
    
    def test_unicode_in_query_result(self, tmp_index_file):
        """测试 Unicode 查询结果"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        engine.add_document(1, {"content": "中文测试内容"})
        engine.add_document(2, {"content": "中文另一个测试"})
        engine.flush()
        
        result = engine.search("中文")
        
        assert result.total_hits == 2
        assert 1 in result.to_list()
        assert 2 in result.to_list()
    
    def test_special_characters_query(self, tmp_index_file):
        """测试特殊字符查询结果"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        engine.add_document(1, {"content": "test@email.com"})
        engine.add_document(2, {"content": "user@domain.org"})
        engine.flush()
        
        result = engine.search("test")
        assert 1 in result.to_list()


class TestFuzzySearchResults:
    """模糊搜索结果测试"""
    
    def test_fuzzy_search_result_type(self, engine_with_data):
        """测试模糊搜索返回类型"""
        result = engine_with_data.fuzzy_search("appel", min_results=0)
        
        # 应该返回 ResultHandle 类型
        assert hasattr(result, 'total_hits')
        assert hasattr(result, 'to_list')
        assert hasattr(result, 'intersect')
    
    def test_fuzzy_result_operations(self, engine_with_data):
        """测试模糊搜索结果的操作"""
        fuzzy_result = engine_with_data.fuzzy_search("appel", min_results=0)
        exact_result = engine_with_data.search("banana")
        
        # 应该能进行集合操作
        intersection = fuzzy_result.intersect(exact_result)
        union = fuzzy_result.union(exact_result)
        
        # 操作不应该失败
        assert intersection is not None
        assert union is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
NanoFTS API 覆盖测试

确保 100% 覆盖所有公开 API：
- create_engine() 函数及所有参数
- UnifiedEngine (SearchEngine) 类所有方法
- ResultHandle (SearchResult) 类所有方法
- FuzzyConfig 类
- 数据导入方法
- 别名测试
"""

import pytest
from nanofts import (
    create_engine,
    UnifiedEngine,
    ResultHandle,
    FuzzyConfig,
    SearchEngine,
    SearchResult,
    __version__,
)


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_api.nfts")


@pytest.fixture
def engine(tmp_index_file):
    """创建测试引擎"""
    return create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)


# ==================== 模块级别测试 ====================

class TestModuleExports:
    """测试模块导出"""
    
    def test_version(self):
        """测试版本号"""
        assert __version__ is not None
        assert isinstance(__version__, str)
        # 版本号应该符合 x.y.z 格式
        parts = __version__.split(".")
        assert len(parts) >= 2
    
    def test_aliases(self):
        """测试类别名"""
        # SearchEngine 应该是 UnifiedEngine 的别名
        assert SearchEngine is UnifiedEngine
        
        # SearchResult 应该是 ResultHandle 的别名
        assert SearchResult is ResultHandle
    
    def test_all_exports_available(self):
        """测试所有导出都可用"""
        assert create_engine is not None
        assert UnifiedEngine is not None
        assert ResultHandle is not None
        assert FuzzyConfig is not None
        assert SearchEngine is not None
        assert SearchResult is not None


# ==================== create_engine 函数测试 ====================

class TestCreateEngineFunction:
    """测试 create_engine 函数所有参数"""
    
    def test_default_parameters(self, tmp_index_file):
        """测试默认参数"""
        engine = create_engine(tmp_index_file)
        assert engine is not None
    
    def test_memory_only_mode(self):
        """测试空 index_file（纯内存模式）"""
        engine = create_engine("")
        assert engine.is_memory_only()
    
    def test_max_chinese_length(self, tmp_index_file):
        """测试 max_chinese_length 参数"""
        engine = create_engine(tmp_index_file, max_chinese_length=2, drop_if_exists=True)
        
        engine.add_document(1, {"content": "中华人民共和国"})
        engine.flush()
        
        # 长度 2 的 n-gram
        result = engine.search("中华")
        assert 1 in result.to_list()
    
    def test_min_term_length(self, tmp_index_file):
        """测试 min_term_length 参数"""
        engine = create_engine(tmp_index_file, min_term_length=4, drop_if_exists=True)
        
        engine.add_document(1, {"content": "ab abc abcd abcde"})
        engine.flush()
        
        # 只有长度 >= 4 的词能被索引
        result = engine.search("abcd")
        assert 1 in result.to_list()
        
        result = engine.search("abc")
        assert 1 not in result.to_list()
    
    def test_fuzzy_threshold(self, tmp_index_file):
        """测试 fuzzy_threshold 参数"""
        engine = create_engine(tmp_index_file, fuzzy_threshold=0.5, drop_if_exists=True)
        
        config = engine.get_fuzzy_config()
        assert config["threshold"] == 0.5
    
    def test_fuzzy_max_distance(self, tmp_index_file):
        """测试 fuzzy_max_distance 参数"""
        engine = create_engine(tmp_index_file, fuzzy_max_distance=3, drop_if_exists=True)
        
        config = engine.get_fuzzy_config()
        assert config["max_distance"] == 3.0
    
    def test_track_doc_terms(self, tmp_index_file):
        """测试 track_doc_terms 参数"""
        engine = create_engine(tmp_index_file, track_doc_terms=True, drop_if_exists=True)
        
        engine.add_document(1, {"content": "track test"})
        
        assert engine.doc_terms_size() > 0
        
        stats = engine.stats()
        assert stats["track_doc_terms"] == 1.0
    
    def test_drop_if_exists(self, tmp_index_file):
        """测试 drop_if_exists 参数"""
        # 创建第一个引擎并写入数据
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "first data"})
        engine1.flush()
        del engine1
        
        # 使用 drop_if_exists=True 重新创建
        engine2 = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 旧数据应该被清除
        result = engine2.search("first")
        assert result.total_hits == 0
    
    def test_lazy_load_true(self, tmp_index_file):
        """测试 lazy_load=True"""
        engine = create_engine(tmp_index_file, lazy_load=True, drop_if_exists=True)
        assert engine.is_lazy_load()
    
    def test_lazy_load_false(self, tmp_index_file):
        """测试 lazy_load=False"""
        engine = create_engine(tmp_index_file, lazy_load=False, drop_if_exists=True)
        assert not engine.is_lazy_load()
    
    def test_cache_size(self, tmp_index_file):
        """测试 cache_size 参数"""
        engine = create_engine(tmp_index_file, cache_size=500, drop_if_exists=True, lazy_load=True)
        # cache_size 主要影响懒加载模式的 LRU 缓存
        assert engine is not None


# ==================== UnifiedEngine 方法完整测试 ====================

class TestUnifiedEngineDocumentOperations:
    """测试 UnifiedEngine 文档操作"""
    
    def test_add_document(self, engine):
        """测试 add_document"""
        engine.add_document(1, {"title": "Test", "content": "Content"})
        engine.flush()
        
        result = engine.search("test")
        assert 1 in result.to_list()
    
    def test_add_documents(self, engine):
        """测试 add_documents 批量添加"""
        docs = [
            (1, {"content": "first"}),
            (2, {"content": "second"}),
            (3, {"content": "third"}),
        ]
        count = engine.add_documents(docs)
        
        assert count == 3
        
        engine.flush()
        assert engine.search("first").total_hits == 1
        assert engine.search("second").total_hits == 1
    
    def test_update_document(self, engine):
        """测试 update_document"""
        engine.add_document(1, {"content": "original"})
        engine.flush()
        
        engine.update_document(1, {"content": "updated"})
        engine.flush()
        
        assert 1 not in engine.search("original").to_list()
        assert 1 in engine.search("updated").to_list()
    
    def test_remove_document(self, engine):
        """测试 remove_document"""
        engine.add_document(1, {"content": "removable"})
        engine.flush()
        
        engine.remove_document(1)
        engine.flush()
        
        assert 1 not in engine.search("removable").to_list()
    
    def test_remove_documents(self, engine):
        """测试 remove_documents 批量删除"""
        for i in range(10):
            engine.add_document(i, {"content": "bulk remove test"})
        engine.flush()
        
        engine.remove_documents([0, 1, 2, 3, 4])
        engine.flush()
        
        result = engine.search("bulk")
        assert result.total_hits == 5


class TestUnifiedEngineSearchOperations:
    """测试 UnifiedEngine 搜索操作"""
    
    def test_search(self, engine):
        """测试 search"""
        engine.add_document(1, {"content": "searchable content"})
        engine.flush()
        
        result = engine.search("searchable")
        assert isinstance(result, ResultHandle)
        assert result.total_hits == 1
    
    def test_fuzzy_search(self, engine):
        """测试 fuzzy_search"""
        engine.add_document(1, {"content": "fuzzy matching"})
        engine.flush()
        
        result = engine.fuzzy_search("fuzzi", min_results=0)
        assert isinstance(result, ResultHandle)
    
    def test_fuzzy_search_default_min_results(self, engine):
        """测试 fuzzy_search 默认 min_results=5"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        # 默认 min_results=5，少于 5 个结果时会使用模糊搜索
        result = engine.fuzzy_search("test")
        assert result is not None
    
    def test_search_batch(self, engine):
        """测试 search_batch"""
        engine.add_document(1, {"content": "apple"})
        engine.add_document(2, {"content": "banana"})
        engine.flush()
        
        results = engine.search_batch(["apple", "banana", "cherry"])
        
        assert len(results) == 3
        assert results[0].total_hits == 1
        assert results[1].total_hits == 1
        assert results[2].total_hits == 0
    
    def test_search_and(self, engine):
        """测试 search_and"""
        engine.add_document(1, {"content": "apple banana"})
        engine.add_document(2, {"content": "apple"})
        engine.flush()
        
        result = engine.search_and(["apple", "banana"])
        
        assert result.total_hits == 1
        assert 1 in result.to_list()
        assert 2 not in result.to_list()
    
    def test_search_or(self, engine):
        """测试 search_or"""
        engine.add_document(1, {"content": "apple"})
        engine.add_document(2, {"content": "banana"})
        engine.flush()
        
        result = engine.search_or(["apple", "banana"])
        
        assert result.total_hits == 2
    
    def test_filter_by_ids(self, engine):
        """测试 filter_by_ids"""
        for i in range(5):
            engine.add_document(i, {"content": "filter test"})
        engine.flush()
        
        result = engine.search("filter")
        filtered = engine.filter_by_ids(result, [0, 2, 4])
        
        assert filtered.total_hits == 3
        assert set(filtered.to_list()) == {0, 2, 4}
    
    def test_exclude_ids(self, engine):
        """测试 exclude_ids"""
        for i in range(5):
            engine.add_document(i, {"content": "exclude test"})
        engine.flush()
        
        result = engine.search("exclude")
        excluded = engine.exclude_ids(result, [1, 3])
        
        assert excluded.total_hits == 3
        assert set(excluded.to_list()) == {0, 2, 4}


class TestUnifiedEngineConfigOperations:
    """测试 UnifiedEngine 配置操作"""
    
    def test_set_fuzzy_config(self, engine):
        """测试 set_fuzzy_config"""
        engine.set_fuzzy_config(threshold=0.8, max_distance=3, max_candidates=30)
        
        config = engine.get_fuzzy_config()
        assert config["threshold"] == 0.8
        assert config["max_distance"] == 3.0
        assert config["max_candidates"] == 30.0
    
    def test_set_fuzzy_config_default_params(self, engine):
        """测试 set_fuzzy_config 默认参数"""
        engine.set_fuzzy_config()  # 使用所有默认值
        
        config = engine.get_fuzzy_config()
        assert config["threshold"] == 0.7
        assert config["max_distance"] == 2.0
        assert config["max_candidates"] == 20.0
    
    def test_get_fuzzy_config(self, engine):
        """测试 get_fuzzy_config"""
        config = engine.get_fuzzy_config()
        
        assert isinstance(config, dict)
        assert "threshold" in config
        assert "max_distance" in config
        assert "max_candidates" in config


class TestUnifiedEnginePersistenceOperations:
    """测试 UnifiedEngine 持久化操作"""
    
    def test_flush(self, engine):
        """测试 flush"""
        engine.add_document(1, {"content": "flush test"})
        
        count = engine.flush()
        assert count >= 0
    
    def test_save(self, engine):
        """测试 save（同 flush）"""
        engine.add_document(1, {"content": "save test"})
        
        count = engine.save()
        assert count >= 0
        
        # save 后数据应该被持久化
        result = engine.search("save")
        assert 1 in result.to_list()
    
    def test_load(self, engine):
        """测试 load"""
        engine.add_document(1, {"content": "load test"})
        engine.flush()
        
        # load 主要清除缓存
        engine.load()
        
        # 数据应该仍然可搜索
        result = engine.search("load")
        assert 1 in result.to_list()
    
    def test_preload(self, tmp_index_file):
        """测试 preload"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=True)
        for i in range(100):
            engine1.add_document(i, {"content": f"preload test {i}"})
        engine1.flush()
        del engine1
        
        engine2 = create_engine(tmp_index_file, lazy_load=True)
        
        preloaded = engine2.preload()
        assert preloaded >= 0
    
    def test_compact(self, engine):
        """测试 compact"""
        for i in range(50):
            engine.add_document(i, {"content": f"compact test {i}"})
        engine.flush()
        
        engine.compact()
        
        # compact 后数据应该仍然存在
        result = engine.search("compact")
        assert result.total_hits == 50


class TestUnifiedEngineCacheOperations:
    """测试 UnifiedEngine 缓存操作"""
    
    def test_clear_cache(self, engine):
        """测试 clear_cache"""
        engine.add_document(1, {"content": "cache test"})
        engine.flush()
        
        # 搜索以填充缓存
        engine.search("cache")
        
        # 清除缓存
        engine.clear_cache()
        
        # 再次搜索应该正常工作
        result = engine.search("cache")
        assert 1 in result.to_list()
    
    def test_clear_doc_terms(self, tmp_index_file):
        """测试 clear_doc_terms"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)
        
        engine.add_document(1, {"content": "doc terms test"})
        
        assert engine.doc_terms_size() > 0
        
        engine.clear_doc_terms()
        
        assert engine.doc_terms_size() == 0


class TestUnifiedEngineStatisticsOperations:
    """测试 UnifiedEngine 统计操作"""
    
    def test_is_memory_only(self, tmp_index_file):
        """测试 is_memory_only"""
        # 文件模式
        engine_file = create_engine(tmp_index_file, drop_if_exists=True)
        assert not engine_file.is_memory_only()
        
        # 内存模式
        engine_memory = create_engine("")
        assert engine_memory.is_memory_only()
    
    def test_term_count(self, engine):
        """测试 term_count"""
        initial = engine.term_count()
        
        engine.add_document(1, {"content": "unique1 unique2"})
        
        after_add = engine.term_count()
        assert after_add >= initial
    
    def test_buffer_size(self, engine):
        """测试 buffer_size"""
        initial = engine.buffer_size()
        
        engine.add_document(1, {"content": "buffer size test"})
        
        after_add = engine.buffer_size()
        assert after_add > initial
        
        engine.flush()
        
        after_flush = engine.buffer_size()
        assert after_flush == 0
    
    def test_doc_terms_size(self, tmp_index_file):
        """测试 doc_terms_size"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)
        
        initial = engine.doc_terms_size()
        assert initial == 0
        
        engine.add_document(1, {"content": "doc terms"})
        
        after_add = engine.doc_terms_size()
        assert after_add > 0
    
    def test_page_count(self, engine):
        """测试 page_count"""
        count = engine.page_count()
        assert count >= 0
        
        # page_count 应该等于 segment_count
        assert count == engine.segment_count()
    
    def test_segment_count(self, engine):
        """测试 segment_count"""
        initial = engine.segment_count()
        
        engine.add_document(1, {"content": "segment test"})
        engine.flush()
        
        after_flush = engine.segment_count()
        assert after_flush >= initial
    
    def test_memtable_size(self, engine):
        """测试 memtable_size"""
        size = engine.memtable_size()
        assert size >= 0
    
    def test_stats(self, engine):
        """测试 stats"""
        engine.add_document(1, {"content": "stats test"})
        engine.flush()
        engine.search("stats")
        
        stats = engine.stats()
        
        assert isinstance(stats, dict)
        
        # 检查所有预期的统计项
        expected_keys = [
            "search_count",
            "fuzzy_search_count",
            "cache_hits",
            "cache_hit_rate",
            "avg_search_us",
            "result_cache_size",
            "buffer_size",
            "term_count",
            "deleted_count",
            "memory_only",
            "lazy_load",
            "track_doc_terms",
        ]
        
        for key in expected_keys:
            assert key in stats, f"Missing stats key: {key}"
    
    def test_deleted_count(self, engine):
        """测试 deleted_count"""
        for i in range(10):
            engine.add_document(i, {"content": f"delete count test {i}"})
        engine.flush()
        
        assert engine.deleted_count() == 0
        
        engine.remove_document(0)
        engine.remove_document(1)
        
        assert engine.deleted_count() == 2
    
    def test_is_lazy_load(self, tmp_index_file):
        """测试 is_lazy_load"""
        engine_lazy = create_engine(tmp_index_file, lazy_load=True, drop_if_exists=True)
        assert engine_lazy.is_lazy_load()
        
        engine_non_lazy = create_engine(tmp_index_file, lazy_load=False, drop_if_exists=True)
        assert not engine_non_lazy.is_lazy_load()
    
    def test_warmup_terms(self, tmp_index_file):
        """测试 warmup_terms"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "warmup terms test"})
        engine1.flush()
        del engine1
        
        engine2 = create_engine(tmp_index_file, lazy_load=True)
        
        warmed = engine2.warmup_terms(["warmup", "terms", "test"])
        assert warmed >= 0
    
    def test_clear_lru_cache(self, tmp_index_file):
        """测试 clear_lru_cache"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=True)
        engine.add_document(1, {"content": "lru cache test"})
        engine.flush()
        
        # 搜索以填充 LRU 缓存
        engine.search("lru")
        
        # 清除 LRU 缓存
        engine.clear_lru_cache()
        
        # 再次搜索应该正常
        result = engine.search("lru")
        assert 1 in result.to_list()
    
    def test_repr(self, tmp_index_file):
        """测试 __repr__"""
        # 内存模式
        engine_memory = create_engine("")
        repr_memory = repr(engine_memory)
        assert "UnifiedEngine" in repr_memory
        assert "memory_only" in repr_memory
        
        # 懒加载模式
        engine_lazy = create_engine(tmp_index_file, lazy_load=True, drop_if_exists=True)
        repr_lazy = repr(engine_lazy)
        assert "UnifiedEngine" in repr_lazy
        assert "lazy_load" in repr_lazy
        
        # 普通模式
        engine_normal = create_engine(tmp_index_file, lazy_load=False, drop_if_exists=True)
        repr_normal = repr(engine_normal)
        assert "UnifiedEngine" in repr_normal


# ==================== ResultHandle 方法完整测试 ====================

class TestResultHandleProperties:
    """测试 ResultHandle 属性"""
    
    def test_total_hits_property(self, engine):
        """测试 total_hits 属性（getter）"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        result = engine.search("test")
        
        # total_hits 是一个 property (getter)
        assert result.total_hits == 1
    
    def test_elapsed_ns_property(self, engine):
        """测试 elapsed_ns 属性（getter）"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        result = engine.search("test")
        
        assert result.elapsed_ns >= 0
    
    def test_fuzzy_used_property(self, engine):
        """测试 fuzzy_used 属性（getter）"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        # 精确搜索
        exact_result = engine.search("test")
        assert exact_result.fuzzy_used == False
        
        # 模糊搜索
        fuzzy_result = engine.fuzzy_search("tset", min_results=0)
        # fuzzy_used 取决于是否真的使用了模糊搜索


class TestResultHandleMethods:
    """测试 ResultHandle 方法"""
    
    def test_elapsed_ms(self, engine):
        """测试 elapsed_ms"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        result = engine.search("test")
        
        elapsed_ms = result.elapsed_ms()
        assert elapsed_ms >= 0
        assert elapsed_ms == result.elapsed_ns / 1_000_000.0
    
    def test_elapsed_us(self, engine):
        """测试 elapsed_us"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        result = engine.search("test")
        
        elapsed_us = result.elapsed_us()
        assert elapsed_us >= 0
        assert elapsed_us == result.elapsed_ns / 1_000.0
    
    def test_contains(self, engine):
        """测试 contains"""
        engine.add_document(1, {"content": "test"})
        engine.add_document(2, {"content": "other"})
        engine.flush()
        
        result = engine.search("test")
        
        assert result.contains(1)
        assert not result.contains(2)
        assert not result.contains(999)
    
    def test_is_empty(self, engine):
        """测试 is_empty"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        result_found = engine.search("test")
        assert not result_found.is_empty()
        
        result_empty = engine.search("nonexistent")
        assert result_empty.is_empty()
    
    def test_top(self, engine):
        """测试 top"""
        for i in range(100):
            engine.add_document(i, {"content": "top test"})
        engine.flush()
        
        result = engine.search("top")
        
        # 默认 top(100)
        top_100 = result.top()
        assert len(top_100) == 100
        
        # 指定数量
        top_10 = result.top(10)
        assert len(top_10) == 10
        
        # 超过总数
        top_200 = result.top(200)
        assert len(top_200) == 100
    
    def test_to_list(self, engine):
        """测试 to_list"""
        for i in range(5):
            engine.add_document(i, {"content": "list test"})
        engine.flush()
        
        result = engine.search("list")
        lst = result.to_list()
        
        assert isinstance(lst, list)
        assert len(lst) == 5
        assert set(lst) == {0, 1, 2, 3, 4}
    
    def test_to_numpy(self, engine):
        """测试 to_numpy"""
        np = pytest.importorskip("numpy")
        
        for i in range(5):
            engine.add_document(i, {"content": "numpy test"})
        engine.flush()
        
        result = engine.search("numpy")
        arr = result.to_numpy()
        
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint32
        assert len(arr) == 5
    
    def test_page(self, engine):
        """测试 page"""
        for i in range(100):
            engine.add_document(i, {"content": "page test"})
        engine.flush()
        
        result = engine.search("page")
        
        # 第一页
        page_0 = result.page(0, 10)
        assert len(page_0) == 10
        
        # 中间页
        page_5 = result.page(50, 10)
        assert len(page_5) == 10
        
        # 最后一页（可能不满）
        page_last = result.page(95, 10)
        assert len(page_last) == 5
        
        # 超出范围
        page_out = result.page(200, 10)
        assert len(page_out) == 0
    
    def test_intersect(self, engine):
        """测试 intersect"""
        engine.add_document(1, {"content": "apple banana"})
        engine.add_document(2, {"content": "apple"})
        engine.add_document(3, {"content": "banana"})
        engine.flush()
        
        r1 = engine.search("apple")
        r2 = engine.search("banana")
        
        intersection = r1.intersect(r2)
        
        assert intersection.total_hits == 1
        assert 1 in intersection.to_list()
    
    def test_union(self, engine):
        """测试 union"""
        engine.add_document(1, {"content": "apple"})
        engine.add_document(2, {"content": "banana"})
        engine.flush()
        
        r1 = engine.search("apple")
        r2 = engine.search("banana")
        
        union = r1.union(r2)
        
        assert union.total_hits == 2
        assert set(union.to_list()) == {1, 2}
    
    def test_difference(self, engine):
        """测试 difference"""
        engine.add_document(1, {"content": "apple banana"})
        engine.add_document(2, {"content": "apple"})
        engine.flush()
        
        r1 = engine.search("apple")
        r2 = engine.search("banana")
        
        diff = r1.difference(r2)
        
        assert diff.total_hits == 1
        assert 2 in diff.to_list()
    
    def test_len(self, engine):
        """测试 __len__"""
        for i in range(10):
            engine.add_document(i, {"content": "len test"})
        engine.flush()
        
        result = engine.search("len")
        
        assert len(result) == 10
    
    def test_repr(self, engine):
        """测试 __repr__"""
        engine.add_document(1, {"content": "repr test"})
        engine.flush()
        
        result = engine.search("repr")
        repr_str = repr(result)
        
        assert "ResultHandle" in repr_str
        assert "hits=" in repr_str
        assert "query=" in repr_str
        assert "repr" in repr_str
        assert "elapsed=" in repr_str


# ==================== FuzzyConfig 类完整测试 ====================

class TestFuzzyConfig:
    """测试 FuzzyConfig 类"""
    
    def test_create_default(self):
        """测试默认创建"""
        config = FuzzyConfig()
        
        assert config.threshold == 0.7
        assert config.max_distance == 2
        assert config.max_candidates == 20
    
    def test_create_with_params(self):
        """测试带参数创建"""
        config = FuzzyConfig(threshold=0.8, max_distance=3, max_candidates=30)
        
        assert config.threshold == 0.8
        assert config.max_distance == 3
        assert config.max_candidates == 30
    
    def test_threshold_getter_setter(self):
        """测试 threshold 属性"""
        config = FuzzyConfig()
        
        # getter
        assert config.threshold == 0.7
        
        # setter
        config.threshold = 0.9
        assert config.threshold == 0.9
    
    def test_max_distance_getter_setter(self):
        """测试 max_distance 属性"""
        config = FuzzyConfig()
        
        # getter
        assert config.max_distance == 2
        
        # setter
        config.max_distance = 5
        assert config.max_distance == 5
    
    def test_max_candidates_getter_setter(self):
        """测试 max_candidates 属性"""
        config = FuzzyConfig()
        
        # getter
        assert config.max_candidates == 20
        
        # setter
        config.max_candidates = 50
        assert config.max_candidates == 50


# ==================== 别名测试 ====================

class TestAliases:
    """测试类型别名"""
    
    def test_search_engine_alias(self, tmp_index_file):
        """测试 SearchEngine 别名"""
        # SearchEngine 应该和 UnifiedEngine 一样工作
        engine = SearchEngine(tmp_index_file, drop_if_exists=True)
        
        engine.add_document(1, {"content": "alias test"})
        engine.flush()
        
        result = engine.search("alias")
        assert 1 in result.to_list()
    
    def test_search_result_alias(self, tmp_index_file):
        """测试 SearchResult 别名"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        engine.add_document(1, {"content": "result alias"})
        engine.flush()
        
        result = engine.search("result")
        
        # result 应该是 SearchResult (ResultHandle) 类型
        assert isinstance(result, SearchResult)
        assert isinstance(result, ResultHandle)


# ==================== 数据导入 API 测试 ====================

class TestDataImportAPIs:
    """测试数据导入 API"""
    
    def test_from_pandas(self, tmp_index_file):
        """测试 from_pandas 方法"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['Pandas A', 'Pandas B', 'Pandas C'],
            'content': ['Content 1', 'Content 2', 'Content 3']
        })
        
        count = engine.from_pandas(df, id_column='id')
        
        assert count == 3
        assert engine.search("pandas").total_hits == 3
    
    def test_from_pandas_with_text_columns(self, tmp_index_file):
        """测试 from_pandas 指定 text_columns"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2],
            'include': ['Include Text', 'Include Text 2'],
            'exclude': ['Exclude Text', 'Exclude Text 2']
        })
        
        count = engine.from_pandas(df, id_column='id', text_columns=['include'])
        
        assert count == 2
        assert engine.search("include").total_hits == 2
        assert engine.search("exclude").total_hits == 0
    
    def test_from_polars(self, tmp_index_file):
        """测试 from_polars 方法"""
        pl = pytest.importorskip("polars")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pl.DataFrame({
            'id': [1, 2, 3],
            'title': ['Polars A', 'Polars B', 'Polars C']
        })
        
        count = engine.from_polars(df, id_column='id')
        
        assert count == 3
        assert engine.search("polars").total_hits == 3
    
    def test_from_polars_with_text_columns(self, tmp_index_file):
        """测试 from_polars 指定 text_columns"""
        pl = pytest.importorskip("polars")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pl.DataFrame({
            'id': [1, 2],
            'visible': ['Visible A', 'Visible B'],
            'hidden': ['Hidden A', 'Hidden B']
        })
        
        count = engine.from_polars(df, id_column='id', text_columns=['visible'])
        
        assert count == 2
        assert engine.search("visible").total_hits == 2
        assert engine.search("hidden").total_hits == 0
    
    def test_from_arrow(self, tmp_index_file):
        """测试 from_arrow 方法"""
        pa = pytest.importorskip("pyarrow")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        table = pa.Table.from_pydict({
            'id': [1, 2, 3],
            'title': ['Arrow A', 'Arrow B', 'Arrow C']
        })
        
        count = engine.from_arrow(table, id_column='id')
        
        assert count == 3
        assert engine.search("arrow").total_hits == 3
    
    def test_from_arrow_with_text_columns(self, tmp_index_file):
        """测试 from_arrow 指定 text_columns"""
        pa = pytest.importorskip("pyarrow")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        table = pa.Table.from_pydict({
            'id': [1, 2],
            'public': ['Public A', 'Public B'],
            'private': ['Private A', 'Private B']
        })
        
        count = engine.from_arrow(table, id_column='id', text_columns=['public'])
        
        assert count == 2
        assert engine.search("public").total_hits == 2
        assert engine.search("private").total_hits == 0
    
    def test_from_parquet(self, tmp_index_file, tmp_path):
        """测试 from_parquet 方法"""
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        table = pa.Table.from_pydict({
            'id': [1, 2, 3],
            'title': ['Parquet A', 'Parquet B', 'Parquet C']
        })
        
        parquet_path = tmp_path / "test.parquet"
        pq.write_table(table, parquet_path)
        
        count = engine.from_parquet(parquet_path, id_column='id')
        
        assert count == 3
        assert engine.search("parquet").total_hits == 3
    
    def test_from_parquet_with_text_columns(self, tmp_index_file, tmp_path):
        """测试 from_parquet 指定 text_columns"""
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        table = pa.Table.from_pydict({
            'id': [1, 2],
            'index': ['Index A', 'Index B'],
            'skip': ['Skip A', 'Skip B']
        })
        
        parquet_path = tmp_path / "test.parquet"
        pq.write_table(table, parquet_path)
        
        count = engine.from_parquet(parquet_path, id_column='id', text_columns=['index'])
        
        assert count == 2
        assert engine.search("index").total_hits == 2
        assert engine.search("skip").total_hits == 0
    
    def test_from_csv(self, tmp_index_file, tmp_path):
        """测试 from_csv 方法"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['CSV A', 'CSV B', 'CSV C']
        })
        
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        count = engine.from_csv(csv_path, id_column='id')
        
        assert count == 3
        assert engine.search("csv").total_hits == 3
    
    def test_from_csv_with_text_columns(self, tmp_index_file, tmp_path):
        """测试 from_csv 指定 text_columns"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2],
            'search': ['Search A', 'Search B'],
            'nosearch': ['NoSearch A', 'NoSearch B']
        })
        
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        count = engine.from_csv(csv_path, id_column='id', text_columns=['search'])
        
        assert count == 2
        assert engine.search("search").total_hits == 2
        assert engine.search("nosearch").total_hits == 0
    
    def test_from_csv_with_options(self, tmp_index_file, tmp_path):
        """测试 from_csv CSV 选项参数"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2],
            'title': ['Options A', 'Options B']
        })
        
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False, sep=';')
        
        count = engine.from_csv(csv_path, id_column='id', sep=';')
        
        assert count == 2
        assert engine.search("options").total_hits == 2
    
    def test_from_json(self, tmp_index_file, tmp_path):
        """测试 from_json 方法"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'title': ['JSON A', 'JSON B', 'JSON C']
        })
        
        json_path = tmp_path / "test.json"
        df.to_json(json_path, orient='records')
        
        count = engine.from_json(json_path, id_column='id')
        
        assert count == 3
        assert engine.search("json").total_hits == 3
    
    def test_from_json_with_text_columns(self, tmp_index_file, tmp_path):
        """测试 from_json 指定 text_columns"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2],
            'show': ['Show A', 'Show B'],
            'hide': ['Hide A', 'Hide B']
        })
        
        json_path = tmp_path / "test.json"
        df.to_json(json_path, orient='records')
        
        count = engine.from_json(json_path, id_column='id', text_columns=['show'])
        
        assert count == 2
        assert engine.search("show").total_hits == 2
        assert engine.search("hide").total_hits == 0
    
    def test_from_json_lines(self, tmp_index_file, tmp_path):
        """测试 from_json JSON Lines 格式"""
        pd = pytest.importorskip("pandas")
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        df = pd.DataFrame({
            'id': [1, 2],
            'title': ['JSONL A', 'JSONL B']
        })
        
        jsonl_path = tmp_path / "test.jsonl"
        df.to_json(jsonl_path, orient='records', lines=True)
        
        count = engine.from_json(jsonl_path, id_column='id', lines=True)
        
        assert count == 2
        assert engine.search("jsonl").total_hits == 2
    
    def test_from_dict(self, tmp_index_file):
        """测试 from_dict 方法"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        data = [
            {'id': 1, 'title': 'Dict A', 'content': 'Content A'},
            {'id': 2, 'title': 'Dict B', 'content': 'Content B'},
            {'id': 3, 'title': 'Dict C', 'content': 'Content C'},
        ]
        
        count = engine.from_dict(data, id_column='id')
        
        assert count == 3
        assert engine.search("dict").total_hits == 3
    
    def test_from_dict_with_text_columns(self, tmp_index_file):
        """测试 from_dict 指定 text_columns"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        data = [
            {'id': 1, 'yes': 'Yes A', 'no': 'No A'},
            {'id': 2, 'yes': 'Yes B', 'no': 'No B'},
        ]
        
        count = engine.from_dict(data, id_column='id', text_columns=['yes'])
        
        assert count == 2
        assert engine.search("yes").total_hits == 2
        assert engine.search("no").total_hits == 0
    
    def test_from_dict_empty(self, tmp_index_file):
        """测试 from_dict 空列表"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        count = engine.from_dict([], id_column='id')
        
        assert count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


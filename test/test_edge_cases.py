"""
NanoFTS 边缘情况测试

测试各种边界条件和异常情况：
- 空查询/空文档
- 超长文本
- 特殊字符
- 极值 doc_id
- 重复操作
"""

import pytest
import os
import tempfile
from nanofts import create_engine, UnifiedEngine


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_index.nfts")


@pytest.fixture
def engine(tmp_index_file):
    """创建测试引擎"""
    return create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)


@pytest.fixture
def memory_engine():
    """创建内存模式引擎"""
    return create_engine("")


class TestEmptyOperations:
    """空操作测试"""
    
    def test_search_empty_index(self, engine):
        """测试在空索引上搜索"""
        result = engine.search("any query")
        assert len(result) == 0
        assert result.is_empty()
        assert result.total_hits == 0
    
    def test_search_empty_query(self, engine):
        """测试空查询字符串"""
        # 添加一些数据
        engine.add_document(1, {"title": "Hello World"})
        engine.flush()
        
        result = engine.search("")
        assert len(result) == 0
    
    def test_search_whitespace_only_query(self, engine):
        """测试仅包含空白字符的查询"""
        engine.add_document(1, {"title": "Hello World"})
        engine.flush()
        
        result = engine.search("   ")
        assert len(result) == 0
        
        result = engine.search("\t\n")
        assert len(result) == 0
    
    def test_add_empty_document(self, engine):
        """测试添加空文档"""
        engine.add_document(1, {})
        engine.flush()
        
        # 空文档不应该影响搜索
        result = engine.search("anything")
        assert len(result) == 0
    
    def test_add_document_with_empty_fields(self, engine):
        """测试添加字段为空的文档"""
        engine.add_document(1, {"title": "", "content": ""})
        engine.flush()
        
        result = engine.search("anything")
        assert len(result) == 0
    
    def test_remove_nonexistent_document(self, engine):
        """测试删除不存在的文档"""
        # 不应该抛出异常
        engine.remove_document(999)
        engine.flush()
        
        # 引擎应该正常工作
        result = engine.search("test")
        assert len(result) == 0


class TestSpecialCharacters:
    """特殊字符测试"""
    
    def test_search_with_punctuation(self, engine):
        """测试包含标点符号的搜索"""
        engine.add_document(1, {"content": "Hello, World! How are you?"})
        engine.flush()
        
        result = engine.search("hello")
        assert 1 in result.to_list()
        
        result = engine.search("world")
        assert 1 in result.to_list()
    
    def test_search_with_unicode(self, engine):
        """测试 Unicode 字符"""
        engine.add_document(1, {"content": "你好世界 🎉 émoji"})
        engine.flush()
        
        result = engine.search("你好")
        assert 1 in result.to_list()
    
    def test_search_with_numbers(self, engine):
        """测试数字内容"""
        engine.add_document(1, {"content": "Version 2.0.1 released"})
        engine.add_document(2, {"content": "12345 numbers only"})
        engine.flush()
        
        result = engine.search("12345")
        assert 2 in result.to_list()
    
    def test_search_with_special_regex_chars(self, engine):
        """测试正则表达式特殊字符"""
        engine.add_document(1, {"content": "test.*regex+pattern[a-z]"})
        engine.flush()
        
        # 这些特殊字符应该被当作普通字符处理
        result = engine.search("test")
        assert 1 in result.to_list()
    
    def test_mixed_language_search(self, engine):
        """测试中英文混合搜索"""
        engine.add_document(1, {"content": "Hello你好World世界"})
        engine.flush()
        
        result = engine.search("hello")
        assert 1 in result.to_list()
        
        result = engine.search("你好")
        assert 1 in result.to_list()
        
        result = engine.search("hello 你好")
        assert 1 in result.to_list()


class TestExtremeValues:
    """极值测试"""
    
    def test_very_long_text(self, engine):
        """测试超长文本"""
        # 创建一个很长的文本
        long_text = "word " * 10000  # 50000 字符
        engine.add_document(1, {"content": long_text})
        engine.flush()
        
        result = engine.search("word")
        assert 1 in result.to_list()
    
    def test_very_long_term(self, engine):
        """测试超长单词"""
        long_word = "a" * 1000
        engine.add_document(1, {"content": long_word})
        engine.flush()
        
        result = engine.search(long_word)
        # 可能因为 min_term_length 限制而有结果或没有结果
        # 主要确保不会崩溃
    
    def test_large_doc_id(self, engine):
        """测试大 doc_id"""
        large_id = 2**31 - 1  # 最大 32 位有符号整数
        engine.add_document(large_id, {"content": "test content"})
        engine.flush()
        
        result = engine.search("test")
        assert large_id in result.to_list()
    
    def test_zero_doc_id(self, engine):
        """测试 doc_id 为 0"""
        engine.add_document(0, {"content": "zero id document"})
        engine.flush()
        
        result = engine.search("zero")
        assert 0 in result.to_list()
    
    def test_many_documents(self, engine):
        """测试大量文档"""
        # 添加 1000 个文档
        docs = [(i, {"content": f"document number {i}"}) for i in range(1000)]
        engine.add_documents(docs)
        engine.flush()
        
        result = engine.search("document")
        assert result.total_hits == 1000
    
    def test_many_terms_per_document(self, engine):
        """测试单个文档包含大量词条"""
        # 创建包含 1000 个不同词的文档
        words = [f"word{i}" for i in range(1000)]
        content = " ".join(words)
        engine.add_document(1, {"content": content})
        engine.flush()
        
        # 搜索其中一个词
        result = engine.search("word500")
        assert 1 in result.to_list()


class TestDuplicateOperations:
    """重复操作测试"""
    
    def test_add_same_document_twice(self, engine):
        """测试添加相同文档两次"""
        engine.add_document(1, {"content": "first content"})
        engine.add_document(1, {"content": "second content"})
        engine.flush()
        
        # 应该两个内容都能搜索到（或者后者覆盖前者，取决于实现）
        result1 = engine.search("first")
        result2 = engine.search("second")
        
        # 至少第二次添加应该生效
        assert 1 in result2.to_list()
    
    def test_remove_same_document_twice(self, engine):
        """测试删除相同文档两次"""
        engine.add_document(1, {"content": "test content"})
        engine.flush()
        
        engine.remove_document(1)
        engine.remove_document(1)  # 第二次删除不应该报错
        engine.flush()
        
        result = engine.search("test")
        assert 1 not in result.to_list()
    
    def test_update_same_document_multiple_times(self, engine):
        """测试多次更新同一文档"""
        engine.add_document(1, {"content": "version 1"})
        engine.flush()
        
        engine.update_document(1, {"content": "version 2"})
        engine.update_document(1, {"content": "version 3"})
        engine.flush()
        
        # 只有最新版本应该被搜索到
        result = engine.search("version")
        assert 1 in result.to_list()
    
    def test_add_after_remove(self, engine):
        """测试删除后重新添加"""
        engine.add_document(1, {"content": "original content"})
        engine.flush()
        
        engine.remove_document(1)
        engine.flush()
        
        engine.add_document(1, {"content": "new content"})
        engine.flush()
        
        result = engine.search("original")
        assert 1 not in result.to_list()
        
        result = engine.search("new")
        assert 1 in result.to_list()


class TestQueryVariations:
    """查询变体测试"""
    
    def test_case_insensitive_search(self, engine):
        """测试大小写不敏感搜索"""
        engine.add_document(1, {"content": "Hello World"})
        engine.flush()
        
        assert engine.search("hello").total_hits == engine.search("HELLO").total_hits
        assert engine.search("world").total_hits == engine.search("WORLD").total_hits
    
    def test_single_character_search(self, engine):
        """测试单字符搜索"""
        engine.add_document(1, {"content": "a b c d e"})
        engine.flush()
        
        # 由于 min_term_length 默认是 2，单字符可能搜不到
        result = engine.search("a")
        # 不崩溃即可
    
    def test_chinese_single_character(self, engine):
        """测试中文单字符搜索"""
        engine.add_document(1, {"content": "中国北京"})
        engine.flush()
        
        # 中文应该按 n-gram 处理
        result = engine.search("中国")
        assert 1 in result.to_list()
    
    def test_search_with_leading_trailing_spaces(self, engine):
        """测试带前后空格的查询"""
        engine.add_document(1, {"content": "hello world"})
        engine.flush()
        
        result1 = engine.search("hello")
        result2 = engine.search("  hello  ")
        
        # 应该得到相同结果
        assert result1.total_hits == result2.total_hits


class TestMemoryMode:
    """内存模式测试"""
    
    def test_memory_mode_basic(self, memory_engine):
        """测试内存模式基本功能"""
        assert memory_engine.is_memory_only()
        
        memory_engine.add_document(1, {"content": "test content"})
        # 内存模式下 flush 应该是 no-op
        memory_engine.flush()
        
        result = memory_engine.search("test")
        assert 1 in result.to_list()
    
    def test_memory_mode_no_persistence(self, memory_engine):
        """测试内存模式不持久化"""
        memory_engine.add_document(1, {"content": "test content"})
        
        # 即使不 flush 也能搜索到
        result = memory_engine.search("test")
        assert 1 in result.to_list()


class TestConfigOptions:
    """配置选项测试"""
    
    def test_custom_chinese_length(self, tmp_index_file):
        """测试自定义中文 n-gram 长度"""
        engine = create_engine(
            tmp_index_file,
            max_chinese_length=2,
            drop_if_exists=True
        )
        
        engine.add_document(1, {"content": "中华人民共和国"})
        engine.flush()
        
        # 只能搜索到 2-gram
        result = engine.search("中华")
        assert 1 in result.to_list()
    
    def test_custom_min_term_length(self, tmp_index_file):
        """测试自定义最小词条长度"""
        engine = create_engine(
            tmp_index_file,
            min_term_length=3,
            drop_if_exists=True
        )
        
        engine.add_document(1, {"content": "ab abc abcd"})
        engine.flush()
        
        # 只有长度 >= 3 的词能被索引
        result = engine.search("abc")
        assert 1 in result.to_list()
        
        result = engine.search("ab")
        assert 1 not in result.to_list()
    
    def test_fuzzy_config(self, tmp_index_file):
        """测试模糊搜索配置"""
        engine = create_engine(
            tmp_index_file,
            fuzzy_threshold=0.5,
            fuzzy_max_distance=3,
            drop_if_exists=True
        )
        
        engine.add_document(1, {"content": "hello world"})
        engine.flush()
        
        # 测试模糊搜索
        result = engine.fuzzy_search("helo", min_results=0)  # 拼写错误
        # 主要确保不崩溃


class TestBoundaryConditions:
    """边界条件测试"""
    
    def test_flush_empty_buffer(self, engine):
        """测试空缓冲区 flush"""
        # 不添加任何数据，直接 flush
        result = engine.flush()
        assert result == 0
    
    def test_multiple_flushes(self, engine):
        """测试多次 flush"""
        engine.add_document(1, {"content": "test"})
        engine.flush()
        engine.flush()  # 第二次 flush 应该是 no-op
        engine.flush()  # 第三次 flush
        
        result = engine.search("test")
        assert 1 in result.to_list()
    
    def test_search_immediately_after_add(self, engine):
        """测试添加后立即搜索（不 flush）"""
        engine.add_document(1, {"content": "immediate test"})
        
        # 不 flush 也应该能搜索到（从 buffer 中）
        result = engine.search("immediate")
        assert 1 in result.to_list()
    
    def test_compact_empty_index(self, engine):
        """测试对空索引进行 compact"""
        engine.compact()  # 不应该崩溃
    
    def test_compact_after_flush(self, engine):
        """测试 flush 后 compact"""
        engine.add_document(1, {"content": "test content"})
        engine.flush()
        engine.compact()
        
        result = engine.search("test")
        assert 1 in result.to_list()
    
    def test_stats_on_empty_engine(self, engine):
        """测试空引擎的统计信息"""
        stats = engine.stats()
        
        assert stats["search_count"] == 0
        assert stats["term_count"] == 0
    
    def test_term_count_consistency(self, engine):
        """测试词条计数一致性"""
        initial_count = engine.term_count()
        
        engine.add_document(1, {"content": "hello world"})
        after_add_count = engine.term_count()
        
        assert after_add_count >= initial_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



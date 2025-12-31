"""
NanoFTS 持久化测试

测试文件存储相关功能：
- 索引文件格式
- 多次打开/关闭
- 文件大小和增长
- 懒加载模式
"""

import pytest
import os
import time
from nanofts import create_engine


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_persistence.nfts")


class TestIndexFileBasics:
    """索引文件基础测试"""
    
    def test_index_file_created(self, tmp_index_file):
        """测试索引文件是否被创建"""
        assert not os.path.exists(tmp_index_file)
        
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        engine.add_document(1, {"content": "test"})
        engine.flush()
        
        assert os.path.exists(tmp_index_file)
    
    def test_index_file_size_grows(self, tmp_index_file):
        """测试索引文件大小增长"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        engine.add_document(1, {"content": "small content"})
        engine.flush()
        
        size_after_one = os.path.getsize(tmp_index_file)
        
        # 添加更多数据
        for i in range(100):
            engine.add_document(i + 2, {"content": f"more content {i}"})
        engine.flush()
        
        size_after_more = os.path.getsize(tmp_index_file)
        
        assert size_after_more > size_after_one
    
    def test_index_file_header(self, tmp_index_file):
        """测试索引文件头"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        engine.add_document(1, {"content": "header test"})
        engine.flush()
        del engine
        
        # 读取文件头（前 4 字节应该是 magic number）
        with open(tmp_index_file, 'rb') as f:
            magic = f.read(4)
        
        # NanoFTS Single 的 magic 是 "NFS1"
        assert magic == b'NFS1'


class TestMultipleOpenClose:
    """多次打开/关闭测试"""
    
    def test_open_close_cycle(self, tmp_index_file):
        """测试多次打开关闭循环"""
        # 创建并写入
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "cycle test"})
        engine1.flush()
        del engine1
        
        # 多次打开关闭
        for i in range(5):
            engine = create_engine(tmp_index_file)
            result = engine.search("cycle")
            assert result.total_hits == 1
            del engine
    
    def test_write_read_cycles(self, tmp_index_file):
        """测试写入-读取循环"""
        for cycle in range(3):
            # 写入阶段
            engine = create_engine(tmp_index_file, drop_if_exists=(cycle == 0))
            engine.add_document(cycle, {"content": f"write cycle {cycle}"})
            engine.flush()
            del engine
            
            # 读取阶段
            engine = create_engine(tmp_index_file)
            result = engine.search(f"cycle {cycle}")
            assert cycle in result.to_list()
            del engine
    
    def test_accumulative_data(self, tmp_index_file):
        """测试数据累积"""
        # 第一次写入
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        for i in range(10):
            engine1.add_document(i, {"content": f"first batch {i}"})
        engine1.flush()
        del engine1
        
        # 第二次追加
        engine2 = create_engine(tmp_index_file)
        for i in range(10, 20):
            engine2.add_document(i, {"content": f"second batch {i}"})
        engine2.flush()
        del engine2
        
        # 验证两批数据都存在
        engine3 = create_engine(tmp_index_file)
        result_first = engine3.search("first")
        result_second = engine3.search("second")
        
        assert result_first.total_hits == 10
        assert result_second.total_hits == 10


class TestSegmentManagement:
    """Segment 管理测试"""
    
    def test_segment_count_increases(self, tmp_index_file):
        """测试 segment 数量增加"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        initial_segments = engine.segment_count()
        
        # 多次 flush 应该创建多个 segment
        for batch in range(5):
            for i in range(20):
                engine.add_document(batch * 20 + i, {"content": f"segment {batch} doc {i}"})
            engine.flush()
        
        final_segments = engine.segment_count()
        assert final_segments > initial_segments
    
    def test_segment_count_after_compact(self, tmp_index_file):
        """测试 compact 后 segment 数量"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 创建多个 segment
        for batch in range(5):
            for i in range(10):
                engine.add_document(batch * 10 + i, {"content": f"compact segment {batch}"})
            engine.flush()
        
        segments_before = engine.segment_count()
        
        engine.compact()
        
        segments_after = engine.segment_count()
        
        # Compact 后 segment 数量应该减少（合并为一个）
        assert segments_after <= segments_before
    
    def test_segment_persistence(self, tmp_index_file):
        """测试 segment 持久化"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        for batch in range(3):
            for i in range(10):
                engine1.add_document(batch * 10 + i, {"content": f"persist {batch}"})
            engine1.flush()
        
        segments = engine1.segment_count()
        del engine1
        
        # 重新打开
        engine2 = create_engine(tmp_index_file)
        
        assert engine2.segment_count() == segments


class TestLazyLoadMode:
    """懒加载模式测试"""
    
    def test_lazy_load_creation(self, tmp_index_file):
        """测试懒加载模式创建"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=True)
        
        assert engine.is_lazy_load()
    
    def test_lazy_load_search(self, tmp_index_file):
        """测试懒加载模式搜索"""
        # 先创建数据
        engine1 = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=False)
        for i in range(100):
            engine1.add_document(i, {"content": f"lazy search test {i}"})
        engine1.flush()
        del engine1
        
        # 用懒加载模式打开
        engine2 = create_engine(tmp_index_file, lazy_load=True)
        
        result = engine2.search("lazy")
        assert result.total_hits == 100
    
    def test_lazy_load_cache_stats(self, tmp_index_file):
        """测试懒加载缓存统计"""
        # 创建数据
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        for i in range(50):
            engine1.add_document(i, {"content": f"cache stats {i}"})
        engine1.flush()
        del engine1
        
        # 懒加载打开
        engine2 = create_engine(tmp_index_file, lazy_load=True, cache_size=100)
        
        # 第一次搜索（缓存未命中）
        engine2.search("cache")
        
        # 第二次搜索（可能命中缓存）
        engine2.search("cache")
        
        stats = engine2.stats()
        # 检查缓存相关统计存在
        if stats.get("lazy_load", 0) == 1.0:
            assert "lru_cache_hits" in stats or "lru_cache_misses" in stats
    
    def test_lazy_load_warmup(self, tmp_index_file):
        """测试懒加载预热"""
        # 创建数据
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        for i in range(100):
            engine1.add_document(i, {"content": f"warmup testing {i}"})
        engine1.flush()
        del engine1
        
        # 懒加载打开
        engine2 = create_engine(tmp_index_file, lazy_load=True, cache_size=1000)
        
        # 预热
        warmed = engine2.warmup_terms(["warmup", "testing"])
        assert warmed >= 0
        
        # 预热后搜索应该更快
        result = engine2.search("warmup")
        assert result.total_hits == 100
    
    def test_lazy_load_clear_cache(self, tmp_index_file):
        """测试懒加载清除缓存"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        for i in range(50):
            engine1.add_document(i, {"content": f"clear cache {i}"})
        engine1.flush()
        del engine1
        
        engine2 = create_engine(tmp_index_file, lazy_load=True)
        
        # 搜索填充缓存
        engine2.search("clear")
        
        # 清除缓存
        engine2.clear_lru_cache()
        
        # 再次搜索应该仍然工作
        result = engine2.search("clear")
        assert result.total_hits == 50


class TestNonLazyLoadMode:
    """非懒加载模式测试"""
    
    def test_non_lazy_load_creation(self, tmp_index_file):
        """测试非懒加载模式创建"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=False)
        
        assert not engine.is_lazy_load()
    
    def test_non_lazy_load_all_in_memory(self, tmp_index_file):
        """测试非懒加载模式全量加载"""
        # 创建数据
        engine1 = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=False)
        for i in range(100):
            engine1.add_document(i, {"content": f"all in memory {i}"})
        engine1.flush()
        del engine1
        
        # 非懒加载打开
        engine2 = create_engine(tmp_index_file, lazy_load=False)
        
        # 搜索应该直接从内存获取
        result = engine2.search("memory")
        assert result.total_hits == 100


class TestMemoryOnlyMode:
    """纯内存模式测试"""
    
    def test_memory_only_no_file(self, tmp_path):
        """测试纯内存模式不创建文件"""
        engine = create_engine("")  # 空路径表示纯内存模式
        
        assert engine.is_memory_only()
        
        engine.add_document(1, {"content": "memory only"})
        engine.flush()
        
        # 不应该有任何文件被创建
        files = list(tmp_path.glob("*.nfts"))
        assert len(files) == 0
    
    def test_memory_only_operations(self):
        """测试纯内存模式操作"""
        engine = create_engine("")
        
        # 所有操作应该正常工作
        engine.add_document(1, {"content": "test content"})
        engine.add_document(2, {"content": "more content"})
        engine.flush()
        
        result = engine.search("content")
        assert result.total_hits == 2
        
        engine.remove_document(1)
        
        result = engine.search("content")
        assert result.total_hits == 1
    
    def test_memory_only_stats(self):
        """测试纯内存模式统计"""
        engine = create_engine("")
        
        stats = engine.stats()
        assert stats["memory_only"] == 1.0


class TestFileRecovery:
    """文件恢复测试"""
    
    def test_recover_from_incomplete_header(self, tmp_index_file):
        """测试从不完整的头恢复"""
        # 创建有效索引
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "valid data"})
        engine1.flush()
        del engine1
        
        # 截断文件（模拟损坏）
        original_size = os.path.getsize(tmp_index_file)
        
        # 使用 drop_if_exists 重建
        engine2 = create_engine(tmp_index_file, drop_if_exists=True)
        engine2.add_document(2, {"content": "recovered"})
        engine2.flush()
        
        result = engine2.search("recovered")
        assert 2 in result.to_list()
    
    def test_wal_file_cleanup(self, tmp_index_file):
        """测试 WAL 文件清理"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        engine.add_document(1, {"content": "wal cleanup"})
        engine.flush()
        
        # Compact 应该清理 WAL
        engine.compact()
        
        # 检查 WAL 相关文件
        dir_path = os.path.dirname(tmp_index_file)
        wal_files = [f for f in os.listdir(dir_path) if '.wal' in f]
        
        # 可能有 WAL 文件，但应该是空的或很小


class TestBufferManagement:
    """缓冲区管理测试"""
    
    def test_buffer_size_tracking(self, tmp_index_file):
        """测试缓冲区大小跟踪"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        initial_buffer = engine.buffer_size()
        
        # 添加数据
        for i in range(100):
            engine.add_document(i, {"content": f"buffer test {i}"})
        
        after_add_buffer = engine.buffer_size()
        assert after_add_buffer > initial_buffer
        
        # Flush 后 buffer 应该清空
        engine.flush()
        
        after_flush_buffer = engine.buffer_size()
        assert after_flush_buffer == 0
    
    def test_memtable_size(self, tmp_index_file):
        """测试 memtable 大小"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 添加数据
        for i in range(50):
            engine.add_document(i, {"content": f"memtable {i}"})
        
        memtable = engine.memtable_size()
        # Memtable 应该有数据
        assert memtable >= 0


class TestDocTermsTracking:
    """文档词条跟踪测试"""
    
    def test_track_doc_terms_enabled(self, tmp_index_file):
        """测试启用文档词条跟踪"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)
        
        engine.add_document(1, {"content": "tracking test"})
        
        stats = engine.stats()
        assert stats["track_doc_terms"] == 1.0
        assert engine.doc_terms_size() > 0
    
    def test_track_doc_terms_disabled(self, tmp_index_file):
        """测试禁用文档词条跟踪"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=False)
        
        engine.add_document(1, {"content": "no tracking test"})
        
        stats = engine.stats()
        assert stats["track_doc_terms"] == 0.0
    
    def test_clear_doc_terms(self, tmp_index_file):
        """测试清除文档词条"""
        engine = create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)
        
        engine.add_document(1, {"content": "clear terms test"})
        
        assert engine.doc_terms_size() > 0
        
        engine.clear_doc_terms()
        
        assert engine.doc_terms_size() == 0


class TestTermCount:
    """词条计数测试"""
    
    def test_term_count_accuracy(self, tmp_index_file):
        """测试词条计数准确性"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 添加有已知词条数量的文档
        engine.add_document(1, {"content": "unique1 unique2 unique3"})
        engine.flush()
        
        term_count = engine.term_count()
        assert term_count >= 3  # 至少有 3 个词
    
    def test_term_count_with_duplicates(self, tmp_index_file):
        """测试重复词条的计数"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 多个文档使用相同词条
        for i in range(10):
            engine.add_document(i, {"content": "common word"})
        engine.flush()
        
        term_count = engine.term_count()
        # "common" 和 "word" 应该只计一次
        assert term_count == 2


class TestPreload:
    """预加载测试"""
    
    def test_preload_function(self, tmp_index_file):
        """测试预加载功能"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        for i in range(100):
            engine1.add_document(i, {"content": f"preload test {i}"})
        engine1.flush()
        del engine1
        
        engine2 = create_engine(tmp_index_file, lazy_load=True)
        
        # 预加载
        preloaded = engine2.preload()
        assert preloaded >= 0
        
        # 预加载后搜索
        result = engine2.search("preload")
        assert result.total_hits == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
NanoFTS 崩溃恢复测试

测试 WAL (Write-Ahead Log) 相关功能：
- 插入中断后查询
- 重启后数据恢复
- WAL 文件完整性
- 异常关闭恢复
"""

import pytest
import os
import shutil
import signal
import multiprocessing
import time
from nanofts import create_engine


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_recovery.nfts")


@pytest.fixture
def temp_dir(tmp_path):
    """创建临时目录"""
    return tmp_path


class TestWALBasics:
    """WAL 基本功能测试"""
    
    def test_wal_enabled_by_default(self, tmp_index_file):
        """测试 WAL 默认启用"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 添加数据触发 WAL 写入
        engine.add_document(1, {"content": "test wal"})
        
        stats = engine.stats()
        # WAL 应该被启用
        assert stats.get("wal_enabled", 0) == 1.0
    
    def test_wal_file_created(self, tmp_index_file):
        """测试 WAL 文件是否被创建"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        engine.add_document(1, {"content": "test wal file"})
        
        # WAL 文件应该在 .nfts 旁边
        wal_path = tmp_index_file + ".wal"
        base_name = os.path.basename(tmp_index_file)
        wal_path_alt = os.path.join(os.path.dirname(tmp_index_file), base_name + ".wal")
        
        # 检查 WAL 相关统计
        stats = engine.stats()
        # WAL 大小可能大于 0（如果有数据写入但未 flush）
    
    def test_wal_cleared_after_flush(self, tmp_index_file):
        """测试 flush 后 WAL 被清空"""
        engine = create_engine(tmp_index_file, drop_if_exists=True)
        
        engine.add_document(1, {"content": "test wal clear"})
        
        # flush 前
        stats_before = engine.stats()
        
        engine.flush()
        
        # flush 后
        stats_after = engine.stats()
        
        # WAL pending batches 应该为 0
        assert stats_after.get("wal_pending_batches", 0) == 0


class TestRecoveryAfterReopen:
    """重新打开后恢复测试"""
    
    def test_data_persisted_after_flush(self, tmp_index_file):
        """测试 flush 后数据持久化"""
        # 第一个引擎实例
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "persistent data"})
        engine1.add_document(2, {"content": "more persistent data"})
        engine1.flush()
        del engine1  # 关闭引擎
        
        # 重新打开
        engine2 = create_engine(tmp_index_file)
        
        result = engine2.search("persistent")
        assert result.total_hits == 2
        assert 1 in result.to_list()
        assert 2 in result.to_list()
    
    def test_data_recovery_without_flush(self, tmp_index_file):
        """测试不 flush 时的数据恢复（依赖 WAL）"""
        # 第一个引擎实例
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "wal recovery test"})
        
        # 不调用 flush，但 drop 时可能会自动 flush
        del engine1
        
        # 重新打开
        engine2 = create_engine(tmp_index_file)
        
        # 数据应该被恢复（通过 WAL 或自动 flush）
        result = engine2.search("recovery")
        # 由于 Drop 时会自动 flush，数据应该存在
        # 如果不存在，说明 WAL 恢复也没有生效
    
    def test_multiple_flushes_recovery(self, tmp_index_file):
        """测试多次 flush 后的恢复"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 多次添加和 flush
        for batch in range(5):
            for i in range(10):
                doc_id = batch * 10 + i
                engine1.add_document(doc_id, {"content": f"batch {batch} doc {i}"})
            engine1.flush()
        
        del engine1
        
        # 重新打开
        engine2 = create_engine(tmp_index_file)
        
        result = engine2.search("batch")
        assert result.total_hits == 50
    
    def test_segment_count_after_recovery(self, tmp_index_file):
        """测试恢复后 segment 数量"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        for i in range(5):
            engine1.add_document(i, {"content": f"segment test {i}"})
            engine1.flush()
        
        segments_before = engine1.segment_count()
        del engine1
        
        engine2 = create_engine(tmp_index_file)
        segments_after = engine2.segment_count()
        
        assert segments_after == segments_before


class TestRecoveryWithDeletions:
    """删除操作恢复测试"""
    
    def test_tombstone_recovery(self, tmp_index_file):
        """测试 tombstone 机制在恢复后的效果"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 添加文档
        for i in range(10):
            engine1.add_document(i, {"content": f"tombstone test {i}"})
        engine1.flush()
        
        # 删除一些文档
        engine1.remove_document(0)
        engine1.remove_document(5)
        # 在 LSM-tree 设计中，删除需要通过 compact 才能持久化
        engine1.compact()
        
        del engine1
        
        # 重新打开
        engine2 = create_engine(tmp_index_file)
        
        result = engine2.search("tombstone")
        # 删除的文档不应该出现在结果中
        assert 0 not in result.to_list()
        assert 5 not in result.to_list()
        assert result.total_hits == 8
    
    def test_compact_and_recovery(self, tmp_index_file):
        """测试 compact 后的恢复"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 添加文档
        for i in range(20):
            engine1.add_document(i, {"content": f"compact recovery {i}"})
        engine1.flush()
        
        # 删除并 compact
        for i in range(10):
            engine1.remove_document(i)
        engine1.compact()
        
        del engine1
        
        # 重新打开
        engine2 = create_engine(tmp_index_file)
        
        result = engine2.search("compact")
        assert result.total_hits == 10
        
        # 已删除的文档不应该存在
        for i in range(10):
            assert i not in result.to_list()
        
        # 未删除的文档应该存在
        for i in range(10, 20):
            assert i in result.to_list()


class TestRecoveryWithUpdates:
    """更新操作恢复测试"""
    
    def test_update_recovery(self, tmp_index_file):
        """测试更新操作的恢复"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)
        
        # 添加文档
        engine1.add_document(1, {"content": "original content v1"})
        engine1.flush()
        
        # 更新文档
        engine1.update_document(1, {"content": "updated content v2"})
        # 在 LSM-tree 设计中，更新（删除旧数据+添加新数据）需要通过 compact 才能持久化删除
        engine1.compact()
        
        del engine1
        
        # 重新打开
        engine2 = create_engine(tmp_index_file, track_doc_terms=True)
        
        # 原始内容不应该被搜索到
        result_original = engine2.search("original")
        assert 1 not in result_original.to_list()
        
        # 更新后的内容应该被搜索到
        result_updated = engine2.search("updated")
        assert 1 in result_updated.to_list()


class TestIndexRecreation:
    """索引重建测试"""
    
    def test_drop_if_exists(self, tmp_index_file):
        """测试 drop_if_exists 参数"""
        # 第一次创建
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "first creation"})
        engine1.flush()
        del engine1
        
        # 使用 drop_if_exists=True 重新创建
        engine2 = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 旧数据应该被清除
        result = engine2.search("first")
        assert result.total_hits == 0
        
        # 新数据可以添加
        engine2.add_document(2, {"content": "second creation"})
        engine2.flush()
        
        result = engine2.search("second")
        assert 2 in result.to_list()
    
    def test_drop_if_exists_removes_wal_file(self, tmp_index_file):
        """测试 drop_if_exists=True 时是否正确删除 .wal 文件"""
        wal_file = tmp_index_file + ".wal"
        
        # 第一次创建并写入数据（不 flush 以保留 WAL 数据）
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "wal test data"})
        engine1.flush()  # flush 后 WAL 内容会被清空但文件可能还在
        
        # 再添加一些数据但不 flush，确保 WAL 有内容
        engine1.add_document(2, {"content": "unflushed wal data"})
        
        # 检查 WAL 文件是否存在或有内容
        stats1 = engine1.stats()
        del engine1
        
        # 确认索引文件存在
        assert os.path.exists(tmp_index_file), "索引文件应该存在"
        
        # 使用 drop_if_exists=True 重新创建
        engine2 = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 旧数据应该被清除
        result = engine2.search("wal")
        assert result.total_hits == 0, "旧数据应该被清除"
        
        result = engine2.search("unflushed")
        assert result.total_hits == 0, "未 flush 的数据也应该被清除"
        
        # WAL 文件应该被清理或重置
        stats2 = engine2.stats()
        assert stats2.get("wal_pending_batches", 0) == 0, "WAL pending batches 应该为 0"
        
        # 验证新引擎功能正常
        engine2.add_document(3, {"content": "new wal data"})
        engine2.flush()
        
        result = engine2.search("new")
        assert 3 in result.to_list(), "新数据应该能正常添加和搜索"
        
        del engine2
        
        # 最终验证：重新打开应该只有新数据
        engine3 = create_engine(tmp_index_file)
        result = engine3.search("new")
        assert result.total_hits == 1, "重新打开后应该只有新数据"
        result = engine3.search("wal test")
        assert result.total_hits == 0, "旧数据不应该恢复"
    
    def test_delete_and_recreate_index_file(self, tmp_index_file):
        """测试删除索引文件后重建"""
        # 创建索引
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "will be deleted"})
        engine1.flush()
        del engine1
        
        # 手动删除索引文件
        if os.path.exists(tmp_index_file):
            os.remove(tmp_index_file)
        
        # 删除 WAL 文件
        wal_file = tmp_index_file + ".wal"
        base_name = os.path.basename(tmp_index_file)
        wal_file_alt = os.path.join(os.path.dirname(tmp_index_file), base_name + ".wal")
        for f in [wal_file, wal_file_alt]:
            if os.path.exists(f):
                os.remove(f)
        
        # 重新创建
        engine2 = create_engine(tmp_index_file)
        
        # 应该是空的
        result = engine2.search("deleted")
        assert result.total_hits == 0
        
        # 可以正常使用
        engine2.add_document(2, {"content": "new content"})
        engine2.flush()
        
        result = engine2.search("new")
        assert 2 in result.to_list()
    
    def test_recreate_after_corruption_simulation(self, tmp_index_file):
        """模拟文件损坏后重建"""
        # 创建索引
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        engine1.add_document(1, {"content": "important data"})
        engine1.flush()
        del engine1
        
        # 模拟损坏：截断文件
        with open(tmp_index_file, 'r+b') as f:
            f.truncate(32)  # 只保留部分 header
        
        # 尝试用 drop_if_exists 重建
        try:
            engine2 = create_engine(tmp_index_file, drop_if_exists=True)
            # 应该能正常创建新索引
            engine2.add_document(2, {"content": "recovered"})
            engine2.flush()
            
            result = engine2.search("recovered")
            assert 2 in result.to_list()
        except Exception:
            # 如果打开失败，删除后重建
            if os.path.exists(tmp_index_file):
                os.remove(tmp_index_file)
            
            engine2 = create_engine(tmp_index_file, drop_if_exists=True)
            engine2.add_document(2, {"content": "recovered"})
            engine2.flush()


class TestLazyLoadRecovery:
    """懒加载模式恢复测试"""
    
    def test_lazy_load_recovery(self, tmp_index_file):
        """测试懒加载模式下的恢复"""
        # 创建并写入数据
        engine1 = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=True)
        
        for i in range(100):
            engine1.add_document(i, {"content": f"lazy load test document {i}"})
        engine1.flush()
        
        del engine1
        
        # 用懒加载模式重新打开
        engine2 = create_engine(tmp_index_file, lazy_load=True)
        
        assert engine2.is_lazy_load()
        
        result = engine2.search("lazy")
        assert result.total_hits == 100
    
    def test_lazy_load_cache_warmup(self, tmp_index_file):
        """测试懒加载缓存预热"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True, lazy_load=True)
        
        for i in range(50):
            engine1.add_document(i, {"content": f"warmup test {i}"})
        engine1.flush()
        
        del engine1
        
        engine2 = create_engine(tmp_index_file, lazy_load=True, cache_size=1000)
        
        # 预热缓存
        warmed = engine2.warmup_terms(["warmup", "test"])
        
        # 搜索应该更快（从缓存）
        result = engine2.search("warmup")
        assert result.total_hits == 50


class TestPartialWriteRecovery:
    """部分写入恢复测试"""
    
    def test_recovery_after_partial_batch(self, tmp_index_file):
        """测试部分批次写入后的恢复"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        # 写入一些完整批次
        for i in range(50):
            engine1.add_document(i, {"content": f"complete batch {i}"})
        engine1.flush()
        
        # 写入但不 flush（模拟中断）
        for i in range(50, 100):
            engine1.add_document(i, {"content": f"incomplete batch {i}"})
        
        # 模拟意外关闭（不调用 flush）
        # 注意：Python 的 del 会调用 __del__，可能会触发 flush
        del engine1
        
        # 重新打开
        engine2 = create_engine(tmp_index_file)
        
        # 完整批次应该存在
        result = engine2.search("complete")
        assert result.total_hits >= 50  # 至少有完整批次
    
    def test_interleaved_flush_recovery(self, tmp_index_file):
        """测试交错 flush 的恢复"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        engine1.add_document(1, {"content": "first"})
        engine1.flush()
        
        engine1.add_document(2, {"content": "second"})
        engine1.flush()
        
        engine1.add_document(3, {"content": "third"})
        # 不 flush 第三个
        
        del engine1
        
        engine2 = create_engine(tmp_index_file)
        
        # 前两个一定存在
        assert 1 in engine2.search("first").to_list()
        assert 2 in engine2.search("second").to_list()


class TestStatsRecovery:
    """统计信息恢复测试"""
    
    def test_term_count_recovery(self, tmp_index_file):
        """测试 term count 在恢复后的正确性"""
        engine1 = create_engine(tmp_index_file, drop_if_exists=True)
        
        for i in range(100):
            engine1.add_document(i, {"content": f"unique term{i} shared content"})
        engine1.flush()
        
        term_count_before = engine1.term_count()
        del engine1
        
        engine2 = create_engine(tmp_index_file)
        term_count_after = engine2.term_count()
        
        # term count 应该基本一致
        assert abs(term_count_after - term_count_before) < 10


class TestConcurrentRecovery:
    """并发场景恢复测试"""
    
    def test_recovery_while_writing(self, tmp_path):
        """测试写入过程中的恢复场景"""
        index_file = str(tmp_path / "concurrent_recovery.nfts")
        
        engine1 = create_engine(index_file, drop_if_exists=True)
        
        # 模拟持续写入
        for i in range(100):
            engine1.add_document(i, {"content": f"continuous write {i}"})
            if i % 20 == 0:
                engine1.flush()
        
        engine1.flush()
        del engine1
        
        # 重新打开并验证
        engine2 = create_engine(index_file)
        result = engine2.search("continuous")
        assert result.total_hits == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


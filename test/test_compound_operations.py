"""
NanoFTS 复合操作测试

测试多个操作组合的场景：
- 删除后重建索引
- 多次更新同一文档
- 混合批量操作
- 复杂查询组合
"""

import pytest
import os
import time
from nanofts import create_engine


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_compound.nfts")


@pytest.fixture
def engine(tmp_index_file):
    """创建测试引擎"""
    return create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)


class TestDeleteAndRebuild:
    """删除后重建测试"""
    
    def test_remove_all_and_rebuild(self, engine):
        """测试删除所有文档后重建"""
        # 添加文档
        for i in range(50):
            engine.add_document(i, {"content": f"original document {i}"})
        engine.flush()
        
        # 验证添加成功
        result = engine.search("original")
        assert result.total_hits == 50
        
        # 删除所有文档
        for i in range(50):
            engine.remove_document(i)
        engine.flush()
        
        # 验证全部删除
        result = engine.search("original")
        assert result.total_hits == 0
        
        # 重新添加文档
        for i in range(100, 150):
            engine.add_document(i, {"content": f"new document {i}"})
        engine.flush()
        
        # 验证新文档
        result = engine.search("new")
        assert result.total_hits == 50
    
    def test_delete_recreate_same_id(self, engine):
        """测试删除后用相同 ID 重新创建"""
        # 创建文档
        engine.add_document(1, {"content": "version one content"})
        engine.flush()
        
        # 删除
        engine.remove_document(1)
        engine.flush()
        
        # 用相同 ID 重新创建
        engine.add_document(1, {"content": "version two content"})
        engine.flush()
        
        # 验证
        result_v1 = engine.search("version one")
        result_v2 = engine.search("version two")
        
        assert 1 not in result_v1.to_list()
        assert 1 in result_v2.to_list()
    
    def test_batch_delete_and_rebuild(self, engine):
        """测试批量删除后重建"""
        # 添加文档
        docs = [(i, {"content": f"batch doc {i}"}) for i in range(100)]
        engine.add_documents(docs)
        engine.flush()
        
        # 批量删除前一半
        engine.remove_documents(list(range(50)))
        engine.flush()
        
        # 验证删除
        result = engine.search("batch")
        assert result.total_hits == 50
        for i in range(50):
            assert i not in result.to_list()
        
        # 重建删除的文档
        new_docs = [(i, {"content": f"rebuilt doc {i}"}) for i in range(50)]
        engine.add_documents(new_docs)
        engine.flush()
        
        # 验证重建
        result = engine.search("rebuilt")
        assert result.total_hits == 50


class TestMultipleUpdates:
    """多次更新测试"""
    
    def test_rapid_updates_same_document(self, engine):
        """测试快速多次更新同一文档"""
        # 使用 vXX 格式避免单字符词被 min_term_length 过滤
        engine.add_document(1, {"content": "version v00"})
        engine.flush()
        
        # 快速更新 10 次
        for v in range(1, 11):
            engine.update_document(1, {"content": f"version v{v:02d}"})
        engine.flush()
        
        # 只有最终版本应该被搜索到
        for v in range(10):
            result = engine.search(f"v{v:02d}")  # 搜索 v00, v01, ..., v09
            if v == 10:
                assert 1 in result.to_list()
            else:
                assert 1 not in result.to_list()
        
        # 验证最终版本 v10
        result = engine.search("v10")
        assert 1 in result.to_list()
    
    def test_update_without_intermediate_flush(self, engine):
        """测试不 flush 的连续更新"""
        engine.add_document(1, {"content": "initial content"})
        
        # 连续更新不 flush
        engine.update_document(1, {"content": "update 1"})
        engine.update_document(1, {"content": "update 2"})
        engine.update_document(1, {"content": "final update"})
        
        engine.flush()
        
        result = engine.search("final")
        assert 1 in result.to_list()
        
        result = engine.search("initial")
        assert 1 not in result.to_list()
    
    def test_update_multiple_documents_same_content(self, engine):
        """测试更新多个文档为相同内容"""
        # 创建不同内容的文档
        for i in range(10):
            engine.add_document(i, {"content": f"unique content {i}"})
        engine.flush()
        
        # 更新所有文档为相同内容
        for i in range(10):
            engine.update_document(i, {"content": "shared content"})
        engine.flush()
        
        # 所有文档都应该能通过 shared 搜索到
        result = engine.search("shared")
        assert result.total_hits == 10


class TestMixedBatchOperations:
    """混合批量操作测试"""
    
    def test_interleaved_add_delete(self, engine):
        """测试交错的添加和删除"""
        # 添加 -> 删除 -> 添加 -> 删除...
        for i in range(10):
            engine.add_document(i * 2, {"content": f"even doc {i * 2}"})
            engine.add_document(i * 2 + 1, {"content": f"odd doc {i * 2 + 1}"})
            engine.remove_document(i * 2)  # 删除偶数
        
        engine.flush()
        
        # 只有奇数文档存在
        result = engine.search("doc")
        assert result.total_hits == 10
        
        for i in range(10):
            assert i * 2 not in result.to_list()  # 偶数不在
            assert i * 2 + 1 in result.to_list()  # 奇数在
    
    def test_add_update_delete_sequence(self, engine):
        """测试添加-更新-删除序列"""
        # 添加 - 使用有意义的词避免 min_term_length 过滤
        engine.add_document(1, {"content": "phase_add original"})
        engine.flush()
        
        # 更新
        engine.update_document(1, {"content": "phase_update modified"})
        engine.flush()
        
        # 删除
        engine.remove_document(1)
        engine.flush()
        
        # 重新添加
        engine.add_document(1, {"content": "phase_readd final"})
        engine.flush()
        
        # 只有最后一步的内容存在
        assert 1 not in engine.search("original").to_list()
        assert 1 not in engine.search("modified").to_list()
        assert 1 in engine.search("final").to_list()
    
    def test_bulk_operations_same_term(self, engine):
        """测试对包含相同词条的文档进行批量操作"""
        # 所有文档都包含 "common"
        for i in range(100):
            engine.add_document(i, {"content": f"common content {i}"})
        engine.flush()
        
        # 删除一半
        engine.remove_documents(list(range(0, 100, 2)))  # 删除偶数
        engine.flush()
        
        result = engine.search("common")
        assert result.total_hits == 50
        
        # 更新剩余的一半
        for i in range(1, 100, 2):
            engine.update_document(i, {"content": f"updated common {i}"})
        engine.flush()
        
        result = engine.search("updated")
        assert result.total_hits == 50


class TestComplexQueryCombinations:
    """复杂查询组合测试"""
    
    def test_and_search_with_updates(self, engine):
        """测试 AND 搜索在更新后的行为"""
        engine.add_document(1, {"content": "apple banana cherry"})
        engine.add_document(2, {"content": "apple banana"})
        engine.add_document(3, {"content": "apple"})
        engine.flush()
        
        # AND 搜索
        result = engine.search_and(["apple", "banana"])
        assert 1 in result.to_list()
        assert 2 in result.to_list()
        assert 3 not in result.to_list()
        
        # 更新文档 3 添加 banana
        engine.update_document(3, {"content": "apple banana"})
        engine.flush()
        
        result = engine.search_and(["apple", "banana"])
        assert 3 in result.to_list()
    
    def test_or_search_with_deletions(self, engine):
        """测试 OR 搜索在删除后的行为"""
        engine.add_document(1, {"content": "cat"})
        engine.add_document(2, {"content": "dog"})
        engine.add_document(3, {"content": "bird"})
        engine.flush()
        
        # OR 搜索
        result = engine.search_or(["cat", "dog"])
        assert result.total_hits == 2
        
        # 删除 cat
        engine.remove_document(1)
        engine.flush()
        
        result = engine.search_or(["cat", "dog"])
        assert result.total_hits == 1
        assert 2 in result.to_list()
    
    def test_result_operations_chain(self, engine):
        """测试结果集操作链"""
        engine.add_document(1, {"content": "python programming language"})
        engine.add_document(2, {"content": "java programming language"})
        engine.add_document(3, {"content": "python scripting"})
        engine.add_document(4, {"content": "java enterprise"})
        engine.flush()
        
        # 搜索 programming
        r1 = engine.search("programming")
        assert r1.total_hits == 2
        
        # 搜索 python
        r2 = engine.search("python")
        assert r2.total_hits == 2
        
        # 交集
        r3 = r1.intersect(r2)
        assert r3.total_hits == 1
        assert 1 in r3.to_list()
        
        # 并集
        r4 = r1.union(r2)
        assert r4.total_hits == 3
        
        # 差集
        r5 = r2.difference(r1)  # python but not programming
        assert r5.total_hits == 1
        assert 3 in r5.to_list()


class TestCompactWithOperations:
    """Compact 与其他操作组合测试"""
    
    def test_compact_after_many_deletions(self, engine):
        """测试大量删除后 compact"""
        # 添加大量文档
        for i in range(500):
            engine.add_document(i, {"content": f"before compact {i}"})
        engine.flush()
        
        # 删除大部分
        engine.remove_documents(list(range(400)))
        engine.flush()
        
        # Compact
        engine.compact()
        
        # 验证数据完整性
        result = engine.search("before")
        assert result.total_hits == 100
        
        for i in range(400):
            assert i not in result.to_list()
        for i in range(400, 500):
            assert i in result.to_list()
    
    def test_operations_after_compact(self, engine):
        """测试 compact 后的正常操作"""
        # 初始数据
        for i in range(100):
            engine.add_document(i, {"content": f"compact test {i}"})
        engine.flush()
        engine.compact()
        
        # Compact 后添加
        engine.add_document(200, {"content": "after compact new"})
        engine.flush()
        
        result = engine.search("after")
        assert 200 in result.to_list()
        
        # Compact 后删除
        engine.remove_document(50)
        engine.flush()
        
        result = engine.search("compact")
        assert 50 not in result.to_list()
        
        # Compact 后更新
        engine.update_document(60, {"content": "modified after compact"})
        engine.flush()
        
        result = engine.search("modified")
        assert 60 in result.to_list()
    
    def test_multiple_compact_cycles(self, engine):
        """测试多次 compact 循环"""
        # 使用更长的标识词避免 min_term_length=2 过滤单字符
        cycle_names = ["alpha", "beta", "gamma"]
        for idx, cycle_name in enumerate(cycle_names):
            # 添加数据
            for i in range(50):
                doc_id = idx * 100 + i
                engine.add_document(doc_id, {"content": f"cycle {cycle_name} doc number{i:02d}"})
            engine.flush()
            
            # 删除一半
            for i in range(25):
                engine.remove_document(idx * 100 + i)
            engine.flush()
            
            # Compact
            engine.compact()
        
        # 验证最终状态
        for cycle_name in cycle_names:
            result = engine.search(f"cycle {cycle_name}")
            assert result.total_hits == 25, f"Expected 25 for {cycle_name}, got {result.total_hits}"


class TestEdgeCaseSequences:
    """边缘情况序列测试"""
    
    def test_flush_compact_flush_sequence(self, engine):
        """测试 flush-compact-flush 序列"""
        engine.add_document(1, {"content": "sequence test"})
        engine.flush()
        
        engine.compact()
        
        engine.add_document(2, {"content": "after compact"})
        engine.flush()
        
        result = engine.search("test")
        assert 1 in result.to_list()
        
        result = engine.search("after")
        assert 2 in result.to_list()
    
    def test_delete_before_first_flush(self, engine):
        """测试首次 flush 前删除"""
        engine.add_document(1, {"content": "will be deleted"})
        engine.add_document(2, {"content": "will remain"})
        
        # 在 flush 前删除
        engine.remove_document(1)
        
        engine.flush()
        
        result = engine.search("will")
        assert 1 not in result.to_list()
        assert 2 in result.to_list()
    
    def test_update_before_first_flush(self, engine):
        """测试首次 flush 前更新"""
        engine.add_document(1, {"content": "original"})
        
        # 在 flush 前更新
        engine.update_document(1, {"content": "updated"})
        
        engine.flush()
        
        assert 1 not in engine.search("original").to_list()
        assert 1 in engine.search("updated").to_list()
    
    def test_same_id_multiple_operations_no_flush(self, engine):
        """测试同一 ID 多次操作不 flush"""
        # 添加 -> 更新 -> 删除 -> 重新添加，全部不 flush
        engine.add_document(1, {"content": "step1"})
        engine.update_document(1, {"content": "step2"})
        engine.remove_document(1)
        engine.add_document(1, {"content": "step3"})
        
        engine.flush()
        
        assert 1 not in engine.search("step1").to_list()
        assert 1 not in engine.search("step2").to_list()
        assert 1 in engine.search("step3").to_list()


class TestCacheAndSearchInteraction:
    """缓存与搜索交互测试"""
    
    def test_cache_invalidation_on_update(self, engine):
        """测试更新时缓存失效"""
        engine.add_document(1, {"content": "cached content"})
        engine.flush()
        
        # 第一次搜索（填充缓存）
        result1 = engine.search("cached")
        assert 1 in result1.to_list()
        
        # 更新文档
        engine.update_document(1, {"content": "new content"})
        engine.flush()
        
        # 缓存应该失效，新搜索应该反映更新
        result2 = engine.search("cached")
        assert 1 not in result2.to_list()
        
        result3 = engine.search("new")
        assert 1 in result3.to_list()
    
    def test_cache_clear_explicit(self, engine):
        """测试显式清除缓存"""
        engine.add_document(1, {"content": "cache test"})
        engine.flush()
        
        # 搜索以填充缓存
        engine.search("cache")
        
        # 清除缓存
        engine.clear_cache()
        
        # 再次搜索应该正常工作
        result = engine.search("cache")
        assert 1 in result.to_list()


class TestFuzzySearchCombinations:
    """模糊搜索组合测试"""
    
    def test_fuzzy_after_exact(self, engine):
        """测试精确搜索后进行模糊搜索"""
        engine.add_document(1, {"content": "hello world"})
        engine.add_document(2, {"content": "helo wrld"})  # 拼写错误
        engine.flush()
        
        # 精确搜索
        exact_result = engine.search("hello")
        assert 1 in exact_result.to_list()
        
        # 模糊搜索（应该能找到拼写错误的）
        fuzzy_result = engine.fuzzy_search("hello", min_results=0)
        # 模糊搜索可能找到更多结果
    
    def test_fuzzy_config_change(self, engine):
        """测试模糊搜索配置更改"""
        engine.add_document(1, {"content": "configuration test"})
        engine.flush()
        
        # 更改模糊配置
        engine.set_fuzzy_config(threshold=0.5, max_distance=3, max_candidates=30)
        
        config = engine.get_fuzzy_config()
        assert config["threshold"] == 0.5
        
        # 模糊搜索应该使用新配置
        result = engine.fuzzy_search("configuraton", min_results=0)  # 拼写错误


class TestStatisticsConsistency:
    """统计信息一致性测试"""
    
    def test_stats_after_operations(self, engine):
        """测试操作后统计信息的一致性"""
        # 初始统计
        stats_initial = engine.stats()
        initial_search_count = stats_initial.get("search_count", 0)
        
        # 添加文档
        for i in range(100):
            engine.add_document(i, {"content": f"stats test {i}"})
        engine.flush()
        
        # 搜索
        for _ in range(10):
            engine.search("stats")
        
        stats_after = engine.stats()
        assert stats_after["search_count"] >= initial_search_count + 10
        assert stats_after["term_count"] > 0
    
    def test_deleted_count_tracking(self, engine):
        """测试删除计数跟踪"""
        for i in range(50):
            engine.add_document(i, {"content": f"delete tracking {i}"})
        engine.flush()
        
        # 删除一些文档
        for i in range(20):
            engine.remove_document(i)
        engine.flush()
        
        stats = engine.stats()
        assert stats.get("deleted_count", 0) >= 20
        
        # Compact 后删除计数应该重置
        engine.compact()
        
        stats_after_compact = engine.stats()
        assert stats_after_compact.get("deleted_count", 0) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


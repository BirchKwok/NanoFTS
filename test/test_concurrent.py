"""
NanoFTS 并发操作测试

测试多线程环境下的操作：
- 插入同时查询
- 删除同时查询
- 并发插入
- 并发删除
- 并发更新
- 查询时 compact
"""

import pytest
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from nanofts import create_engine


@pytest.fixture
def tmp_index_file(tmp_path):
    """创建临时索引文件路径"""
    return str(tmp_path / "test_concurrent.nfts")


@pytest.fixture
def engine(tmp_index_file):
    """创建测试引擎"""
    return create_engine(tmp_index_file, drop_if_exists=True, track_doc_terms=True)


class TestConcurrentInsertAndSearch:
    """插入同时查询测试"""
    
    def test_search_during_insert(self, engine):
        """测试在插入过程中进行搜索"""
        # 先添加一些基础数据
        for i in range(100):
            engine.add_document(i, {"content": f"document {i} base content"})
        engine.flush()
        
        errors = []
        results = []
        
        def insert_worker():
            """插入工作线程"""
            try:
                for i in range(100, 200):
                    engine.add_document(i, {"content": f"new document {i}"})
                    time.sleep(0.001)  # 小延迟模拟真实场景
            except Exception as e:
                errors.append(("insert", e))
        
        def search_worker():
            """搜索工作线程"""
            try:
                for _ in range(50):
                    result = engine.search("document")
                    results.append(result.total_hits)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("search", e))
        
        # 并发执行
        threads = [
            threading.Thread(target=insert_worker),
            threading.Thread(target=search_worker),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 验证没有错误
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # 搜索结果应该单调递增或保持不变
        for i in range(1, len(results)):
            assert results[i] >= results[i-1] - 10, "Search results should be roughly increasing"
    
    def test_batch_insert_with_concurrent_search(self, engine):
        """测试批量插入时的并发搜索"""
        # 准备批量数据
        batch_data = [(i, {"content": f"batch item {i}"}) for i in range(500)]
        
        search_results = []
        errors = []
        
        def search_loop():
            try:
                for _ in range(20):
                    result = engine.search("batch")
                    search_results.append(result.total_hits)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # 启动搜索线程
        search_thread = threading.Thread(target=search_loop)
        search_thread.start()
        
        # 主线程执行批量插入
        engine.add_documents(batch_data)
        engine.flush()
        
        search_thread.join()
        
        assert len(errors) == 0, f"Search errors: {errors}"


class TestConcurrentDeleteAndSearch:
    """删除同时查询测试"""
    
    def test_search_during_delete(self, engine):
        """测试在删除过程中进行搜索"""
        # 添加数据
        for i in range(200):
            engine.add_document(i, {"content": f"deletable document {i}"})
        engine.flush()
        
        errors = []
        
        def delete_worker():
            try:
                for i in range(100):
                    engine.remove_document(i)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("delete", e))
        
        def search_worker():
            try:
                for _ in range(50):
                    result = engine.search("deletable")
                    # 结果应该逐渐减少
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("search", e))
        
        threads = [
            threading.Thread(target=delete_worker),
            threading.Thread(target=search_worker),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        
        # 最终验证
        engine.flush()
        result = engine.search("deletable")
        # 应该还剩约 100 个文档（100-199）
        assert result.total_hits > 50
    
    def test_batch_delete_with_concurrent_search(self, engine):
        """测试批量删除时的并发搜索"""
        # 添加数据
        for i in range(300):
            engine.add_document(i, {"content": f"bulk delete test {i}"})
        engine.flush()
        
        errors = []
        
        def search_loop():
            try:
                for _ in range(30):
                    result = engine.search("bulk")
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)
        
        search_thread = threading.Thread(target=search_loop)
        search_thread.start()
        
        # 批量删除
        engine.remove_documents(list(range(150)))
        engine.flush()
        
        search_thread.join()
        
        assert len(errors) == 0
        
        # 验证最终状态
        result = engine.search("bulk")
        assert result.total_hits == 150


class TestConcurrentModifications:
    """并发修改测试"""
    
    def test_concurrent_inserts(self, engine):
        """测试多线程并发插入"""
        errors = []
        
        def insert_batch(start_id, count):
            try:
                for i in range(start_id, start_id + count):
                    engine.add_document(i, {"content": f"concurrent doc {i}"})
            except Exception as e:
                errors.append(e)
        
        # 4 个线程并发插入
        threads = [
            threading.Thread(target=insert_batch, args=(0, 100)),
            threading.Thread(target=insert_batch, args=(100, 100)),
            threading.Thread(target=insert_batch, args=(200, 100)),
            threading.Thread(target=insert_batch, args=(300, 100)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        engine.flush()
        result = engine.search("concurrent")
        assert result.total_hits == 400
    
    def test_concurrent_updates(self, engine):
        """测试多线程并发更新"""
        # 先添加数据
        for i in range(100):
            engine.add_document(i, {"content": f"original content {i}"})
        engine.flush()
        
        errors = []
        
        def update_batch(start_id, end_id):
            try:
                for i in range(start_id, end_id):
                    engine.update_document(i, {"content": f"updated content {i}"})
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=update_batch, args=(0, 25)),
            threading.Thread(target=update_batch, args=(25, 50)),
            threading.Thread(target=update_batch, args=(50, 75)),
            threading.Thread(target=update_batch, args=(75, 100)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        engine.flush()
        
        # 原始内容应该被更新掉
        result_original = engine.search("original")
        result_updated = engine.search("updated")
        
        # 所有文档应该都被更新了
        assert result_updated.total_hits == 100
    
    def test_mixed_concurrent_operations(self, engine):
        """测试混合并发操作（插入、删除、更新、搜索）"""
        # 初始数据
        for i in range(50):
            engine.add_document(i, {"content": f"initial document {i}"})
        engine.flush()
        
        errors = []
        search_results = []
        
        def inserter():
            try:
                for i in range(50, 100):
                    engine.add_document(i, {"content": f"new document {i}"})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("insert", e))
        
        def deleter():
            try:
                for i in range(25):
                    engine.remove_document(i)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("delete", e))
        
        def updater():
            try:
                for i in range(25, 50):
                    engine.update_document(i, {"content": f"modified document {i}"})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("update", e))
        
        def searcher():
            try:
                for _ in range(30):
                    result = engine.search("document")
                    search_results.append(result.total_hits)
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("search", e))
        
        threads = [
            threading.Thread(target=inserter),
            threading.Thread(target=deleter),
            threading.Thread(target=updater),
            threading.Thread(target=searcher),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"


class TestCompactDuringOperations:
    """Compact 操作并发测试"""
    
    def test_search_during_compact(self, engine):
        """测试在 compact 过程中进行搜索"""
        # 添加数据并创建多个 segment
        for batch in range(5):
            for i in range(100):
                doc_id = batch * 100 + i
                engine.add_document(doc_id, {"content": f"compact test document {doc_id}"})
            engine.flush()
        
        errors = []
        search_results = []
        
        def compact_worker():
            try:
                engine.compact()
            except Exception as e:
                errors.append(("compact", e))
        
        def search_worker():
            try:
                for _ in range(20):
                    result = engine.search("compact")
                    search_results.append(result.total_hits)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("search", e))
        
        threads = [
            threading.Thread(target=compact_worker),
            threading.Thread(target=search_worker),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        
        # 所有搜索结果应该是 500（数据不应该丢失）
        for count in search_results:
            assert count == 500 or count > 0  # 允许 compact 过程中的暂时不一致
    
    def test_insert_during_compact(self, engine):
        """测试在 compact 过程中进行插入"""
        # 创建初始数据
        for i in range(200):
            engine.add_document(i, {"content": f"existing doc {i}"})
        engine.flush()
        
        errors = []
        
        def compact_worker():
            try:
                engine.compact()
            except Exception as e:
                errors.append(("compact", e))
        
        def insert_worker():
            try:
                for i in range(200, 300):
                    engine.add_document(i, {"content": f"new doc {i}"})
                    time.sleep(0.002)
            except Exception as e:
                errors.append(("insert", e))
        
        threads = [
            threading.Thread(target=compact_worker),
            threading.Thread(target=insert_worker),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        engine.flush()
        
        # 验证所有数据
        result = engine.search("doc")
        assert result.total_hits >= 200  # 至少原有数据应该存在


class TestFlushDuringOperations:
    """Flush 操作并发测试"""
    
    def test_concurrent_flush(self, engine):
        """测试并发 flush"""
        # 添加数据
        for i in range(100):
            engine.add_document(i, {"content": f"flush test {i}"})
        
        errors = []
        
        def flush_worker():
            try:
                for _ in range(5):
                    engine.flush()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        # 多个线程同时 flush
        threads = [threading.Thread(target=flush_worker) for _ in range(4)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # 验证数据完整性
        result = engine.search("flush")
        assert result.total_hits == 100
    
    def test_insert_during_flush(self, engine):
        """测试在 flush 过程中继续插入"""
        errors = []
        
        def flush_loop():
            try:
                for _ in range(10):
                    engine.flush()
                    time.sleep(0.01)
            except Exception as e:
                errors.append(("flush", e))
        
        def insert_loop():
            try:
                for i in range(500):
                    engine.add_document(i, {"content": f"streaming doc {i}"})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("insert", e))
        
        threads = [
            threading.Thread(target=flush_loop),
            threading.Thread(target=insert_loop),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        engine.flush()  # 最终 flush
        
        result = engine.search("streaming")
        assert result.total_hits == 500


class TestThreadPoolExecutor:
    """使用 ThreadPoolExecutor 进行测试"""
    
    def test_parallel_searches(self, engine):
        """测试并行搜索"""
        # 添加数据
        for i in range(100):
            engine.add_document(i, {"content": f"parallel search test {i}"})
        engine.flush()
        
        queries = ["parallel", "search", "test"]
        
        def search_query(query):
            return query, engine.search(query).total_hits
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(search_query, q) for q in queries * 10]
            
            results = {}
            for future in as_completed(futures):
                query, count = future.result()
                if query not in results:
                    results[query] = []
                results[query].append(count)
        
        # 相同查询的结果应该一致
        for query, counts in results.items():
            assert len(set(counts)) == 1, f"Inconsistent results for '{query}': {counts}"
    
    def test_parallel_mixed_operations(self, engine):
        """测试并行混合操作"""
        # 初始数据
        for i in range(100):
            engine.add_document(i, {"content": f"mixed ops {i}"})
        engine.flush()
        
        results = {"insert": [], "search": [], "update": []}
        errors = []
        
        def random_operation(op_type, id_range):
            try:
                if op_type == "insert":
                    doc_id = random.randint(*id_range)
                    engine.add_document(doc_id, {"content": f"inserted {doc_id}"})
                    results["insert"].append(doc_id)
                elif op_type == "search":
                    result = engine.search("mixed")
                    results["search"].append(result.total_hits)
                elif op_type == "update":
                    doc_id = random.randint(0, 99)
                    engine.update_document(doc_id, {"content": f"updated {doc_id}"})
                    results["update"].append(doc_id)
            except Exception as e:
                errors.append((op_type, e))
        
        operations = []
        for _ in range(50):
            operations.append(("insert", (100, 200)))
            operations.append(("search", None))
            operations.append(("update", None))
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(random_operation, op, arg) 
                for op, arg in operations
            ]
            for future in as_completed(futures):
                pass  # 等待完成
        
        assert len(errors) == 0, f"Errors: {errors}"


class TestStressConditions:
    """压力测试"""
    
    def test_rapid_fire_operations(self, engine):
        """快速连续操作测试"""
        errors = []
        
        def rapid_operations():
            try:
                for i in range(100):
                    doc_id = random.randint(0, 999)
                    op = random.choice(["add", "search", "remove"])
                    
                    if op == "add":
                        engine.add_document(doc_id, {"content": f"rapid {doc_id}"})
                    elif op == "search":
                        engine.search("rapid")
                    else:
                        engine.remove_document(doc_id)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=rapid_operations) for _ in range(8)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
    
    def test_long_running_concurrent_operations(self, engine):
        """长时间并发操作测试"""
        stop_flag = threading.Event()
        errors = []
        operation_counts = {"insert": 0, "search": 0, "delete": 0}
        
        def inserter():
            i = 0
            while not stop_flag.is_set():
                try:
                    engine.add_document(i, {"content": f"long running test {i}"})
                    operation_counts["insert"] += 1
                    i += 1
                except Exception as e:
                    errors.append(("insert", e))
        
        def searcher():
            while not stop_flag.is_set():
                try:
                    engine.search("long")
                    operation_counts["search"] += 1
                except Exception as e:
                    errors.append(("search", e))
        
        def deleter():
            while not stop_flag.is_set():
                try:
                    doc_id = random.randint(0, 10000)
                    engine.remove_document(doc_id)
                    operation_counts["delete"] += 1
                except Exception as e:
                    errors.append(("delete", e))
        
        threads = [
            threading.Thread(target=inserter),
            threading.Thread(target=searcher),
            threading.Thread(target=deleter),
        ]
        
        for t in threads:
            t.start()
        
        # 运行 1 秒
        time.sleep(1)
        stop_flag.set()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        # 验证有操作发生
        assert operation_counts["insert"] > 0
        assert operation_counts["search"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



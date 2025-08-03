"""
内存管理模块 - 用于处理大规模数据时的内存优化
"""

import gc
import psutil
import threading
import time
from typing import Optional, Callable, Any
from pathlib import Path


class MemoryManager:
    """内存管理器，用于监控和控制内存使用"""
    
    def __init__(self, 
                 memory_limit_mb: int = 8192,  # 8GB默认限制
                 warning_threshold: float = 0.8,  # 80%时警告
                 force_gc_threshold: float = 0.9,  # 90%时强制GC
                 check_interval: float = 5.0):  # 5秒检查一次
        """
        初始化内存管理器
        
        Args:
            memory_limit_mb: 内存限制（MB）
            warning_threshold: 警告阈值（0-1）
            force_gc_threshold: 强制GC阈值（0-1）
            check_interval: 检查间隔（秒）
        """
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.warning_threshold = warning_threshold
        self.force_gc_threshold = force_gc_threshold
        self.check_interval = check_interval
        
        self._running = False
        self._monitor_thread = None
        self._callbacks = []
        self._lock = threading.Lock()
        
        # 获取进程对象
        self._process = psutil.Process()
        
        # 缓存内存使用情况，减少psutil调用频率
        self._cached_usage = None
        self._cache_timestamp = 0
        self._cache_ttl = 1.0  # 缓存1秒
        
    def get_memory_usage(self) -> dict:
        """获取当前内存使用情况（带缓存，提升性能）"""
        import time
        
        current_time = time.time()
        
        # 使用缓存避免频繁的psutil调用
        if (self._cached_usage and 
            current_time - self._cache_timestamp < self._cache_ttl):
            return self._cached_usage
        
        try:
            memory_info = self._process.memory_info()
            system_memory = psutil.virtual_memory()
            
            usage = {
                'rss_bytes': memory_info.rss,  # 物理内存
                'vms_bytes': memory_info.vms,  # 虚拟内存
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': self._process.memory_percent(),
                'system_available_mb': system_memory.available / 1024 / 1024,
                'system_used_percent': system_memory.percent,
                'limit_mb': self.memory_limit_bytes / 1024 / 1024,
                'usage_ratio': memory_info.rss / self.memory_limit_bytes
            }
            
            # 更新缓存
            self._cached_usage = usage
            self._cache_timestamp = current_time
            
            return usage
        except Exception as e:
            print(f"Error getting memory usage: {e}")
            return self._cached_usage or {}
    
    def is_memory_critical(self) -> bool:
        """检查内存是否处于危险状态"""
        usage = self.get_memory_usage()
        if not usage:
            return False
        
        return usage.get('usage_ratio', 0) > self.force_gc_threshold
    
    def is_memory_warning(self) -> bool:
        """检查内存是否需要警告"""
        usage = self.get_memory_usage()
        if not usage:
            return False
        
        return usage.get('usage_ratio', 0) > self.warning_threshold
    
    def force_cleanup(self) -> dict:
        """强制清理内存"""
        before_usage = self.get_memory_usage()
        
        # 执行垃圾回收
        gc.collect()
        
        # 触发回调函数
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"Error in memory cleanup callback: {e}")
        
        after_usage = self.get_memory_usage()
        
        freed_mb = (before_usage.get('rss_mb', 0) - after_usage.get('rss_mb', 0))
        
        return {
            'before_mb': before_usage.get('rss_mb', 0),
            'after_mb': after_usage.get('rss_mb', 0),
            'freed_mb': freed_mb
        }
    
    def add_cleanup_callback(self, callback: Callable[[], None]):
        """添加内存清理回调函数"""
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_cleanup_callback(self, callback: Callable[[], None]):
        """移除内存清理回调函数"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def start_monitoring(self):
        """开始内存监控"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """内存监控循环"""
        while self._running:
            try:
                if self.is_memory_critical():
                    print("Memory usage critical, forcing cleanup...")
                    cleanup_result = self.force_cleanup()
                    print(f"Memory cleanup: freed {cleanup_result['freed_mb']:.1f}MB")
                elif self.is_memory_warning():
                    print("Memory usage warning, consider reducing buffer size")
                
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in memory monitor: {e}")
                time.sleep(self.check_interval)


class BatchProcessor:
    """批量处理器，用于内存安全的大数据处理"""
    
    def __init__(self, 
                 memory_manager: MemoryManager,
                 initial_batch_size: int = 10000,
                 min_batch_size: int = 1000,
                 max_batch_size: int = 100000):
        """
        初始化批量处理器
        
        Args:
            memory_manager: 内存管理器
            initial_batch_size: 初始批量大小
            min_batch_size: 最小批量大小
            max_batch_size: 最大批量大小
        """
        self.memory_manager = memory_manager
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        self._performance_history = []
        self._max_history = 10
    
    def get_optimal_batch_size(self) -> int:
        """根据内存使用情况动态调整批量大小"""
        usage = self.memory_manager.get_memory_usage()
        if not usage:
            return self.current_batch_size
        
        usage_ratio = usage.get('usage_ratio', 0)
        
        if usage_ratio > self.memory_manager.force_gc_threshold:
            # 内存危险，大幅减少批量大小
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 4
            )
        elif usage_ratio > self.memory_manager.warning_threshold:
            # 内存警告，适度减少批量大小
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
        elif usage_ratio < 0.5:
            # 内存充足，可以适度增加批量大小
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        
        return self.current_batch_size
    
    def process_in_batches(self, data_iterator, process_func: Callable, 
                          progress_callback: Optional[Callable] = None):
        """以批量方式处理数据"""
        processed_count = 0
        batch = []
        
        for item in data_iterator:
            batch.append(item)
            
            if len(batch) >= self.get_optimal_batch_size():
                # 处理当前批次
                try:
                    process_func(batch)
                    processed_count += len(batch)
                    
                    if progress_callback:
                        progress_callback(processed_count)
                    
                    # 检查内存并可能触发清理
                    if self.memory_manager.is_memory_warning():
                        self.memory_manager.force_cleanup()
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                
                batch = []
        
        # 处理剩余的数据
        if batch:
            try:
                process_func(batch)
                processed_count += len(batch)
                
                if progress_callback:
                    progress_callback(processed_count)
            except Exception as e:
                print(f"Error processing final batch: {e}")
        
        return processed_count


# 全局内存管理器实例
_global_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """获取全局内存管理器实例"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
        _global_memory_manager.start_monitoring()
    return _global_memory_manager


def set_memory_limit(limit_mb: int):
    """设置全局内存限制"""
    manager = get_memory_manager()
    manager.memory_limit_bytes = limit_mb * 1024 * 1024


def get_memory_usage_mb() -> float:
    """获取当前内存使用量（MB）"""
    manager = get_memory_manager()
    usage = manager.get_memory_usage()
    return usage.get('rss_mb', 0)


def force_memory_cleanup() -> dict:
    """强制执行内存清理"""
    manager = get_memory_manager()
    return manager.force_cleanup()
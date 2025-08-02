"""
平台特定的文件句柄管理器
专门处理Windows上的文件句柄未及时释放问题
"""

import platform
import time
import gc
from typing import Optional, Callable, Any
from pathlib import Path
import contextlib


class FileHandleManager:
    """平台特定的文件句柄管理器"""
    
    def __init__(self):
        self.is_windows = platform.system() == 'Windows'
        self.retry_count = 3 if self.is_windows else 1
        self.retry_delay = 0.01  # 10ms delay between retries
    
    @contextlib.contextmanager
    def safe_file_operation(self, file_path: Path, mode: str = 'rb'):
        """安全的文件操作上下文管理器
        
        Args:
            file_path: 文件路径
            mode: 文件打开模式
            
        Yields:
            文件对象
        """
        file_obj = None
        for attempt in range(self.retry_count):
            try:
                file_obj = open(file_path, mode)
                yield file_obj
                break
            except (OSError, IOError, PermissionError) as e:
                if file_obj:
                    try:
                        file_obj.close()
                    except:
                        pass
                    file_obj = None
                
                if attempt == self.retry_count - 1:
                    raise e
                
                # Windows特定的句柄释放策略
                if self.is_windows:
                    self._force_handle_release()
                    time.sleep(self.retry_delay * (attempt + 1))
            finally:
                if file_obj:
                    try:
                        file_obj.close()
                    except:
                        pass
                    
                # Windows特定的后处理
                if self.is_windows:
                    self._post_operation_cleanup()
    
    def safe_file_delete(self, file_path: Path) -> bool:
        """安全删除文件
        
        Args:
            file_path: 要删除的文件路径
            
        Returns:
            是否删除成功
        """
        if not file_path.exists():
            return True
            
        for attempt in range(self.retry_count):
            try:
                file_path.unlink()
                return True
            except (OSError, IOError, PermissionError) as e:
                if attempt == self.retry_count - 1:
                    # 在Windows上，如果删除失败，至少记录错误但不抛出异常
                    if self.is_windows:
                        print(f"Warning: Could not delete file {file_path}: {e}")
                        return False
                    else:
                        raise e
                
                # Windows特定的句柄释放策略
                if self.is_windows:
                    self._force_handle_release()
                    time.sleep(self.retry_delay * (attempt + 1))
        
        return False
    
    def _force_handle_release(self):
        """强制释放文件句柄（Windows特定）"""
        if self.is_windows:
            # 强制垃圾回收，释放未关闭的文件句柄
            gc.collect()
            # 额外的短暂等待，让操作系统释放句柄
            time.sleep(0.001)  # 1ms
    
    def _post_operation_cleanup(self):
        """操作后清理（Windows特定）"""
        if self.is_windows:
            # 强制垃圾回收
            gc.collect()


# 全局文件管理器实例
_file_manager = None

def get_file_manager() -> FileHandleManager:
    """获取全局文件管理器实例"""
    global _file_manager
    if _file_manager is None:
        _file_manager = FileHandleManager()
    return _file_manager


def safe_read_shard(file_path: Path) -> tuple[dict, dict]:
    """安全读取分片文件
    
    Args:
        file_path: 分片文件路径
        
    Returns:
        (meta_data, shard_data) 元组
    """
    import msgpack
    from pyroaring import BitMap
    
    fm = get_file_manager()
    
    with fm.safe_file_operation(file_path, 'rb') as f:
        meta_size = int.from_bytes(f.read(4), 'big')
        meta_data = f.read(meta_size)
        meta = msgpack.unpackb(meta_data, raw=False)
        
        shard_data = {}
        for term, (offset, size) in meta.items():
            f.seek(4 + meta_size + offset)
            bitmap_data = f.read(size)
            shard_data[term] = bitmap_data
    
    return meta, shard_data


def safe_write_shard(file_path: Path, meta: dict, data: dict) -> bool:
    """安全写入分片文件
    
    Args:
        file_path: 分片文件路径
        meta: 元数据
        data: 分片数据
        
    Returns:
        是否写入成功
    """
    import msgpack
    
    fm = get_file_manager()
    
    try:
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        with fm.safe_file_operation(file_path, 'wb') as f:
            meta_data = msgpack.packb(meta, use_bin_type=True)
            f.write(len(meta_data).to_bytes(4, 'big'))
            f.write(meta_data)
            for bitmap_data in data.values():
                f.write(bitmap_data)
        
        return True
    except Exception as e:
        print(f"Error writing shard {file_path}: {e}")
        return False


def safe_delete_shard(file_path: Path) -> bool:
    """安全删除分片文件
    
    Args:
        file_path: 分片文件路径
        
    Returns:
        是否删除成功
    """
    fm = get_file_manager()
    return fm.safe_file_delete(file_path)
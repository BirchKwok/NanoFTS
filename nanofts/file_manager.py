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
    """安全读取分片文件，支持损坏文件的恢复
    
    Args:
        file_path: 分片文件路径
        
    Returns:
        (meta_data, shard_data) 元组
    """
    import msgpack
    from pyroaring import BitMap
    
    fm = get_file_manager()
    
    # 先检查文件大小是否合理
    if not file_path.exists():
        return {}, {}
    
    file_size = file_path.stat().st_size
    if file_size < 4:  # 至少需要4字节存储元数据大小
        print(f"Warning: Shard file {file_path} is too small ({file_size} bytes), treating as empty")
        return {}, {}
    
    try:
        with fm.safe_file_operation(file_path, 'rb') as f:
            # 安全读取元数据大小
            try:
                meta_size_bytes = f.read(4)
                if len(meta_size_bytes) != 4:
                    raise ValueError("Incomplete meta size header")
                meta_size = int.from_bytes(meta_size_bytes, 'big')
            except Exception as e:
                print(f"Warning: Failed to read meta size from {file_path}: {e}")
                return {}, {}
            
            # 检查元数据大小是否合理
            if meta_size < 0 or meta_size > file_size - 4:
                print(f"Warning: Invalid meta size {meta_size} in {file_path} (file size: {file_size})")
                return {}, {}
            
            # 安全读取元数据
            try:
                meta_data = f.read(meta_size)
                if len(meta_data) != meta_size:
                    raise ValueError(f"Expected {meta_size} bytes of metadata, got {len(meta_data)}")
                meta = msgpack.unpackb(meta_data, raw=False)
            except Exception as e:
                print(f"Warning: Failed to unpack metadata from {file_path}: {e}")
                return {}, {}
            
            # 安全读取分片数据
            shard_data = {}
            for term, (offset, size) in meta.items():
                try:
                    actual_offset = 4 + meta_size + offset
                    if actual_offset + size > file_size:
                        print(f"Warning: Data for term '{term}' extends beyond file size in {file_path}")
                        continue
                    
                    f.seek(actual_offset)
                    bitmap_data = f.read(size)
                    if len(bitmap_data) != size:
                        print(f"Warning: Expected {size} bytes for term '{term}', got {len(bitmap_data)} in {file_path}")
                        continue
                    
                    # 验证bitmap数据是否可以正确反序列化
                    try:
                        BitMap.deserialize(bitmap_data)
                        shard_data[term] = bitmap_data
                    except Exception as bitmap_e:
                        print(f"Warning: Failed to deserialize bitmap for term '{term}' in {file_path}: {bitmap_e}")
                        continue
                        
                except Exception as e:
                    print(f"Warning: Failed to read data for term '{term}' in {file_path}: {e}")
                    continue
        
        return meta, shard_data
        
    except Exception as e:
        print(f"Error: Failed to read shard {file_path}: {e}")
        # 如果文件损坏，尝试备份并返回空数据
        backup_path = file_path.with_suffix('.corrupted')
        try:
            if not backup_path.exists():
                file_path.rename(backup_path)
                print(f"Corrupted file backed up to {backup_path}")
        except Exception as backup_e:
            print(f"Warning: Failed to backup corrupted file: {backup_e}")
        
        return {}, {}


def safe_write_shard(file_path: Path, meta: dict, data: dict) -> bool:
    """原子性安全写入分片文件
    
    Args:
        file_path: 分片文件路径
        meta: 元数据
        data: 分片数据
        
    Returns:
        是否写入成功
    """
    import msgpack
    import tempfile
    import os
    
    fm = get_file_manager()
    
    if not meta or not data:
        # 如果数据为空，删除现有文件
        return safe_delete_shard(file_path)
    
    try:
        file_path.parent.mkdir(exist_ok=True, parents=True)
        
        # 优化的写入策略：对于新文件直接写入，对于现有文件使用临时文件
        use_temp_file = file_path.exists()
        
        if use_temp_file:
            # 使用临时文件确保原子性
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='wb', 
                    dir=file_path.parent, 
                    prefix=f'.{file_path.name}.',
                    suffix='.tmp',
                    delete=False
                ) as f:
                    temp_file = f.name
                    
                    # 序列化并写入数据
                    meta_data = msgpack.packb(meta, use_bin_type=True)
                    f.write(len(meta_data).to_bytes(4, 'big'))
                    f.write(meta_data)
                    
                    for term in meta.keys():
                        if term in data:
                            f.write(data[term])
                    
                    f.flush()
                    if hasattr(os, 'fsync'):  # 某些系统可能不支持fsync
                        os.fsync(f.fileno())
                
                # 原子性移动
                if os.name == 'nt' and file_path.exists():
                    file_path.unlink()
                Path(temp_file).rename(file_path)
                
            except Exception as temp_e:
                # 清理临时文件
                if temp_file and Path(temp_file).exists():
                    try:
                        Path(temp_file).unlink()
                    except:
                        pass
                raise temp_e
        else:
            # 新文件直接写入，提升性能
            with fm.safe_file_operation(file_path, 'wb') as f:
                meta_data = msgpack.packb(meta, use_bin_type=True)
                f.write(len(meta_data).to_bytes(4, 'big'))
                f.write(meta_data)
                
                for term in meta.keys():
                    if term in data:
                        f.write(data[term])
                
                f.flush()
            
        # 轻量级验证：仅检查文件大小
        try:
            if file_path.stat().st_size < 8:
                print(f"Warning: Written file {file_path} appears to be too small")
                return False
        except Exception as verify_e:
            print(f"Warning: Failed to verify written file {file_path}: {verify_e}")
            return False
        
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
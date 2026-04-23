#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字幕结果缓存模块

实现视频与字幕结果的关联存储，避免重复调用豆包API
支持文件哈希验证和过期时间检查
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SubtitleCache:
    """
    字幕结果缓存管理类
    """
    
    def __init__(self, cache_dir: Optional[str] = None, expire_days: int = 30):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存存储目录，默认在项目目录下创建 .subtitle_cache
            expire_days: 缓存过期天数，默认30天
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / ".subtitle_cache"
        
        self.cache_dir = Path(cache_dir)
        self.expire_days = expire_days
        self.cache_index_file = self.cache_dir / "cache_index.json"
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载缓存索引
        self.cache_index = self._load_cache_index()
    
    def _get_file_hash(self, file_path: str) -> str:
        """
        计算文件的SHA256哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件哈希值字符串
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"计算文件哈希失败: {e}")
            # 如果文件读取失败，使用文件名+修改时间作为备选
            try:
                stat = os.stat(file_path)
                backup_str = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
                return hashlib.sha256(backup_str.encode()).hexdigest()
            except:
                return hashlib.sha256(file_path.encode()).hexdigest()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """
        加载缓存索引文件
        
        Returns:
            缓存索引字典
        """
        if not self.cache_index_file.exists():
            return {"version": "1.0", "entries": {}}
        
        try:
            with open(self.cache_index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 清理过期缓存
                self._clean_expired_cache(data.get("entries", {}))
                return data
        except Exception as e:
            logger.warning(f"加载缓存索引失败: {e}")
            return {"version": "1.0", "entries": {}}
    
    def _save_cache_index(self):
        """保存缓存索引到文件"""
        try:
            with open(self.cache_index_file, "w", encoding="utf-8") as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def _clean_expired_cache(self, entries: Dict[str, Any]):
        """
        清理过期的缓存
        
        Args:
            entries: 缓存条目字典
        """
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, entry in entries.items():
            try:
                save_time = datetime.fromisoformat(entry.get("save_time", ""))
                if current_time - save_time > timedelta(days=self.expire_days):
                    keys_to_remove.append(key)
                    # 删除缓存文件
                    cache_file = self.cache_dir / entry.get("cache_file", "")
                    if cache_file.exists():
                        cache_file.unlink()
            except Exception as e:
                logger.warning(f"检查缓存过期失败: {e}")
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del entries[key]
    
    def _get_cache_key(self, file_path: str, song_index: Optional[int] = None, 
                      language: str = "zh-CN", caption_type: str = "auto") -> str:
        """
        生成缓存键
        
        Args:
            file_path: 视频/音频文件路径
            song_index: 歌曲索引（可选，用于区分同一视频中的不同歌曲）
            language: 语言类型
            caption_type: 字幕识别类型
            
        Returns:
            缓存键字符串
        """
        file_hash = self._get_file_hash(file_path)
        if song_index is not None:
            key_str = f"{file_hash}_song{song_index}_{language}_{caption_type}"
        else:
            key_str = f"{file_hash}_{language}_{caption_type}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_cached_subtitle(self, file_path: str, song_index: Optional[int] = None,
                         language: str = "zh-CN", caption_type: str = "auto") -> Optional[Dict[str, Any]]:
        """
        获取缓存的字幕结果
        
        Args:
            file_path: 视频/音频文件路径
            song_index: 歌曲索引（可选）
            language: 语言类型
            caption_type: 字幕识别类型
            
        Returns:
            缓存的字幕结果，无缓存返回None
        """
        cache_key = self._get_cache_key(file_path, song_index, language, caption_type)
        entries = self.cache_index.get("entries", {})
        
        if cache_key not in entries:
            logger.info(f"无缓存记录: {Path(file_path).name}" + (f" 歌曲{song_index+1}" if song_index is not None else ""))
            return None
        
        entry = entries[cache_key]
        
        # 检查缓存是否过期
        try:
            save_time = datetime.fromisoformat(entry.get("save_time", ""))
            if datetime.now() - save_time > timedelta(days=self.expire_days):
                logger.info(f"缓存已过期: {Path(file_path).name}" + (f" 歌曲{song_index+1}" if song_index is not None else ""))
                # 移除过期缓存
                cache_file = self.cache_dir / entry.get("cache_file", "")
                if cache_file.exists():
                    cache_file.unlink()
                del entries[cache_key]
                self._save_cache_index()
                return None
        except Exception as e:
            logger.warning(f"检查缓存过期失败: {e}")
        
        # 读取缓存文件
        cache_file = self.cache_dir / entry.get("cache_file", "")
        if not cache_file.exists():
            logger.warning(f"缓存文件不存在: {cache_file}")
            del entries[cache_key]
            self._save_cache_index()
            return None
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            logger.info(f"命中缓存: {Path(file_path).name}" + (f" 歌曲{song_index+1}" if song_index is not None else ""))
            return result
        except Exception as e:
            logger.error(f"读取缓存文件失败: {e}")
            return None
    
    def save_subtitle(self, file_path: str, subtitle_result: Dict[str, Any],
                     song_index: Optional[int] = None,
                     language: str = "zh-CN", caption_type: str = "auto"):
        """
        保存字幕结果到缓存
        
        Args:
            file_path: 视频/音频文件路径
            subtitle_result: 字幕结果字典
            song_index: 歌曲索引（可选）
            language: 语言类型
            caption_type: 字幕识别类型
        """
        cache_key = self._get_cache_key(file_path, song_index, language, caption_type)
        cache_file_name = f"cache_{cache_key}.json"
        cache_file_path = self.cache_dir / cache_file_name
        
        try:
            # 保存缓存文件
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(subtitle_result, f, ensure_ascii=False, indent=2)
            
            # 更新索引
            entries = self.cache_index.get("entries", {})
            entries[cache_key] = {
                "file_path": str(Path(file_path).resolve()),
                "file_name": Path(file_path).name,
                "song_index": song_index,
                "language": language,
                "caption_type": caption_type,
                "cache_file": cache_file_name,
                "save_time": datetime.now().isoformat(),
                "result_count": len(subtitle_result.get("sentences", []))
            }
            
            self._save_cache_index()
            logger.info(f"字幕缓存已保存: {Path(file_path).name}" + (f" 歌曲{song_index+1}" if song_index is not None else ""))
            
        except Exception as e:
            logger.error(f"保存字幕缓存失败: {e}")
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        清理缓存
        
        Args:
            older_than_days: 只清理早于指定天数的缓存，None表示全部清理
        """
        if older_than_days is None:
            # 清空所有缓存
            for cache_file in self.cache_dir.glob("cache_*.json"):
                cache_file.unlink()
            self.cache_index = {"version": "1.0", "entries": {}}
            self._save_cache_index()
            logger.info("所有缓存已清理")
        else:
            # 清理指定天数前的缓存
            self.expire_days = older_than_days
            self._clean_expired_cache(self.cache_index.get("entries", {}))
            self._save_cache_index()
            logger.info(f"已清理 {older_than_days} 天前的缓存")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        entries = self.cache_index.get("entries", {})
        total_size = 0
        for entry in entries.values():
            cache_file = self.cache_dir / entry.get("cache_file", "")
            if cache_file.exists():
                total_size += cache_file.stat().st_size
        
        return {
            "total_entries": len(entries),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir.resolve()),
            "expire_days": self.expire_days
        }


# 全局缓存管理器实例
_global_cache: Optional[SubtitleCache] = None


def get_subtitle_cache() -> SubtitleCache:
    """
    获取全局字幕缓存管理器单例
    
    Returns:
        SubtitleCache实例
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = SubtitleCache()
    return _global_cache

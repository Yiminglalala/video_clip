"""
双模型分类器 v9.0
彻底抛弃手工规则，用 SOTA 深度学习模型进行分类
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MusicClassifier:
    """
    音乐结构分类器
    
    使用两个模型：
    - AST: 场景分类（讲话/鼓掌/观众合唱/音乐）
    - MERT: 结构分类（前奏/主歌/副歌/间奏/独奏/尾奏）
    """
    
    def __init__(self, gpu_processor: 'GPUProcessor', config: 'GPUConfig'):
        self.gpu = gpu_processor
        self.config = config
        
        # 场景标签映射（AST）
        self.SCENE_LABELS = ['speech', 'clap', 'crowd', 'music']
        
        # 结构标签映射（MERT）
        self.STRUCT_LABELS = ['intro', 'verse', 'chorus', 'bridge', 'solo', 'outro']
    
    def predict(self, 
                audio_path: str,
                vocal_path: Optional[str],
                boundaries: List[Tuple[float, float]],
                global_stats: Optional[Dict] = None) -> List[Dict]:
        """
        对每个歌曲片段进行分类
        
        Args:
            audio_path: 原始音频路径
            vocal_path: Demucs 分离后的人声路径
            boundaries: 歌曲边界列表 [(start, end), ...]
            global_stats: 全局统计量（可选）
        
        Returns:
            [
                {
                    'start': 0.0,
                    'end': 30.0,
                    'scene': 'music',
                    'scene_confidence': 0.95,
                    'structure': 'chorus',
                    'structure_confidence': 0.88
                },
                ...
            ]
        """
        import librosa
        
        results = []
        
        for i, (start, end) in enumerate(boundaries):
            # 加载音频片段
            try:
                y, sr = librosa.load(audio_path, sr=44100, offset=start, duration=end-start)
            except Exception as e:
                logger.warning(f"加载音频片段失败: {e}")
                continue
            
            # ═══════════════════════════════════════════════════════════════
            # Step 1: AST 场景分类
            # ═══════════════════════════════════════════════════════════════
            if self.config.ENABLE_AST and vocal_path:
                try:
                    y_voc, sr_voc = librosa.load(vocal_path, sr=44100, offset=start, duration=end-start)
                    scene_probs = self.gpu.classify_scene(y_voc, sr_voc)
                except Exception as e:
                    logger.warning(f"AST 场景分类失败: {e}")
                    scene_probs = {'music': 1.0}
            else:
                scene_probs = self._rule_based_scene(y, sr)
            
            scene_label = max(scene_probs, key=scene_probs.get)
            scene_confidence = scene_probs[scene_label]
            
            # ═══════════════════════════════════════════════════════════════
            # Step 2: MERT 结构分类
            # ═══════════════════════════════════════════════════════════════
            if self.config.ENABLE_MERT:
                structure_label, structure_confidence = self._mert_structure_classify(
                    audio_path, start, end
                )
            else:
                # 降级到规则分类
                structure_label, structure_confidence = self._rule_based_structure(
                    y, sr, start, end, global_stats
                )
            
            results.append({
                'start': start,
                'end': end,
                'duration': end - start,
                'scene': scene_label,
                'scene_confidence': scene_confidence,
                'structure': structure_label,
                'structure_confidence': structure_confidence,
            })
        
        return results
    
    def _mert_structure_classify(self, 
                                  audio_path: str, 
                                  start: float, 
                                  end: float) -> Tuple[str, float]:
        """
        使用 MERT 进行结构分类
        """
        # 提取 MERT 嵌入
        embeddings = self.gpu.extract_mert_embeddings(audio_path)
        
        if embeddings is None:
            return 'verse', 0.5
        
        # 简化：基于嵌入特征的 K-means 聚类
        # 实际应该用预训练的结构分类头
        try:
            from sklearn.cluster import KMeans
            
            # 采样嵌入（每 1 秒取一帧）
            fps = 50  # MERT 默认帧率
            start_frame = int(start * fps)
            end_frame = int(end * fps)
            
            if end_frame > start_frame + 5:
                segment_emb = embeddings[start_frame:end_frame:5]  # 每5帧取1个
                
                # K-means 聚类为 3 类
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                labels = kmeans.fit_predict(segment_emb)
                
                # 统计各簇的大小
                unique, counts = np.unique(labels, return_counts=True)
                dominant_cluster = unique[np.argmax(counts)]
                
                # 映射到结构标签（简化版）
                # 能量高的簇 -> chorus
                # 能量低的簇 -> intro/outro
                # 中等能量 -> verse
                energy_per_cluster = []
                for c in unique:
                    cluster_emb = segment_emb[labels == c]
                    energy_per_cluster.append(np.mean(np.abs(cluster_emb)))
                
                max_energy_cluster = unique[np.argmax(energy_per_cluster)]
                
                if dominant_cluster == max_energy_cluster:
                    return 'chorus', 0.75
                elif start / 60 < 1:  # 开头 1 分钟
                    return 'intro', 0.6
                else:
                    return 'verse', 0.6
            else:
                return 'verse', 0.5
                
        except Exception as e:
            logger.warning(f"MERT 结构分类失败: {e}")
            return 'verse', 0.5
    
    def _rule_based_scene(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        降级：基于规则的场景分类
        """
        # 能量
        rms = np.sqrt(np.mean(y**2))
        
        # 过零率
        zcr = np.mean(librosa.feature.zero_crossing_rate(y, sr=sr))
        
        # 频谱质心
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 简化的场景判断
        if rms < 0.01:
            return {'speech': 0.1, 'clap': 0.1, 'crowd': 0.1, 'music': 0.7}
        elif zcr > 0.15 and centroid < 2000:
            # 高过零率 + 低质心 = 鼓掌/观众
            return {'speech': 0.1, 'clap': 0.4, 'crowd': 0.4, 'music': 0.1}
        elif zcr < 0.08 and rms > 0.05:
            # 低过零率 + 高能量 = 音乐
            return {'speech': 0.1, 'clap': 0.1, 'crowd': 0.1, 'music': 0.7}
        else:
            # 混合场景
            return {'speech': 0.2, 'clap': 0.2, 'crowd': 0.2, 'music': 0.4}
    
    def _rule_based_structure(self, 
                              y: np.ndarray, 
                              sr: int,
                              start: float,
                              end: float,
                              global_stats: Optional[Dict]) -> Tuple[str, float]:
        """
        降级：基于规则的结构分类
        """
        duration = end - start
        
        # 能量
        rms = np.sqrt(np.mean(y**2))
        rms_mean = global_stats.get('rms_mean', 0.05) if global_stats else 0.05
        relative_rms = rms / rms_mean
        
        # 频谱质心
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        centroid_mean = global_stats.get('centroid_mean', 2000) if global_stats else 2000
        
        # 位置
        position_ratio = start / max(end, 1)
        
        # 简化判断
        if position_ratio < 0.1:
            return 'intro', 0.6
        elif position_ratio > 0.9:
            return 'outro', 0.6
        elif relative_rms > 1.2:
            return 'chorus', 0.65
        elif relative_rms < 0.8 and centroid < centroid_mean * 0.9:
            return 'bridge', 0.5
        else:
            return 'verse', 0.6
    
    def merge_scene_structure(self, segments: List[Dict]) -> List[Dict]:
        """
        合并场景和结构标签为最终标签
        
        映射规则：
        scene='music' + structure='chorus' -> 副歌
        scene='music' + structure='verse' -> 主歌
        scene='music' + structure='intro' -> 前奏
        scene='music' + structure='bridge' -> 间奏
        scene='music' + structure='outro' -> 尾奏
        scene='music' + structure='solo' -> 独奏
        scene='crowd' -> 观众合唱
        scene='clap' -> 掌声欢呼
        scene='speech' -> 讲话
        """
        LABEL_MAP = {
            ('music', 'chorus'): 'chorus',
            ('music', 'verse'): 'verse',
            ('music', 'intro'): 'intro',
            ('music', 'bridge'): 'interlude',
            ('music', 'outro'): 'outro',
            ('music', 'solo'): 'solo',
            ('crowd', 'music'): 'audience',
            ('clap', 'music'): 'crowd',
            ('speech', 'music'): 'speech',
        }
        
        for seg in segments:
            key = (seg['scene'], seg['structure'])
            seg['label'] = LABEL_MAP.get(key, 'other')
            # 综合置信度
            seg['confidence'] = (seg['scene_confidence'] + seg['structure_confidence']) / 2
        
        return segments

"""
GPU 核心处理模块 v9.0
封装所有 SOTA 深度学习模型（Demucs/MERT/AST）
"""

import os
import logging
import tempfile
from typing import Dict, Tuple, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 全局模型缓存
_model_cache = {}


def get_device() -> str:
    """自动检测可用设备"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class GPUProcessor:
    """
    GPU 核心处理器
    封装所有深度学习模型
    """
    
    def __init__(self, config: 'GPUConfig'):
        self.config = config
        self.device = config.DEVICE if config.ENABLE_GPU else "cpu"
        self._demucs = None
        self._mert = None
        self._mert_processor = None
        self._ast = None
        self._ast_processor = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Demucs 人声分离
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_demucs(self):
        """加载 Demucs 模型"""
        if self._demucs is not None:
            return self._demucs
        
        try:
            import torch
            from demucs import pretrained
            
            logger.info(f"加载 Demucs 模型: {self.config.DEMUCS_MODEL}, device={self.device}")
            self._demucs = pretrained.get_model(self.config.DEMUCS_MODEL)
            self._demucs.load_state_dict(torch.load(
                self.config.DEMUCS_WEIGHTS,
                map_location=self.device
            ))
            self._demucs = self._demucs.to(self.device)
            self._demucs.eval()
            logger.info("Demucs 加载成功")
            return self._demucs
        except Exception as e:
            logger.warning(f"Demucs 加载失败: {e}")
            return None
    
    def separate_audio(self, audio_path: str) -> Dict[str, str]:
        """
        人声分离
        
        Returns:
            {
                'vocals': '/tmp/vocals.wav',
                'accompaniment': '/tmp/accompaniment.wav',
                'drums': '/tmp/drums.wav',
                'bass': '/tmp/bass.wav',
                'other': '/tmp/other.wav'
            }
        """
        if not self.config.ENABLE_DEMUCS:
            logger.info("Demucs 未启用，跳过分离")
            return {'original': audio_path}
        
        model = self.load_demucs()
        if model is None:
            return {'original': audio_path}
        
        try:
            import torch
            import torchaudio as T
            
            # 加载音频
            waveform, sr = T.load(audio_path)
            if sr != 44100:
                waveform = T.resample(waveform, sr, 44100)
            
            # 移动到 GPU
            if self.device != "cpu":
                waveform = waveform.to(self.device)
            
            # 分离
            with torch.no_grad():
                # 标准化到 [-1, 1]
                if waveform.abs().max() > 1:
                    waveform = waveform / waveform.abs().max()
                # 扩展 batch 维度
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                elif waveform.dim() == 2 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Demucs 输出: [batch, sources, samples]
                ref = waveform.mean(dim=(1, 2), keepdim=True)
                sources = model(waveform / ref.abs().max()) * ref.abs().max()
                sources = sources.squeeze(0).cpu().numpy()
            
            # 保存分离后的轨道
            result = {}
            source_names = ['drums', 'bass', 'other', 'vocals']  # 默认顺序
            
            out_dir = tempfile.mkdtemp()
            
            # 保存各轨道
            for i, name in enumerate(source_names):
                if i < len(sources):
                    out_path = os.path.join(out_dir, f"{name}.wav")
                    source_waveform = torch.from_numpy(sources[i]).unsqueeze(0)
                    T.save(source_waveform, out_path, sample_rate=44100)
                    result[name] = out_path
            
            # 合并伴奏（drums + bass + other）
            acc_path = os.path.join(out_dir, "accompaniment.wav")
            acc_waveform = torch.from_numpy(sources[:3].sum(axis=0)).unsqueeze(0)
            T.save(acc_waveform, acc_path, sample_rate=44100)
            result['accompaniment'] = acc_path
            
            logger.info(f"Demucs 分离完成: {list(result.keys())}")
            return result
            
        except Exception as e:
            logger.error(f"Demucs 分离失败: {e}")
            return {'original': audio_path}
    
    # ═══════════════════════════════════════════════════════════════════════
    # MERT 音乐结构分析
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_mert(self):
        """加载 MERT 模型"""
        if self._mert is not None:
            return self._mert, self._mert_processor
        
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            
            logger.info(f"加载 MERT 模型: {self.config.MERT_MODEL}")
            self._mert_processor = AutoFeatureExtractor.from_pretrained(self.config.MERT_MODEL)
            self._mert = AutoModel.from_pretrained(
                self.config.MERT_MODEL,
                trust_remote_code=True
            )
            self._mert = self._mert.to(self.device)
            self._mert.eval()
            logger.info("MERT 加载成功")
            return self._mert, self._mert_processor
        except Exception as e:
            logger.warning(f"MERT 加载失败: {e}")
            return None, None
    
    def extract_mert_embeddings(self, audio_path: str) -> Optional[np.ndarray]:
        """
        提取 MERT 嵌入序列
        
        Returns:
            embeddings: (T, 768) 时间帧 x 嵌入维度
        """
        if not self.config.ENABLE_MERT:
            return None
        
        model, processor = self.load_mert()
        if model is None:
            return None
        
        try:
            import librosa
            import torch
            
            # 加载音频
            y, sr = librosa.load(audio_path, sr=24000, mono=True)
            
            # 提取特征
            inputs = processor(
                y, 
                sampling_rate=24000,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()
            
            logger.info(f"MERT 嵌入形状: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"MERT 特征提取失败: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # AST 场景分类
    # ═══════════════════════════════════════════════════════════════════════
    
    def load_ast(self):
        """加载 AST 音频分类模型"""
        if self._ast is not None:
            return self._ast, self._ast_processor
        
        try:
            from transformers import AutoModel, AutoFeatureExtractor
            
            logger.info(f"加载 AST 模型: {self.config.AST_MODEL}")
            self._ast_processor = AutoFeatureExtractor.from_pretrained(self.config.AST_MODEL)
            self._ast = AutoModel.from_pretrained(self.config.AST_MODEL)
            self._ast = self._ast.to(self.device)
            self._ast.eval()
            logger.info("AST 加载成功")
            return self._ast, self._ast_processor
        except Exception as e:
            logger.warning(f"AST 加载失败: {e}")
            return None, None
    
    def classify_scene(self, audio_segment: np.ndarray, sr: int = 44100) -> Dict[str, float]:
        """
        场景分类（讲话/鼓掌/观众合唱/音乐）
        
        Returns:
            {
                'speech': 0.1,
                'clap': 0.8,
                'crowd': 0.05,
                'music': 0.05
            }
        """
        if not self.config.ENABLE_AST:
            return {'music': 1.0}
        
        model, processor = self.load_ast()
        if model is None:
            return {'music': 1.0}
        
        try:
            import torch
            
            # 提取特征
            inputs = processor(
                audio_segment,
                sampling_rate=sr,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # AST 输出是 logits，需要映射到类别
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state.mean(dim=1)
                probs = torch.softmax(logits, dim=-1)
            
            # 简化：返回固定映射（实际需要根据 AST 类别映射）
            result = probs.squeeze(0).cpu().numpy()
            return {
                'speech': float(result[0]) if len(result) > 0 else 0.1,
                'clap': float(result[1]) if len(result) > 1 else 0.1,
                'crowd': float(result[2]) if len(result) > 2 else 0.1,
                'music': float(result[3]) if len(result) > 3 else 0.8,
            }
            
        except Exception as e:
            logger.error(f"AST 分类失败: {e}")
            return {'music': 1.0}
    
    # ═══════════════════════════════════════════════════════════════════════
    # 显存清理
    # ═══════════════════════════════════════════════════════════════════════
    
    def cleanup(self):
        """清理显存"""
        if self.device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
                logger.info("GPU 显存已清理")
            except:
                pass


def create_gpu_processor(config: 'GPUConfig') -> GPUProcessor:
    """工厂函数：创建 GPU 处理器"""
    return GPUProcessor(config)

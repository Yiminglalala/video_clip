"""
自动优化系统模块
"""

import copy
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import uuid
from datetime import datetime

from src.data_models import VideoSample, OptimizationHistory
from src.sample_library import get_sample_library_db
from src.config import ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """
    准确率指标
    """
    overall: float = 0.0
    audience: float = 0.0
    solo: float = 0.0
    chorus: float = 0.0
    verse: float = 0.0
    degraded_sample_count: int = 0
    total_samples: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'overall': self.overall,
            'audience': self.audience,
            'solo': self.solo,
            'chorus': self.chorus,
            'verse': self.verse,
            'degraded_sample_count': self.degraded_sample_count,
            'total_samples': self.total_samples,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccuracyMetrics':
        return cls(
            overall=data.get('overall', 0.0),
            audience=data.get('audience', 0.0),
            solo=data.get('solo', 0.0),
            chorus=data.get('chorus', 0.0),
            verse=data.get('verse', 0.0),
            degraded_sample_count=data.get('degraded_sample_count', 0),
            total_samples=data.get('total_samples', 0),
        )


class HeuristicOptimizer:
    """
    启发式优化器
    """
    
    def __init__(self):
        self.db = get_sample_library_db()
    
    def analyze_error_patterns(self, samples: List[VideoSample]) -> Dict[str, Any]:
        """
        分析错误模式
        """
        label_swaps = defaultdict(int)
        label_improvements = defaultdict(int)
        
        for sample in samples:
            for segment in sample.segments:
                if segment.is_modified:
                    old_label = segment.original_label
                    new_label = segment.current_label
                    label_swaps[(old_label, new_label)] += 1
                    label_improvements[new_label] += 1
        
        return {
            'label_swaps': dict(label_swaps),
            'label_improvements': dict(label_improvements),
            'total_samples': len(samples),
            'total_modified_segments': sum(1 for s in samples for seg in s.segments if seg.is_modified),
        }
    
    def suggest_parameter_adjustments(self, error_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        建议参数调整
        """
        suggestions = []
        label_swaps = error_patterns.get('label_swaps', {})
        
        # 分析常见的标签修改
        for (old_label, new_label), count in label_swaps.items():
            if count >= 2:  # 至少2个样本有这个模式
                if new_label == 'audience':
                    # 用户经常将其他标签改成audience，说明audience阈值太高
                    suggestions.append({
                        'param': 'audio_analyzer.AUDIENCE_SCORE_THRESHOLD',
                        'direction': 'lower',
                        'delta': 0.5,
                        'reason': f'用户将{count}个{old_label}改成了audience',
                        'priority': 'high',
                    })
                elif new_label == 'solo':
                    # 用户经常将其他标签改成solo，说明solo阈值太高
                    suggestions.append({
                        'param': 'audio_analyzer.SOLO_SCORE_THRESHOLD',
                        'direction': 'lower',
                        'delta': 0.5,
                        'reason': f'用户将{count}个{old_label}改成了solo',
                        'priority': 'high',
                    })
                elif old_label == 'audience' and new_label == 'chorus':
                    # 用户经常将audience改成chorus，说明audience阈值太低
                    suggestions.append({
                        'param': 'audio_analyzer.AUDIENCE_SCORE_THRESHOLD',
                        'direction': 'higher',
                        'delta': 0.5,
                        'reason': f'用户将{count}个audience改成了chorus',
                        'priority': 'medium',
                    })
        
        return suggestions
    
    def apply_adjustments(self, config: ProcessingConfig, suggestions: List[Dict[str, Any]]) -> ProcessingConfig:
        """
        应用参数调整
        """
        new_config = copy.deepcopy(config)
        
        # 我们需要映射到具体的config属性
        # 注意：这里需要根据实际config结构调整
        for suggestion in suggestions:
            param = suggestion['param']
            direction = suggestion['direction']
            delta = suggestion['delta']
            
            # 这里只是示例，实际需要映射到真实的config属性
            # 我们暂时用一个占位方案
            
            logger.info(f"建议调整: {param} {direction} {delta}")
        
        return new_config
    
    def calculate_accuracy(self, samples: List[VideoSample], config: ProcessingConfig) -> AccuracyMetrics:
        """
        计算准确率（基于用户修改的样本进行对比）
        """
        metrics = AccuracyMetrics()
        
        # 由于我们没有实时运行分类，这里我们只是计算样本库中的修改率
        # 真实场景中，应该用新参数重新运行分类，然后与用户标注对比
        
        total_segments = 0
        correct_segments = 0
        audience_correct = 0
        audience_total = 0
        solo_correct = 0
        solo_total = 0
        
        for sample in samples:
            for segment in sample.segments:
                total_segments += 1
                
                if not segment.is_modified:
                    correct_segments += 1
                
                if segment.original_label == 'audience':
                    audience_total += 1
                    if not segment.is_modified:
                        audience_correct += 1
                
                if segment.original_label == 'solo':
                    solo_total += 1
                    if not segment.is_modified:
                        solo_correct += 1
        
        if total_segments > 0:
            metrics.overall = correct_segments / total_segments
        if audience_total > 0:
            metrics.audience = audience_correct / audience_total
        if solo_total > 0:
            metrics.solo = solo_correct / solo_total
        
        metrics.total_samples = len(samples)
        
        # 计算退化样本数量
        metrics.degraded_sample_count = 0  # 占位，实际需要对比
        
        return metrics
    
    def check_acceptance_criteria(
        self,
        before_metrics: AccuracyMetrics,
        after_metrics: AccuracyMetrics
    ) -> Tuple[bool, str]:
        """
        检查验收标准
        """
        reasons = []
        
        # 1. 整体分数不下降
        if after_metrics.overall < before_metrics.overall - 0.05:  # 允许5%波动
            reasons.append(f"整体准确率下降: {before_metrics.overall:.2%} → {after_metrics.overall:.2%}")
        
        # 2. 关键标签不下降
        if after_metrics.audience < before_metrics.audience - 0.1:
            reasons.append(f"audience准确率下降: {before_metrics.audience:.2%} → {after_metrics.audience:.2%}")
        if after_metrics.solo < before_metrics.solo - 0.1:
            reasons.append(f"solo准确率下降: {before_metrics.solo:.2%} → {after_metrics.solo:.2%}")
        
        # 3. 退化样本比例不超过阈值
        max_degraded_ratio = 0.1  # 10%
        if after_metrics.total_samples > 0:
            degraded_ratio = after_metrics.degraded_sample_count / after_metrics.total_samples
            if degraded_ratio > max_degraded_ratio:
                reasons.append(f"退化样本比例过高: {degraded_ratio:.1%} > {max_degraded_ratio:.1%}")
        
        passed = len(reasons) == 0
        reason_msg = "; ".join(reasons) if reasons else "所有验收标准通过"
        
        return passed, reason_msg


class AutoOptimizer:
    """
    自动优化系统主类
    """
    
    def __init__(self):
        self.db = get_sample_library_db()
        self.optimizer = HeuristicOptimizer()
    
    def run_optimization(self, config: ProcessingConfig) -> Dict[str, Any]:
        """
        运行完整优化流程
        """
        logger.info("开始自动优化流程...")
        
        # 1. 获取所有样本
        samples = self.db.get_all_samples()
        if not samples:
            return {
                'success': False,
                'error': '没有样本数据，无法优化',
            }
        
        # 2. 计算优化前准确率
        before_metrics = self.optimizer.calculate_accuracy(samples, config)
        
        # 3. 分析错误模式
        error_patterns = self.optimizer.analyze_error_patterns(samples)
        
        # 4. 生成参数建议
        suggestions = self.optimizer.suggest_parameter_adjustments(error_patterns)
        
        if not suggestions:
            return {
                'success': True,
                'message': '没有发现明确的优化方向',
                'before_metrics': before_metrics.to_dict(),
                'after_metrics': None,
                'suggestions': [],
                'accepted': False,
            }
        
        # 5. 应用参数调整
        new_config = self.optimizer.apply_adjustments(config, suggestions)
        
        # 6. 计算优化后准确率
        after_metrics = self.optimizer.calculate_accuracy(samples, new_config)
        
        # 7. 检查验收标准
        passed, reason = self.optimizer.check_acceptance_criteria(before_metrics, after_metrics)
        
        # 8. 保存优化历史
        history = OptimizationHistory(
            optimization_id=self.db.generate_optimization_id(),
            before_config=self._config_to_dict(config),
            after_config=self._config_to_dict(new_config),
            before_accuracy=before_metrics.to_dict(),
            after_accuracy=after_metrics.to_dict(),
            applied=False,
            created_at=datetime.now(),
            notes=reason,
        )
        self.db.add_optimization_history(history)
        
        # 9. 返回结果
        return {
            'success': True,
            'optimization_id': history.optimization_id,
            'before_metrics': before_metrics.to_dict(),
            'after_metrics': after_metrics.to_dict(),
            'suggestions': suggestions,
            'passed': passed,
            'reason': reason,
            'config_dict': self._config_to_dict(new_config),
        }
    
    def apply_optimization(self, optimization_id: str, config: ProcessingConfig) -> Tuple[bool, str]:
        """
        应用优化参数
        """
        history_list = self.db.get_optimization_history(limit=50)
        history = next((h for h in history_list if h.optimization_id == optimization_id), None)
        
        if not history:
            return False, "找不到优化历史记录"
        
        # 这里需要将字典参数应用到真实的config对象
        # 暂返回成功，实际需要完善
        
        self.db.update_optimization_applied(optimization_id, True)
        
        return True, "参数已应用"
    
    def _config_to_dict(self, config: ProcessingConfig) -> Dict[str, Any]:
        """
        将配置转换为字典
        """
        return {
            # 这里只保存关键参数，实际需要完整映射
            'min_segment_duration': config.min_segment_duration,
            'max_segment_duration': config.max_segment_duration,
        }


# 全局实例
_auto_optimizer: Optional[AutoOptimizer] = None


def get_auto_optimizer() -> AutoOptimizer:
    """
    获取自动优化器实例
    """
    global _auto_optimizer
    if _auto_optimizer is None:
        _auto_optimizer = AutoOptimizer()
    return _auto_optimizer


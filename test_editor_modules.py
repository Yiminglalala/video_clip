"""
测试新创建的编辑器模块
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def test_data_models():
    """
    测试数据模型
    """
    print("📋 测试数据模型...")
    
    from src.data_models import EditableSegment
    
    # 测试 EditableSegment
    seg = EditableSegment(
        segment_id="test_001",
        start_time=0.0,
        end_time=10.0,
        original_label="verse",
        current_label="verse",
        confidence=0.9
    )
    
    assert seg.duration == 10.0
    assert seg.original_confidence == 0.9
    
    # 测试修改
    seg.current_label = "chorus"
    seg.is_modified = True
    
    # 测试序列化
    d = seg.to_dict()
    assert d['segment_id'] == "test_001"
    assert d['is_modified'] is True
    
    # 测试反序列化
    seg2 = EditableSegment.from_dict(d)
    assert seg2.current_label == "chorus"
    
    print("✅ 数据模型测试通过！")
    return True


def test_sample_library():
    """
    测试样本库
    """
    print("📚 测试样本库...")
    
    from src.sample_library import get_sample_library_db
    
    db = get_sample_library_db()
    
    # 测试生成ID
    sample_id = db.generate_sample_id()
    assert sample_id.startswith("sample_")
    
    opt_id = db.generate_optimization_id()
    assert opt_id.startswith("opt_")
    
    print("✅ 样本库测试通过！")
    return True


def test_auto_optimizer():
    """
    测试自动优化器
    """
    print("⚙️ 测试自动优化器...")
    
    from src.auto_optimizer import AccuracyMetrics
    
    # 测试准确率指标
    metrics = AccuracyMetrics()
    metrics.overall = 0.85
    metrics.audience = 0.78
    metrics.solo = 0.70
    
    d = metrics.to_dict()
    assert d['overall'] == 0.85
    assert d['audience'] == 0.78
    
    print("✅ 自动优化器测试通过！")
    return True


def test_preview_editor():
    """
    测试预览编辑器
    """
    print("🎬 测试预览编辑器...")
    
    from src.preview_editor import get_preview_editor
    
    editor = get_preview_editor()
    # 简单测试实例化
    assert editor is not None
    
    print("✅ 预览编辑器测试通过！")
    return True


def main():
    """
    主测试函数
    """
    print("="*50)
    print("🧪 开始测试视频编辑系统模块")
    print("="*50)
    
    tests = [
        ("数据模型", test_data_models),
        ("样本库", test_sample_library),
        ("自动优化器", test_auto_optimizer),
        ("预览编辑器", test_preview_editor),
    ]
    
    all_passed = True
    for name, test_func in tests:
        try:
            print(f"\n{'='*30}")
            print(f"测试: {name}")
            print(f"{'='*30}")
            passed = test_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("❌ 部分测试失败")
    print("="*50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


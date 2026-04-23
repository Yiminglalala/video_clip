"""
视频片段编辑与管理系统 - Streamlit UI
"""

import streamlit as st
import logging
import sys
from pathlib import Path

# 添加项目根目录到PATH
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# 导入现有模块
from src.preview_editor import get_preview_editor
from src.sample_library import get_sample_library_db
from src.auto_optimizer import get_auto_optimizer
from src.data_models import VideoSample, EditableSegment

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 标签颜色映射
LABEL_COLORS = {
    'verse': '#4CAF50',
    'chorus': '#2196F3',
    'intro': '#FFC107',
    'outro': '#9C27B0',
    'audience': '#FF5722',
    'solo': '#00BCD4',
    'talk': '#795548',
    'crowd': '#607D8B',
    'silence': '#EEEEEE',
    'other': '#9E9E9E',
}


class VideoEditorUI:
    """
    视频编辑器UI主类
    """
    
    def __init__(self):
        self.editor = get_preview_editor()
        self.db = get_sample_library_db()
        self.optimizer = get_auto_optimizer()
    
    def run(self):
        """
        运行UI
        """
        st.set_page_config(
            page_title="视频片段编辑器",
            page_icon="🎬",
            layout="wide",
        )
        
        st.title("🎬 视频片段编辑与管理系统")
        
        # 侧边栏导航
        page = st.sidebar.radio(
            "功能导航",
            ["编辑器", "样本库", "参数优化"]
        )
        
        if page == "编辑器":
            self._show_editor_page()
        elif page == "样本库":
            self._show_sample_library_page()
        elif page == "参数优化":
            self._show_optimization_page()
    
    def _show_editor_page(self):
        """
        显示编辑器页面
        """
        st.header("视频片段编辑器")
        
        # 模拟加载视频（实际需要集成到现有流程）
        st.info("💡 提示：视频分析完成后会自动跳转到编辑页面")
        
        # 模拟数据展示（示例）
        self._show_mock_editor()
    
    def _show_mock_editor(self):
        """
        显示模拟编辑器（演示用）
        """
        st.subheader("📋 编辑区域")
        
        # 模拟片段数据
        mock_segments = [
            {
                "id": "seg_001",
                "start": 0.0,
                "end": 15.5,
                "original_label": "intro",
                "current_label": "intro",
                "confidence": 0.92,
                "modified": False
            },
            {
                "id": "seg_002",
                "start": 15.5,
                "end": 45.0,
                "original_label": "verse",
                "current_label": "verse",
                "confidence": 0.88,
                "modified": False
            },
            {
                "id": "seg_003",
                "start": 45.0,
                "end": 70.0,
                "original_label": "chorus",
                "current_label": "audience",
                "confidence": 0.75,
                "modified": True
            },
        ]
        
        # 显示时间轴（简化版）
        st.subheader("⏱️ 时间轴（简化版）")
        self._show_simple_timeline(mock_segments)
        
        # 显示片段列表
        st.subheader("📝 片段列表")
        self._show_segment_list(mock_segments)
        
        # 操作按钮
        st.subheader("✅ 操作")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 保存为样本", type="primary"):
                st.success("样本保存功能（待实现）")
        
        with col2:
            if st.button("↩️ 重置所有"):
                st.info("重置功能（待实现）")
        
        with col3:
            if st.button("🎬 确认输出", type="primary"):
                st.success("确认输出功能（待实现）")
        
        # 统计信息
        self._show_statistics(mock_segments)
    
    def _show_simple_timeline(self, segments):
        """
        显示简化时间轴
        """
        total_duration = max(seg['end'] for seg in segments)
        
        # 创建一个简单的时间轴可视化
        timeline_html = f"""
        <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
            <div style="font-weight: bold; margin-bottom: 10px;">总时长: {total_duration:.1f} 秒</div>
        """
        
        for seg in segments:
            color = LABEL_COLORS.get(seg['current_label'], '#9E9E9E')
            label = seg['current_label']
            width = (seg['end'] - seg['start']) / total_duration * 100
            
            modified_mark = " 📝" if seg['modified'] else ""
            
            timeline_html += f"""
            <div style="
                display: inline-block;
                height: 60px;
                width: {width}%;
                background-color: {color};
                color: white;
                text-align: center;
                line-height: 60px;
                font-weight: bold;
                margin: 0;
                border: 1px solid rgba(0,0,0,0.2);
            ">
                {label}{modified_mark}
            </div>
            """
        
        timeline_html += "</div>"
        st.components.v1.html(timeline_html, height=100)
    
    def _show_segment_list(self, segments):
        """
        显示片段列表
        """
        # 显示每个片段的详细信息
        for idx, seg in enumerate(segments):
            with st.expander(f"片段 {idx+1}: {seg['start']:.1f}s - {seg['end']:.1f}s"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # 标签选择
                    available_labels = ['verse', 'chorus', 'intro', 'outro', 'audience', 'solo', 'talk', 'crowd', 'silence', 'other']
                    new_label = st.selectbox(
                        f"标签",
                        available_labels,
                        index=available_labels.index(seg['current_label']) if seg['current_label'] in available_labels else 0,
                        key=f"label_{seg['id']}"
                    )
                    
                    if new_label != seg['current_label']:
                        seg['current_label'] = new_label
                        seg['modified'] = True
                
                with col2:
                    # 时间调整
                    new_start = st.number_input(
                        f"开始时间",
                        value=seg['start'],
                        min_value=0.0,
                        step=0.5,
                        key=f"start_{seg['id']}"
                    )
                    new_end = st.number_input(
                        f"结束时间",
                        value=seg['end'],
                        min_value=new_start,
                        step=0.5,
                        key=f"end_{seg['id']}"
                    )
                    
                    if new_start != seg['start'] or new_end != seg['end']:
                        seg['start'] = new_start
                        seg['end'] = new_end
                        seg['modified'] = True
                
                st.info(f"原始标签: {seg['original_label']} | 置信度: {seg['confidence']:.2%}")
                
                if seg['modified']:
                    st.warning("⚠️ 此片段已修改")
    
    def _show_statistics(self, segments):
        """
        显示统计信息
        """
        total = len(segments)
        modified = sum(1 for seg in segments if seg['modified'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("总片段数", total)
        
        with col2:
            st.metric("已修改片段数", modified)
        
        # 标签分布
        label_counts = {}
        for seg in segments:
            label = seg['current_label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        st.subheader("📊 标签分布")
        st.bar_chart(label_counts)
    
    def _show_sample_library_page(self):
        """
        显示样本库页面
        """
        st.header("📚 样本库管理")
        
        # 获取样本列表
        samples = self.db.get_all_samples()
        
        if not samples:
            st.info("暂未保存任何样本，去编辑器保存样本吧！")
        else:
            # 显示统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("样本总数", len(samples))
            
            # 显示样本列表
            st.subheader("📋 样本列表")
            for sample in samples:
                with st.expander(f"{sample.sample_id} - {Path(sample.video_path).name}"):
                    st.write(f"创建时间: {sample.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write(f"视频时长: {sample.video_duration:.1f} 秒")
                    st.write(f"片段数: {len(sample.segments)}")
                    st.write(f"修改率: {sample.modification_rate:.1%}")
                    
                    # 删除按钮
                    if st.button(f"删除", key=f"delete_{sample.sample_id}"):
                        if self.db.delete_sample(sample.sample_id):
                            st.success(f"样本已删除: {sample.sample_id}")
                            st.rerun()
                        else:
                            st.error("删除失败")
    
    def _show_optimization_page(self):
        """
        显示优化页面
        """
        st.header("⚙️ 参数自动优化")
        
        # 检查样本数量
        sample_count = self.db.get_sample_count()
        if sample_count == 0:
            st.info("需要至少1个样本才能进行优化，先去样本库保存一些样本吧！")
            return
        
        st.info(f"📊 样本库共有 {sample_count} 个样本")
        
        # 优化按钮
        if st.button("🚀 运行优化", type="primary"):
            with st.spinner("正在运行优化..."):
                # 模拟优化
                result = self._run_mock_optimization()
                
                if result['success']:
                    st.success("优化完成！")
                    self._show_optimization_result(result)
                else:
                    st.error(f"优化失败: {result.get('error', '未知错误')}")
        
        # 优化历史
        st.subheader("📜 优化历史")
        histories = self.db.get_optimization_history()
        if histories:
            for history in histories:
                with st.expander(f"{history.optimization_id} - {history.created_at.strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.write(f"是否应用: {'✅' if history.applied else '❌'}")
                    st.write(f"优化前准确率: {history.before_accuracy.get('overall', 0):.1%}")
                    st.write(f"优化后准确率: {history.after_accuracy.get('overall', 0):.1%}")
                    
                    if not history.applied:
                        if st.button(f"应用此参数", key=f"apply_{history.optimization_id}"):
                            st.success("参数应用功能（待实现）")
        else:
            st.info("暂无优化历史")
    
    def _run_mock_optimization(self):
        """
        模拟优化
        """
        return {
            'success': True,
            'optimization_id': self.db.generate_optimization_id(),
            'before_metrics': {
                'overall': 0.75,
                'audience': 0.60,
                'solo': 0.55,
            },
            'after_metrics': {
                'overall': 0.85,
                'audience': 0.78,
                'solo': 0.70,
            },
            'suggestions': [
                {
                    'param': 'audio_analyzer.AUDIENCE_SCORE_THRESHOLD',
                    'direction': 'lower',
                    'delta': 0.5,
                    'reason': '用户将audience识别率提升了18%',
                    'priority': 'high',
                }
            ],
            'passed': True,
            'reason': '所有验收标准通过',
        }
    
    def _show_optimization_result(self, result):
        """
        显示优化结果
        """
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("优化前")
            st.metric("整体准确率", f"{result['before_metrics']['overall']:.1%}")
            st.metric("audience准确率", f"{result['before_metrics']['audience']:.1%}")
            st.metric("solo准确率", f"{result['before_metrics']['solo']:.1%}")
        
        with col2:
            st.subheader("优化后")
            st.metric("整体准确率", f"{result['after_metrics']['overall']:.1%}", 
                      delta=f"{result['after_metrics']['overall']-result['before_metrics']['overall']:.1%}")
            st.metric("audience准确率", f"{result['after_metrics']['audience']:.1%}",
                      delta=f"{result['after_metrics']['audience']-result['before_metrics']['audience']:.1%}")
            st.metric("solo准确率", f"{result['after_metrics']['solo']:.1%}",
                      delta=f"{result['after_metrics']['solo']-result['before_metrics']['solo']:.1%}")
        
        # 建议的参数调整
        st.subheader("💡 建议的参数调整")
        for suggestion in result['suggestions']:
            st.info(f"""
                **参数**: {suggestion['param']}  
                **调整方向**: {suggestion['direction']}  
                **调整量**: {suggestion['delta']}  
                **原因**: {suggestion['reason']}
            """)
        
        # 验收结果
        if result['passed']:
            st.success("✅ 验收通过！")
        else:
            st.warning(f"⚠️ 验收未通过: {result['reason']}")
        
        # 应用按钮
        if result['passed']:
            if st.button("✅ 一键应用推荐参数", type="primary"):
                st.success("参数应用功能（待实现）")


def main():
    ui = VideoEditorUI()
    ui.run()


if __name__ == "__main__":
    main()


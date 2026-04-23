import os
import sys
import tempfile
import shutil

sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.audio_analyzer import separate_vocals

# 测试音频
audio_path = 'D:/video_clip/output/demucs_test/vocals.wav'
print(f'Testing separate_vocals with: {audio_path}')

# 创建临时输出目录
tmpdir = tempfile.mkdtemp(prefix="dmucs_test_")
print(f'Output dir: {tmpdir}')

try:
    vocals_path, no_vocals_path = separate_vocals(
        audio_path,
        out_dir=tmpdir,
        model='htdemucs_ft',
        device='cuda'
    )

    print(f'\nResult:')
    print(f'  vocals_path: {vocals_path}')
    print(f'  no_vocals_path: {no_vocals_path}')

    if vocals_path and os.path.exists(vocals_path):
        import soundfile as sf
        vwav, vsr = sf.read(vocals_path, dtype='float32')
        print(f'  vocals: {vwav.shape}, sr={vsr}, max={vwav.max():.4f}')
        print(f'  vocals length: {len(vwav)/vsr:.2f}s')
    else:
        print('  ERROR: vocals.wav not found!')

    if no_vocals_path and os.path.exists(no_vocals_path):
        import soundfile as sf
        nwav, nsr = sf.read(no_vocals_path, dtype='float32')
        print(f'  no_vocals: {nwav.shape}, sr={nsr}, max={nwav.max():.4f}')
    else:
        print('  ERROR: no_vocals.wav not found!')

finally:
    # 清理
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
        print(f'\nCleaned up: {tmpdir}')
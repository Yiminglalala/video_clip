import os
import sys
import tempfile
import shutil
import torch
import soundfile as sf

sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.audio_analyzer import separate_vocals
import logging
logging.basicConfig(level=logging.INFO)

audio_path = 'D:/video_clip/output/demucs_test/vocals.wav'
print(f'Testing with: {audio_path}')

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
        vwav, vsr = sf.read(vocals_path, dtype='float32')
        print(f'  vocals shape: {vwav.shape}')
        print(f'  vocals sr: {vsr}')
        print(f'  vocals max: {vwav.max()}')
        print(f'  vocals length: {len(vwav)/vsr:.2f}s')
    else:
        print('  ERROR: vocals.wav not found!')

finally:
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
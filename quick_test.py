# -*- coding: utf-8 -*-
"""快速验证：1.7B 加载+单段推理"""
import os, sys, time, torch, traceback

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

log = open(r"D:\video_clip\output\lyric_aligned\quick_test.log", 'w', encoding='utf-8')
def lp(s):
    print(s); log.write(str(s)+'\n'); log.flush()

lp(f"[{time.strftime('%H:%M:%S')}] Start")

# 加载
lp(f"[{time.strftime('%H:%M:%S')}] Loading...")
sys.stdout.flush()
from qwen_asr import Qwen3ASRModel
import librosa
import numpy as np
from scipy.signal import butter, sosfilt

t0 = time.time()
model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", device_map="cuda:0")
lp(f"[{time.strftime('%H:%M:%S')}] Model loaded in {time.time()-t0:.1f}s")
lp(f"  GPU alloc={torch.cuda.memory_allocated(0)/1024**3:.1f}GB")

# 用已有的音频片段
seg_wav = r"D:\video_clip\output\lyric_aligned\seg_00.wav"
if not os.path.exists(seg_wav):
    # 提取前18秒
    VIDEO_PATH = r'D:\video_clip\output\A-Lin_声音梦境线上音乐会_20260412_1157\Song_01\Song_01_副歌_01m12s-01m41s.mp4'
    subprocess = __import__('subprocess')
    subprocess.run(['ffmpeg','-y','-i',VIDEO_PATH,'-t','18',
                    '-acodec','pcm_s16le','-ar','16000','-ac','1', seg_wav],
                   capture_output=True, check=True)
    lp(f"  Extracted audio to {seg_wav}")

lp(f"[{time.strftime('%H:%M:%S')}] Loading audio...")
y, sr = librosa.load(seg_wav, sr=16000, mono=True)
lp(f"  Audio shape={y.shape}, duration={len(y)/sr:.1f}s")

# 预处理
sos = butter(4, 100, btype='high', fs=sr, output='sos')
y_proc = sosfilt(sos, y)
peak = np.max(np.abs(y_proc))
if peak > 0:
    y_proc = y_proc * 0.9 / peak
y_proc = y_proc.astype(np.float32)

# 推理
lp(f"[{time.strftime('%H:%M:%S')}] Transcribing (this may take a while)...")
sys.stdout.flush()
t1 = time.time()
try:
    result = model.transcribe(audio=(y_proc, 16000))
    t2 = time.time()
    lp(f"[{time.strftime('%H:%M:%S')}] Done in {t2-t1:.1f}s! RTF={(t2-t1)/(len(y)/sr):.2f}")
    if isinstance(result, dict):
        text = result.get('text', '')
        lang = result.get('language', '?')
    elif hasattr(result, 'text'):
        text = result.text; lang = '?'
    else:
        text = str(result); lang = '?'
    lp(f"  [{lang}] \"{text[:100]}\"")
except Exception as e:
    lp(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc(file=log)

lp(f"\nGPU final: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB")
lp("DONE.")
log.close()

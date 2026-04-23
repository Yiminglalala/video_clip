"""Fun-ASR 测试 - 对比 Qwen3 和 Whisper"""
import os
os.environ["HF_HOME"] = r"D:\video_clip\.cache\huggingface"

from funasr import AutoModel
import librosa
import numpy as np
from scipy.signal import butter, sosfilt

AUDIO = r"D:\video_clip\output\lyric_aligned\audio.wav"

# Preprocess
y, sr = librosa.load(AUDIO, sr=16000, mono=True)
sos = butter(4, 100, btype='high', fs=sr, output='sos')
y = sosfilt(sos, y)
peak = np.max(np.abs(y))
if peak > 0: y = y * 0.9 / peak
audio_np = y.astype(np.float32)

# Use the correct stable model name
print("Loading FunASR Paraformer-large + VAD + PUNC...")
model = AutoModel(
    model="paraformer-zh",           # Paraformer large 中文语音识别
    vad_model="fsmn-vad",            # 语音活动检测（VAD）
    punc_model="ct-punc-c",          # 标点恢复
    device="cuda:0",
)

print("Transcribing...")
res = model.generate(
    input=audio_np,
    cache={},
    language="zh",
    use_itn=True,
    batch_size_s=300,
    hotword="",  # 可以在这里加热词，比如歌名"荒唐"
)

print("\n=== FunASR Result ===")
print(f"Raw result type: {type(res)}, len={len(res)}")
if res:
    print(f"First item keys: {list(res[0].keys()) if isinstance(res[0], dict) else type(res[0])}")

for i, seg in enumerate(res):
    text = seg.get('text', str(seg)) if isinstance(seg, dict) else str(seg)
    ts = seg.get('timestamp', None) if isinstance(seg, dict) else None
    if ts is not None:
        # ts might be list of [start_ms, end_ms] or nested
        try:
            start_s = float(ts[0]) / 1000.0  # FunASR uses milliseconds
            end_s = float(ts[1]) / 1000.0
            print(f"  [{i:02d}] [{start_s:.2f}-{end_s:.2f}s] {text}")
        except:
            print(f"  [{i:02d}] [ts={ts}] {text}")
    else:
        print(f"  [{i:02d}] {text}")

# Save raw result
import json
with open(r'D:\video_clip\output\lyric_aligned\funasr_result.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print(f"\nSaved ({len(res)} segments)")

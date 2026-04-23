import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
print(f'CUDA available: {torch.cuda.is_available()}')

import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model

model = get_model('htdemucs_ft')
model.to('cuda')
model.eval()
print(f'Model sources: {model.sources}')
print(f'Model samplerate: {model.samplerate}')

audio_path = 'D:/video_clip/output/demucs_test/vocals.wav'
print(f'Testing with: {audio_path}')

wav, sr = sf.read(audio_path, dtype='float32')
print(f'Audio (samples, channels): {wav.shape}, sr={sr}')

# 截取前10秒测试
max_samples = 10 * sr
if wav.shape[0] > max_samples:
    wav = wav[:max_samples]
    print(f'Truncated to: {wav.shape}')

import librosa
if sr != model.samplerate:
    wav = librosa.resample(wav.T, orig_sr=sr, target_sr=model.samplerate).T
print(f'Resampled: {wav.shape}')

wav = torch.from_numpy(wav).float()
wav = wav.T  # (samples, channels) -> (channels, samples)
if wav.shape[0] == 1:
    wav = wav.repeat(2, 1)
wav = wav.unsqueeze(0).to('cuda')
print(f'Tensor (batch, channels, samples): {wav.shape}')

with torch.no_grad():
    # 禁用shifts减少内存使用
    out = apply_model(model, wav, device='cuda', shifts=0, split=True, overlap=0.25, progress=False, segment=7)

print(f'Output type: {type(out)}, shape: {out.shape}')

vocals_idx = model.sources.index('vocals')
vocals = out[0, vocals_idx]
print(f'Vocals shape: {vocals.shape}')

import tempfile
tmpdir = tempfile.mkdtemp()
mono = vocals.mean(dim=0).cpu().numpy()
vpath = os.path.join(tmpdir, 'vocals_new.wav')
sf.write(vpath, mono, model.samplerate)
print(f'Saved vocals to: {vpath}')
print(f'Vocals file exists: {os.path.exists(vpath)}')
print(f'Vocals length: {len(mono)/model.samplerate:.2f}s')
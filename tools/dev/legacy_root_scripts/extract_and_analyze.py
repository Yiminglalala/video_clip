import os
import sys
import tempfile
import subprocess

# 切视频2的45-56秒的音频
video2 = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (17).mp4"
output_audio = r"D:\video_clip\output\video2_45_56.wav"

print(f"切割视频: {video2}")
print(f"时间段: 45-56秒")
print(f"输出: {output_audio}")

# 使用ffmpeg切音频
cmd = [
    "ffmpeg", "-y",
    "-ss", "45",
    "-t", "11",
    "-i", video2,
    "-ar", "22050",
    "-ac", "1",
    "-acodec", "pcm_s16le",
    "-f", "wav",
    output_audio
]

print(f"\n执行命令: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"\n✅ 切割成功!")
    print(f"输出文件: {output_audio}")
    
    # 检查文件大小
    size_mb = os.path.getsize(output_audio) / (1024 * 1024)
    print(f"文件大小: {size_mb:.2f} MB")
else:
    print(f"\n❌ 切割失败!")
    print(f"错误: {result.stderr}")
    sys.exit(1)

# 现在分析这个音频的特征
print(f"\n" + "="*80)
print(f"开始分析音频特征...")
print(f"="*80)

import numpy as np
import librosa

y, sr = librosa.load(output_audio, sr=22050, mono=True)
print(f"\n音频信息:")
print(f"  时长: {len(y)/sr:.2f} 秒")
print(f"  采样率: {sr} Hz")
print(f"  样本数: {len(y)}")
print(f"  dtype: {y.dtype}")
print(f"  min: {y.min():.4f}, max: {y.max():.4f}, mean: {y.mean():.4f}, std: {y.std():.4f}")

# 计算一些基础特征
print(f"\n基础特征:")

# RMS能量
rms = librosa.feature.rms(y=y)[0]
print(f"  RMS: mean={rms.mean():.4f}, std={rms.std():.4f}")

# 过零率
zcr = librosa.feature.zero_crossing_rate(y=y)[0]
print(f"  ZCR: mean={zcr.mean():.4f}, std={zcr.std():.4f}")

# 频谱质心
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
print(f"  Centroid: mean={centroid.mean():.1f} Hz")

# 频谱平坦度
flatness = librosa.feature.spectral_flatness(y=y)[0]
print(f"  Flatness: mean={flatness.mean():.4f}")

# 节拍强度
try:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strength = onset_env.mean()
    beat_regularity = 1.0 - onset_env.std() / (onset_env.mean() + 1e-6)
    beat_regularity = max(0.0, min(1.0, beat_regularity))
    print(f"  Beat strength: {beat_strength:.4f}")
    print(f"  Beat regularity: {beat_regularity:.4f}")
except Exception as e:
    print(f"  Beat features: failed - {e}")

# 谐波比
try:
    harmonic = librosa.effects.harmonic(y)
    harmonic_ratio = np.sqrt(np.mean(harmonic**2)) / (np.sqrt(np.mean(y**2)) + 1e-8)
    print(f"  Harmonic ratio: {harmonic_ratio:.4f}")
except Exception as e:
    print(f"  Harmonic ratio: failed - {e}")

# 音调强度
try:
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_strength = magnitudes.max(axis=0).mean()
    pitch_strength_norm = pitch_strength / (np.abs(y).max() + 1e-6)
    print(f"  Pitch strength: {pitch_strength_norm:.4f}")
except Exception as e:
    print(f"  Pitch strength: failed - {e}")

# Chroma熵
try:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    ch_entropies = []
    for i in range(chroma.shape[1]):
        ch = chroma[:, i]
        ch_norm = ch / (ch.sum() + 1e-8)
        ch_norm = ch_norm[ch_norm > 0.01]
        if len(ch_norm) > 0:
            ce = -np.sum(ch_norm * np.log2(ch_norm + 1e-10))
            ch_entropies.append(ce)
    if ch_entropies:
        chroma_entropy = np.mean(ch_entropies)
        print(f"  Chroma entropy: {chroma_entropy:.4f}")
except Exception as e:
    print(f"  Chroma entropy: failed - {e}")

print(f"\n" + "="*80)
print(f"分析完成!")
print(f"="*80)
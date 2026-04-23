import os
import sys

sys.path.insert(0, 'D:/video_clip')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from src.processor import LiveVideoProcessor, ProcessingConfig

config = ProcessingConfig(
    output_dir=r"D:\video_clip\output\test_demucs_e2e",
    enable_demucs=True,
    demucs_model="htdemucs_ft",
    enable_songformer=True,
    songformer_device="cuda",
    min_segment_duration=8.0,
    max_segment_duration=15.0,
)

print(f"Config: enable_demucs={config.enable_demucs}, demucs_model={config.demucs_model}")

processor = LiveVideoProcessor(config)
print(f"Processor audio_analyzer: {processor.audio_analyzer}")
print(f"Processor audio_analyzer config enable_demucs: {processor.config.enable_demucs}")

input_video = "D:/video_clip/input/zhou_shen_test.mp4"
print(f"Processing: {input_video}")

try:
    result, output_files = processor.process_video(input_video)
    print(f"\nResult type: {type(result)}")
    print(f"Analysis result: {result}")
    print(f"\nSongs found: {len(result.songs)}")
    for i, song in enumerate(result.songs):
        print(f"  Song {i+1}: {song.song_name}")
        print(f"    Segments: {len(song.segments)}")
        verse_count = 0
        chorus_count = 0
        for seg in song.segments:
            label = getattr(seg, 'label', 'unknown')
            print(f"      - {label} ({seg.start_time:.1f}s - {seg.end_time:.1f}s)")
            if 'verse' in label.lower() or '主歌' in label:
                verse_count += 1
            if 'chorus' in label.lower() or '副歌' in label:
                chorus_count += 1
        print(f"    Summary: {verse_count} verses, {chorus_count} choruses")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
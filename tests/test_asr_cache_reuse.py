# -*- coding: utf-8 -*-

import tempfile
import unittest
import wave
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.audio_analyzer import Segment
from src.processor import LiveVideoProcessor, ProcessingConfig


class AsrCacheReuseTests(unittest.TestCase):
    def _processor(self, output_dir: str) -> LiveVideoProcessor:
        processor = LiveVideoProcessor.__new__(LiveVideoProcessor)
        processor.config = ProcessingConfig(
            output_dir=output_dir,
            enable_subtitle=True,
            enable_songformer=False,
            strict_songformer=False,
            enable_demucs=False,
        )
        processor._cached_asr_results = {}
        processor._songformer_analyzer = None
        processor.audio_analyzer = None
        processor._log_gpu_memory = lambda *_args, **_kwargs: None
        processor._cleanup_gpu_stage = lambda *_args, **_kwargs: None
        processor._attach_song_continuity_signatures = lambda songs, *_args, **_kwargs: songs
        processor._merge_split_songs = lambda songs: songs
        processor._identify_song_from_lyrics = lambda *_args, **_kwargs: None
        processor._compute_song_stats = lambda *_args, **_kwargs: {}
        processor._analyze_song_segments = lambda *_args, **_kwargs: [
            Segment(
                start_time=0.0,
                end_time=20.0,
                label="verse",
                confidence=0.8,
                song_index=0,
                songformer_label="verse",
            )
        ]
        return processor

    def _write_test_wav(self, path: Path, duration_sec: float = 20.0, sr: int = 16000) -> None:
        samples = np.zeros(int(duration_sec * sr), dtype=np.int16)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes(samples.tobytes())

    def test_reuses_existing_subtitle_cache_without_calling_doubao(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "video.mp4")
            Path(video_path).write_bytes(b"fake-video")
            audio_path = str(Path(tmp) / "audio.wav")
            self._write_test_wav(Path(audio_path))
            processor = self._processor(tmp)
            cached_result = {
                "engine": "doubao",
                "text": "cached lyrics",
                "sentences": [{"start": 0.0, "end": 1.0, "text": "cached lyrics"}],
                "words": [],
            }

            with patch("src.processor.get_subtitle_cache") as cache_factory:
                cache = cache_factory.return_value
                cache.get_cached_subtitle.return_value = cached_result
                with patch.object(processor, "_run_doubao_asr", side_effect=AssertionError("should not call doubao")):
                    songs = processor._create_songs(
                        [(0.0, 20.0)],
                        audio_path=audio_path,
                        progress_callback=None,
                        singer="singer",
                        source_video_path=video_path,
                    )

            self.assertEqual(len(songs), 1)
            self.assertEqual(processor._cached_asr_results[0], cached_result)
            cache.save_subtitle.assert_not_called()

    def test_saves_subtitle_cache_after_doubao_call_when_cache_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "video.mp4")
            Path(video_path).write_bytes(b"fake-video")
            audio_path = str(Path(tmp) / "audio.wav")
            self._write_test_wav(Path(audio_path))
            processor = self._processor(tmp)
            asr_result = {
                "engine": "doubao",
                "text": "new lyrics",
                "sentences": [{"start": 0.0, "end": 1.0, "text": "new lyrics"}],
                "words": [],
            }

            with patch("src.processor.get_subtitle_cache") as cache_factory:
                cache = cache_factory.return_value
                cache.get_cached_subtitle.return_value = None
                with patch.object(processor, "_run_doubao_asr", return_value=asr_result) as run_asr:
                    processor._create_songs(
                        [(0.0, 20.0)],
                        audio_path=audio_path,
                        progress_callback=None,
                        singer="singer",
                        source_video_path=video_path,
                    )

            run_asr.assert_called_once()
            cache.save_subtitle.assert_called_once()
            self.assertEqual(processor._cached_asr_results[0], asr_result)

    def test_reuses_legacy_timestamped_asr_cache_and_migrates_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = str(Path(tmp) / "video.mp4")
            Path(video_path).write_bytes(b"fake-video")
            audio_path = str(Path(tmp) / "audio.wav")
            self._write_test_wav(Path(audio_path))
            processor = self._processor(tmp)
            cached_result = {
                "engine": "doubao",
                "text": "legacy cached lyrics",
                "sentences": [{"start": 0.0, "end": 1.0, "text": "legacy cached lyrics"}],
                "words": [],
            }

            legacy_dir = Path(tmp) / "asr_cache" / "20260520_120000_video"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "song_001.json").write_text(
                json.dumps({"asr_result": cached_result}, ensure_ascii=False),
                encoding="utf-8",
            )
            (legacy_dir / "index.json").write_text(
                json.dumps(
                    {
                        "video_path": video_path,
                        "songs": [{"song_index": 0, "file": "song_001.json"}],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch("src.processor.get_subtitle_cache") as cache_factory:
                cache = cache_factory.return_value
                cache.get_cached_subtitle.return_value = None
                with patch.object(processor, "_run_doubao_asr", side_effect=AssertionError("should not call doubao")):
                    processor._create_songs(
                        [(0.0, 20.0)],
                        audio_path=audio_path,
                        progress_callback=None,
                        singer="singer",
                        source_video_path=video_path,
                    )

            cache.save_subtitle.assert_called_once()
            self.assertEqual(processor._cached_asr_results[0], cached_result)


if __name__ == "__main__":
    unittest.main()

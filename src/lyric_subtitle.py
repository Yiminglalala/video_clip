# -*- coding: utf-8 -*-
"""
歌词字幕增强链路

主流程：
1. 优先使用用户确认的歌名/歌手或粘贴/上传的 LRC / 纯文本歌词
2. 免费歌词源抓取失败时，不再把 Shazam 当主路径硬猜
3. 按用户选择的歌词起始行生成标准歌词字幕时间轴
4. WhisperX / ASR 只作为可选辅助定位，不作为最终歌词文本来源
"""

from __future__ import annotations

import contextlib
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "output" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DB = CACHE_DIR / "lyrics_cache.sqlite3"

_WHISPERX_MODELS: Dict[str, object] = {}
_ALIGN_MODELS: Dict[str, tuple] = {}
logger = logging.getLogger(__name__)
_SONG_ID_MISS_CACHE: Dict[str, float] = {}
_ACR_BLOCKED_CREDENTIALS: Dict[str, float] = {}


@dataclass
class SongMatch:
    title: str
    artist: str
    album: str = ""
    provider: str = "shazam"
    confidence: float = 0.0
    track_id: str = ""
    raw: Optional[dict] = None


@dataclass
class SongHint:
    title: str = ""
    artist: str = ""
    lyrics_text: str = ""
    start_line: int = 0
    end_line: Optional[int] = None
    clip_start_sec: Optional[float] = None
    offset_sec: float = 0.0


@dataclass
class LyricLine:
    index: int
    text: str
    start: Optional[float] = None
    end: Optional[float] = None


def _init_cache() -> None:
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lyrics_cache (
                cache_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS song_id_cache (
                audio_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                title TEXT NOT NULL,
                artist TEXT NOT NULL,
                album TEXT NOT NULL,
                confidence REAL NOT NULL,
                track_id TEXT NOT NULL,
                payload TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text)
    return text


def _normalize_song_key(title: str, artist: str) -> str:
    return f"{_normalize_text(title)}::{_normalize_text(artist)}"


def _audio_cache_key(audio_path: str) -> str:
    try:
        st = os.stat(audio_path)
        size = int(st.st_size)
        head = b""
        tail = b""
        with open(audio_path, "rb") as f:
            head = f.read(65536)
            if size > 65536:
                f.seek(max(0, size - 65536))
                tail = f.read(65536)
        h = hashlib.sha1()
        h.update(str(size).encode("utf-8"))
        h.update(head)
        h.update(tail)
        return h.hexdigest()
    except Exception:
        return ""


def _acr_credential_fingerprint(host: str, access_key: str, access_secret: str) -> str:
    raw = f"{host}|{access_key}|{access_secret}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _get_cached_song_match(audio_key: str) -> Optional[SongMatch]:
    if not audio_key:
        return None
    _init_cache()
    conn = sqlite3.connect(CACHE_DB)
    try:
        row = conn.execute(
            """
            SELECT provider, title, artist, album, confidence, track_id, payload
            FROM song_id_cache WHERE audio_key = ?
            """,
            (audio_key,),
        ).fetchone()
        if not row:
            return None
        provider, title, artist, album, confidence, track_id, payload = row
        raw = None
        if payload:
            try:
                raw = json.loads(payload)
            except Exception:
                raw = None
        return SongMatch(
            title=str(title or ""),
            artist=str(artist or ""),
            album=str(album or ""),
            provider=str(provider or "cache"),
            confidence=float(confidence or 0.0),
            track_id=str(track_id or ""),
            raw=raw,
        )
    finally:
        conn.close()


def _set_cached_song_match(audio_key: str, match: SongMatch) -> None:
    if not audio_key or not match or not str(match.title or "").strip():
        return
    _init_cache()
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO song_id_cache(
                audio_key, provider, title, artist, album, confidence, track_id, payload, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audio_key,
                str(match.provider or ""),
                str(match.title or ""),
                str(match.artist or ""),
                str(match.album or ""),
                float(match.confidence or 0.0),
                str(match.track_id or ""),
                json.dumps(match.raw, ensure_ascii=False) if isinstance(match.raw, dict) else None,
                int(time.time()),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _requests_get(url: str, **kwargs):
    timeout = kwargs.pop("timeout", 12)
    headers = kwargs.pop("headers", {})
    headers.setdefault(
        "User-Agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    )
    return requests.get(
        url,
        timeout=timeout,
        headers=headers,
        proxies={"http": None, "https": None},
        **kwargs,
    )


@contextlib.contextmanager
def _clean_proxy_env():
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
    backup = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_whisperx_model(model_name: str = "small"):
    cache_key = model_name
    if cache_key in _WHISPERX_MODELS:
        return _WHISPERX_MODELS[cache_key]

    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    _WHISPERX_MODELS[cache_key] = model
    return model


def _load_align_model(language: str):
    import torch
    import whisperx

    normalized_language = {"yue": "zh", "zh-cn": "zh", "zh-tw": "zh"}.get(language or "", language or "zh")
    if normalized_language in _ALIGN_MODELS:
        return _ALIGN_MODELS[normalized_language]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = whisperx.load_align_model(language_code=normalized_language, device=device)
    _ALIGN_MODELS[normalized_language] = (model, metadata, device)
    return model, metadata, device


def transcribe_clip_with_whisperx(audio: np.ndarray, sr: int, model_name: str = "small") -> dict:
    import librosa

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    try:
        model = _load_whisperx_model(model_name)
        result = model.transcribe(audio, batch_size=4, language=None)
        segments = result.get("segments", [])
        text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
        avg_conf = 0.0
        scores = [float(seg.get("avg_logprob", -1.0)) for seg in segments if seg.get("avg_logprob") is not None]
        if scores:
            avg_conf = sum(scores) / len(scores)
        return {
            "engine": "whisperx",
            "language": result.get("language") or "auto",
            "segments": segments,
            "text": text,
            "confidence": avg_conf,
        }
    except Exception:
        import whisper

        model = whisper.load_model(model_name)
        result = model.transcribe(audio, language=None, word_timestamps=False, verbose=False)
        segments = [
            {
                "text": seg.get("text", "").strip(),
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "avg_logprob": float(seg.get("avg_logprob", -1.0)),
            }
            for seg in result.get("segments", [])
            if seg.get("text")
        ]
        scores = [seg["avg_logprob"] for seg in segments]
        return {
            "engine": "openai-whisper",
            "language": result.get("language") or "auto",
            "segments": segments,
            "text": result.get("text", "").strip(),
            "confidence": (sum(scores) / len(scores)) if scores else 0.0,
        }


def _get_song_provider_order() -> List[str]:
    raw = os.environ.get("SONG_IDENTIFY_PROVIDER_ORDER", "acrcloud,shazamio,shazam")
    providers = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not providers:
        return ["acrcloud", "shazamio", "shazam"]
    return providers


def _identify_song_with_acrcloud(audio_path: str) -> Optional[SongMatch]:
    host = (
        os.environ.get("ACRCLOUD_HOST")
        or os.environ.get("ACR_HOST")
        or ""
    ).strip()
    access_key = (
        os.environ.get("ACRCLOUD_ACCESS_KEY")
        or os.environ.get("ACR_ACCESS_KEY")
        or ""
    ).strip()
    access_secret = (
        os.environ.get("ACRCLOUD_ACCESS_SECRET")
        or os.environ.get("ACR_ACCESS_SECRET")
        or ""
    ).strip()

    if not (host and access_key and access_secret):
        return None

    timeout_sec = float(os.environ.get("ACRCLOUD_TIMEOUT_SEC", "10") or "10")
    rec_len = int(float(os.environ.get("ACRCLOUD_REC_LENGTH_SEC", "15") or "15"))
    cred_fp = _acr_credential_fingerprint(host, access_key, access_secret)
    blocked_until = float(_ACR_BLOCKED_CREDENTIALS.get(cred_fp, 0.0) or 0.0)
    if blocked_until > time.time():
        return None

    try:
        host_clean = host.strip()
        if not host_clean.startswith(("http://", "https://")):
            host_clean = f"https://{host_clean}"
        url = f"{host_clean.rstrip('/')}/v1/identify"

        with open(audio_path, "rb") as f:
            audio_buf = f.read()
        max_bytes = max(120000, int(rec_len * 16000 * 2))
        payload_buf = audio_buf[:max_bytes]
        if not payload_buf:
            return None

        timestamp = str(int(time.time()))
        to_sign = f"POST\n/v1/identify\n{access_key}\naudio\n1\n{timestamp}"
        signature = base64.b64encode(
            hmac.new(
                access_secret.encode("utf-8"),
                to_sign.encode("utf-8"),
                digestmod=hashlib.sha1,
            ).digest()
        ).decode("utf-8")
        sample = base64.b64encode(payload_buf).decode("utf-8")
        form = {
            "access_key": access_key,
            "sample_bytes": str(len(payload_buf)),
            "sample": sample,
            "timestamp": timestamp,
            "signature": signature,
            "data_type": "audio",
            "signature_version": "1",
        }
        with _clean_proxy_env():
            resp = requests.post(
                url,
                data=form,
                timeout=timeout_sec,
                headers={"User-Agent": "video-clip/1.0"},
                proxies={"http": None, "https": None},
            )
        if resp.status_code != 200:
            logger.warning("acrcloud http status=%s", resp.status_code)
            return None
        # ACRCloud返回Content-Type:text/plain(无charset)，resp.json()会误用ISO-8859-1导致中文乱码
        data = json.loads(resp.content.decode("utf-8"))
        if not isinstance(data, dict):
            return None
        status = data.get("status") or {}
        status_code = int(status.get("code") if status.get("code") is not None else -1)
        if status_code != 0:
            logger.warning(
                "acrcloud status code=%s msg=%s",
                status_code,
                str(status.get("msg") or ""),
            )
            if status_code in (3014, 3003, 3005, 3001):
                # 鉴权错误触发短路，避免整轮重复消耗配额/网络
                _ACR_BLOCKED_CREDENTIALS[cred_fp] = time.time() + 900.0
            return None

        music_items = ((data.get("metadata") or {}).get("music")) or []
        if not music_items:
            return None
        track = music_items[0] or {}
        title = str(track.get("title") or "").strip()
        if not title:
            return None

        artists = track.get("artists") or []
        artist = ", ".join(
            str(item.get("name") or "").strip()
            for item in artists
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        )
        album = str(((track.get("album") or {}).get("name")) or "").strip()
        score = float(track.get("score", 0.0) or 0.0)
        confidence = max(0.0, min(1.0, score / 100.0)) if score else 0.85
        track_id = (
            str(track.get("acrid") or "").strip()
            or str(((track.get("external_ids") or {}).get("isrc")) or "").strip()
        )
        _ACR_BLOCKED_CREDENTIALS.pop(cred_fp, None)
        return SongMatch(
            title=title,
            artist=artist,
            album=album,
            provider="acrcloud",
            confidence=confidence,
            track_id=track_id,
            raw=track,
        )
    except Exception as exc:
        logger.info("acrcloud identify failed: %s", exc)
        return None


def probe_acrcloud_auth(timeout_sec: float = 8.0) -> Tuple[bool, str]:
    """快速探测 ACRCloud 鉴权状态，不依赖真实音频。"""
    host = (os.environ.get("ACRCLOUD_HOST") or os.environ.get("ACR_HOST") or "").strip()
    access_key = (os.environ.get("ACRCLOUD_ACCESS_KEY") or os.environ.get("ACR_ACCESS_KEY") or "").strip()
    access_secret = (os.environ.get("ACRCLOUD_ACCESS_SECRET") or os.environ.get("ACR_ACCESS_SECRET") or "").strip()
    if not (host and access_key and access_secret):
        return False, "缺少 ACRCloud Host/Key/Secret"

    try:
        host_clean = host if host.startswith(("http://", "https://")) else f"https://{host}"
        url = f"{host_clean.rstrip('/')}/v1/identify"
        sample_bytes = b"\x00" * 1000
        timestamp = str(int(time.time()))
        to_sign = f"POST\n/v1/identify\n{access_key}\naudio\n1\n{timestamp}"
        signature = base64.b64encode(
            hmac.new(
                access_secret.encode("utf-8"),
                to_sign.encode("utf-8"),
                digestmod=hashlib.sha1,
            ).digest()
        ).decode("utf-8")
        form = {
            "access_key": access_key,
            "sample_bytes": str(len(sample_bytes)),
            "sample": base64.b64encode(sample_bytes).decode("utf-8"),
            "timestamp": timestamp,
            "signature": signature,
            "data_type": "audio",
            "signature_version": "1",
        }
        with _clean_proxy_env():
            resp = requests.post(
                url,
                data=form,
                timeout=timeout_sec,
                headers={"User-Agent": "video-clip/1.0"},
                proxies={"http": None, "https": None},
            )
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        # 同L400：ACRCloud返回无charset，必须显式UTF-8解码
        payload = json.loads(resp.content.decode("utf-8")) if resp.content else {}
        status = (payload or {}).get("status") or {}
        code = int(status.get("code") if status.get("code") is not None else -1)
        msg = str(status.get("msg") or "")
        if code == 3014:
            return False, "鉴权失败（invalid signature），请核对 Host/Key/Secret"
        if code in (3001, 3003, 3005):
            return False, f"鉴权失败（code={code}, {msg}）"
        if code in (0, 1001):
            # 0=识曲成功；1001=无识别结果（鉴权通常已通过）
            return True, f"鉴权通过（code={code}, {msg or 'ok'}）"
        if code >= 3000:
            return False, f"鉴权失败（code={code}, {msg}）"
        return True, f"接口可访问（code={code}, {msg or 'ok'}）"
    except Exception as exc:
        return False, f"连接失败: {exc}"


def _identify_song_with_shazamio(audio_path: str) -> Optional[SongMatch]:
    try:
        import asyncio
        from shazamio import Shazam as ShazamIO

        async def _recognize():
            with _clean_proxy_env():
                shazam = ShazamIO()
                timeout_sec = float(os.environ.get("SHAZAMIO_TIMEOUT_SEC", "6") or "6")
                return await asyncio.wait_for(shazam.recognize(audio_path), timeout=timeout_sec)

        result = asyncio.run(_recognize())
        track = result.get("track") if isinstance(result, dict) else None
        if track:
            return SongMatch(
                title=track.get("title", "").strip(),
                artist=track.get("subtitle", "").strip(),
                provider="shazamio",
                confidence=1.0 if track.get("title") else 0.0,
                track_id=str(track.get("key", "")),
                raw=track,
            )
    except Exception:
        return None
    return None


def _identify_song_with_shazamapi(audio_path: str) -> Optional[SongMatch]:
    try:
        from ShazamAPI import Shazam
    except Exception:
        return None

    result_holder: Dict[str, object] = {"track": None, "error": None}

    def _runner():
        try:
            with open(audio_path, "rb") as f, _clean_proxy_env():
                shazam = Shazam(f.read())
                for _offset, result in shazam.recognizeSong():
                    track = result.get("track")
                    if track:
                        result_holder["track"] = track
                        return
        except Exception as exc:
            result_holder["error"] = exc

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(timeout=float(os.environ.get("SHAZAMAPI_TIMEOUT_SEC", "6") or "6"))
    if t.is_alive():
        return None
    if result_holder.get("error"):
        return None
    best_track = result_holder.get("track")
    if not isinstance(best_track, dict):
        return None

    sections = best_track.get("sections") or []
    album = ""
    for section in sections:
        for meta in section.get("metadata", []):
            if meta.get("title") in {"Album", "专辑"}:
                album = meta.get("text", "")
                break
        if album:
            break

    return SongMatch(
        title=best_track.get("title", "").strip(),
        artist=best_track.get("subtitle", "").strip(),
        album=album.strip(),
        provider="shazam",
        confidence=1.0 if best_track.get("title") else 0.0,
        track_id=str(best_track.get("key", "")),
        raw=best_track,
    )


def identify_song_from_file(audio_path: str) -> Optional[SongMatch]:
    audio_key = _audio_cache_key(audio_path)
    if audio_key:
        cached = _get_cached_song_match(audio_key)
        if cached and cached.title:
            return cached
        miss_ts = float(_SONG_ID_MISS_CACHE.get(audio_key, 0.0) or 0.0)
        if miss_ts and (time.time() - miss_ts) < 600:
            return None

    for provider in _get_song_provider_order():
        if provider == "acrcloud":
            match = _identify_song_with_acrcloud(audio_path)
        elif provider == "shazamio":
            match = _identify_song_with_shazamio(audio_path)
        elif provider in {"shazam", "shazamapi"}:
            match = _identify_song_with_shazamapi(audio_path)
        else:
            continue
        if match and match.title:
            if audio_key:
                _set_cached_song_match(audio_key, match)
            return match
    if audio_key:
        _SONG_ID_MISS_CACHE[audio_key] = time.time()
    return None


def _get_cached_lyrics(cache_key: str) -> Optional[dict]:
    _init_cache()
    conn = sqlite3.connect(CACHE_DB)
    try:
        row = conn.execute(
            "SELECT payload FROM lyrics_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])
    finally:
        conn.close()


def _set_cached_lyrics(cache_key: str, provider: str, title: str, artist: str, payload: dict) -> None:
    _init_cache()
    conn = sqlite3.connect(CACHE_DB)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO lyrics_cache(cache_key, provider, title, artist, payload, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cache_key, provider, title, artist, json.dumps(payload, ensure_ascii=False), int(time.time())),
        )
        conn.commit()
    finally:
        conn.close()


def _fetch_lyrics_lrclib(song: SongMatch) -> Optional[dict]:
    """
    从 LRClib API 获取歌词（带时间戳优先）。
    
    策略：
      1. 先找 synced=True 的结果（带逐词/逐句时间戳）
      2. 如果没有 synced 结果，返回 None（让 netease 接手）
      3. 不再降级到 plainLyrics（无时间戳对 ASR 无用）
    """
    try:
        resp = _requests_get(
            "https://lrclib.net/api/search",
            params={"track_name": song.title, "artist_name": song.artist},
        )
        if resp.status_code != 200:
            return None
        results = resp.json()
        if not isinstance(results, list) or not results:
            return None

        def score(item: dict) -> float:
            item_title = item.get("trackName", "")
            item_artist = item.get("artistName", "")
            return SequenceMatcher(
                None,
                _normalize_song_key(item_title, item_artist),
                _normalize_song_key(song.title, song.artist),
            ).ratio()

        # --- 第一轮：只看带时间戳的结果 ---
        synced_results = [r for r in results if r.get("syncedLyrics")]
        if synced_results:
            best = max(synced_results, key=score)
            s = score(best)
            if s >= 0.45:  # synced 匹配阈值可稍低（有时间的更稀有）
                return {
                    "provider": "lrclib-synced",
                    "title": best.get("trackName", song.title),
                    "artist": best.get("artistName", song.artist),
                    "lyrics": best["syncedLyrics"].strip(),
                    "synced": True,
                    "match_score": round(s, 3),
                    "line_count": best["syncedLyrics"].strip().count('\n') + 1,
                }

        # --- 无 synced 结果 → 返回 None，交给 netease ---
        # （不再降级到 plainLyrics，因为无时间戳歌词无法用于 ASR 对齐）
        return None
    except Exception:
        return None


def _fetch_lyrics_netease(song: SongMatch) -> Optional[dict]:
    try:
        search_resp = _requests_get(
            "https://music.163.com/api/search/get/web",
            params={
                "csrf_token": "",
                "s": f"{song.title} {song.artist}",
                "type": 1,
                "offset": 0,
                "total": "true",
                "limit": 10,
            },
            headers={"Referer": "https://music.163.com/"},
        )
        if search_resp.status_code != 200:
            return None
        search_data = search_resp.json()
        songs = (((search_data or {}).get("result") or {}).get("songs")) or []
        if not songs:
            return None

        def score(item: dict) -> float:
            item_title = item.get("name", "")
            artists = " ".join(a.get("name", "") for a in item.get("artists", []))
            return SequenceMatcher(None, _normalize_song_key(item_title, artists), _normalize_song_key(song.title, song.artist)).ratio()

        best = max(songs, key=score)
        if score(best) < 0.55:
            return None
        lyric_resp = _requests_get(
            "https://music.163.com/api/song/lyric",
            params={"id": best.get("id"), "lv": 1, "kv": 1, "tv": -1},
            headers={"Referer": "https://music.163.com/"},
        )
        if lyric_resp.status_code != 200:
            return None
        lyric_data = lyric_resp.json()
        lyrics_text = (((lyric_data or {}).get("lrc") or {}).get("lyric") or "").strip()
        if not lyrics_text:
            return None
        return {
            "provider": "netease",
            "title": best.get("name", song.title),
            "artist": " / ".join(a.get("name", "") for a in best.get("artists", [])),
            "lyrics": lyrics_text,
            "synced": bool(re.search(r"\[\d{2}:\d{2}(?:\.\d{1,3})?\]", lyrics_text)),
        }
    except Exception:
        return None


def fetch_lyrics(song: SongMatch) -> Optional[dict]:
    """
    多源并行搜索带时间戳歌词（LRC格式）。
    
    策略：
      1. 先查缓存
      2. 并行请求所有源：lrclib-synced > netease
      3. 优先返回 synced=True 的结果
      4. 如果多源都有结果，做交叉校准取最优
    
    Args:
        song: 歌曲信息（title + artist）
    
    Returns:
        {
            "provider": "lrclib-synced" | "netease" | "merged",
            "title": str,
            "artist": str, 
            "lyrics": str (带时间戳的 LRC 文本),
            "synced": bool,
            "match_score": float,
            "line_count": int,
            "sources_tried": list,  # 调试用，记录尝试了哪些源
            "cross_validated": bool,  # 是否经过多源交叉验证
        }
        或 None（全部失败）
    """
    cache_key = _normalize_song_key(song.title, song.artist)
    cached = _get_cached_lyrics(cache_key)
    if cached and cached.get("synced"):
        cached["sources_tried"] = ["cache"]
        return cached

    import concurrent.futures
    sources_tried = []
    results = {}

    # 并行调用所有歌词源
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_name = {
            executor.submit(_fetch_lyrics_lrclib, song): "lrclib-synced",
            executor.submit(_fetch_lyrics_netease, song): "netease",
        }
        
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                payload = future.result(timeout=15)  # 单个超时15秒
                if payload and payload.get("synced"):
                    results[name] = payload
                    sources_tried.append(f"{name}✅")
                else:
                    sources_tried.append(f"{name}⏭️")
            except Exception:
                sources_tried.append(f"{name}❌")

    if not results:
        # 所有源都没有带时间戳的歌词
        return None

    # --- 选择最优结果 ---
    best_source = max(results.keys(), key=lambda k: results[k].get("match_score", 0))
    best_result = results[best_source]
    
    # --- 多源交叉校验 ---
    cross_validated = False
    if len(results) >= 2:
        # 有多个源的结果，做文本相似度校验
        texts = {k: _normalize_text(v.get("lyrics", "")) for k, v in results.items()}
        pairs = [(a, b) for i, a in enumerate(texts.values()) 
                 for b in list(texts.values())[i+1:]]
        similarities = [SequenceMatcher(None, p[0], p[1]).ratio() for p in pairs]
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        
        if avg_sim >= 0.7:
            # 多源一致，选匹配度最高的那个
            cross_validated = True

    result = {
        **best_result,
        "sources_tried": sources_tried,
        "cross_validated": cross_validated,
    }

    _set_cached_lyrics(cache_key, result["provider"], song.title, song.artist, result)
    return result


def build_song_from_hint(song_hint: Optional[dict | SongHint]) -> Optional[SongMatch]:
    if not song_hint:
        return None
    if isinstance(song_hint, SongHint):
        title = song_hint.title
        artist = song_hint.artist
    else:
        title = str(song_hint.get("title") or "")
        artist = str(song_hint.get("artist") or "")
    title = title.strip()
    artist = artist.strip()
    if re.fullmatch(r"song[_\-\s]*\d+", title, flags=re.IGNORECASE):
        return None
    if not title:
        return None
    return SongMatch(title=title, artist=artist, provider="hint", confidence=0.8)


def parse_lrc_text(lyrics_text: str) -> List[LyricLine]:
    lines: List[LyricLine] = []
    timestamp_re = re.compile(r"\[(\d{2}):(\d{2})(?:\.(\d{1,3}))?\]")
    metadata_re = re.compile(r"^\[(?:ti|ar|al|au|by|offset|length|tool|ve):", re.IGNORECASE)
    for raw_line in (lyrics_text or "").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        if metadata_re.match(raw_line):
            continue
        matches = list(timestamp_re.finditer(raw_line))
        text = timestamp_re.sub("", raw_line).strip()
        if not text:
            continue
        if matches:
            for match in matches:
                minute = int(match.group(1))
                second = int(match.group(2))
                fraction = match.group(3) or "0"
                scale = 100 if len(fraction) == 2 else (10 if len(fraction) == 1 else 1000)
                timestamp = minute * 60 + second + int(fraction) / scale
                lines.append(LyricLine(index=len(lines), text=text, start=timestamp))
        else:
            lines.append(LyricLine(index=len(lines), text=text))

    for i in range(len(lines) - 1):
        if lines[i].start is not None and lines[i + 1].start is not None:
            lines[i].end = max(lines[i].start, lines[i + 1].start)
    return lines


def slice_lyric_lines(
    lyric_lines: List[LyricLine],
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> List[LyricLine]:
    """按 0-based 行号截取用户确认的歌词窗口。"""
    if not lyric_lines:
        return []
    start = max(0, min(int(start_index or 0), len(lyric_lines) - 1))
    if end_index is None:
        end = len(lyric_lines)
    else:
        end = max(start + 1, min(int(end_index), len(lyric_lines)))
    return lyric_lines[start:end]


def _apply_offset_to_segments(segments: List[dict], duration: float, offset_sec: float) -> List[dict]:
    shifted = []
    for seg in segments:
        start = float(seg.get("start", 0.0)) + float(offset_sec or 0.0)
        end = float(seg.get("end", start + 0.2)) + float(offset_sec or 0.0)
        if end <= 0 or start >= duration:
            continue
        start = max(0.0, start)
        end = min(duration, max(end, start + 0.2))
        shifted.append({"text": seg.get("text", "").strip(), "start": round(start, 3), "end": round(end, 3)})
    return [seg for seg in shifted if seg["text"]]


def _build_lrc_segments_from_selected_lines(
    lines: List[LyricLine],
    duration: float,
    offset_sec: float = 0.0,
    clip_start_sec: Optional[float] = None,
) -> List[dict]:
    if not lines:
        return []
    if not any(line.start is not None for line in lines):
        return _apply_offset_to_segments(_build_initial_segments(lines, duration), duration, offset_sec)

    if clip_start_sec is None:
        first_start = next((line.start for line in lines if line.start is not None), 0.0) or 0.0
    else:
        first_start = max(0.0, float(clip_start_sec))
    segments = []
    for idx, line in enumerate(lines):
        if line.start is None:
            continue
        next_start = None
        for next_line in lines[idx + 1:]:
            if next_line.start is not None:
                next_start = next_line.start
                break
        raw_start = float(line.start) - first_start
        if line.end is not None:
            raw_end = float(line.end) - first_start
        elif next_start is not None:
            raw_end = float(next_start) - first_start
        else:
            raw_end = duration
        if raw_end <= 0 or raw_start >= duration:
            continue
        segments.append({"text": line.text, "start": max(0.0, raw_start), "end": max(raw_end, raw_start + 0.8)})
    return _apply_offset_to_segments(segments, duration, offset_sec)


def build_standard_lyric_subtitle_result(
    audio_segment: np.ndarray,
    sr: int,
    lyrics_text: str,
    title: str = "",
    artist: str = "",
    provider: str = "manual",
    start_line: int = 0,
    end_line: Optional[int] = None,
    clip_start_sec: Optional[float] = None,
    offset_sec: float = 0.0,
) -> Optional[dict]:
    lyric_lines = parse_lrc_text(lyrics_text)
    if not lyric_lines:
        return None

    duration = len(audio_segment) / float(sr)
    if clip_start_sec is not None and any(line.start is not None for line in lyric_lines):
        selected_lines = [
            line for line in lyric_lines
            if line.start is not None
            and (line.end or line.start + 0.8) >= float(clip_start_sec)
            and line.start <= float(clip_start_sec) + duration
        ]
    else:
        selected_lines = slice_lyric_lines(lyric_lines, start_line, end_line)
    sentences = _build_lrc_segments_from_selected_lines(selected_lines, duration, offset_sec, clip_start_sec)
    if not sentences:
        return None

    return {
        "engine": "standard-lyrics",
        "language": "lyrics",
        "text": " ".join(seg["text"] for seg in sentences).strip(),
        "words": [],
        "sentences": sentences,
        "confidence": 1.0,
        "song": {"title": title, "artist": artist, "provider": "manual"},
        "lyrics_provider": provider,
        "lyric_start_line": int(start_line or 0),
        "lyric_end_line": end_line,
        "clip_start_sec": clip_start_sec,
        "offset_sec": float(offset_sec or 0.0),
    }


def _find_best_lyric_window(rough_text: str, lyric_lines: List[LyricLine], max_window: int = 6) -> List[LyricLine]:
    if not lyric_lines:
        return []
    normalized_rough = _normalize_text(rough_text)
    if not normalized_rough:
        return lyric_lines[: min(3, len(lyric_lines))]

    best_score = -1.0
    best_slice = lyric_lines[: min(3, len(lyric_lines))]
    upper = min(max_window, len(lyric_lines))
    for window_size in range(1, upper + 1):
        for start in range(0, len(lyric_lines) - window_size + 1):
            candidate = lyric_lines[start : start + window_size]
            joined = "".join(line.text for line in candidate)
            score = SequenceMatcher(None, normalized_rough, _normalize_text(joined)).ratio()
            if score > best_score:
                best_score = score
                best_slice = candidate
    return best_slice


def locate_lyric_window_in_audio(
    audio: np.ndarray,
    sr: int,
    lyrics_text: str,
    model_name: str = "tiny",
    max_window: int = 10,
) -> Optional[dict]:
    """用粗 ASR 文本在整首歌词中滑窗匹配，自动定位当前音频片段对应的歌词位置。"""
    lyric_lines = parse_lrc_text(lyrics_text)
    if not lyric_lines:
        return None

    rough = transcribe_clip_with_whisperx(audio, sr, model_name=model_name)
    rough_text = rough.get("text", "")
    normalized_rough = _normalize_text(rough_text)
    if not normalized_rough:
        return None

    best_score = -1.0
    best_start = 0
    best_end = min(len(lyric_lines), max(1, min(max_window, len(lyric_lines))))
    max_size = min(max_window, len(lyric_lines))
    for window_size in range(1, max_size + 1):
        for start in range(0, len(lyric_lines) - window_size + 1):
            candidate = lyric_lines[start:start + window_size]
            candidate_text = "".join(line.text for line in candidate)
            score = SequenceMatcher(None, normalized_rough, _normalize_text(candidate_text)).ratio()
            if score > best_score:
                best_score = score
                best_start = start
                best_end = start + window_size

    selected = lyric_lines[best_start:best_end]
    clip_start_sec = None
    first_timed = next((line for line in selected if line.start is not None), None)
    if first_timed is not None:
        clip_start_sec = float(first_timed.start)

    return {
        "rough_text": rough_text.strip(),
        "score": float(best_score),
        "start_line": best_start,
        "end_line": best_end,
        "clip_start_sec": clip_start_sec,
        "matched_text": " ".join(line.text for line in selected).strip(),
        "engine": rough.get("engine", ""),
    }


def _build_initial_segments(lines: List[LyricLine], duration: float) -> List[dict]:
    if not lines:
        return []
    total_chars = max(sum(max(1, len(_normalize_text(line.text))) for line in lines), 1)
    current = 0.0
    segments = []
    for idx, line in enumerate(lines):
        weight = max(1, len(_normalize_text(line.text))) / total_chars
        seg_duration = duration * weight
        start = current
        if idx == len(lines) - 1:
            end = duration
        else:
            end = min(duration, current + seg_duration)
        segments.append({"text": line.text, "start": round(start, 3), "end": round(max(end, start + 0.2), 3)})
        current = end
    return segments


def _aligned_words_to_sentences(aligned_segments: Iterable[dict]) -> List[dict]:
    sentences = []
    for seg in aligned_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        sentences.append({"text": text, "start": start, "end": end})
    return sentences


def align_lyrics_to_audio(
    audio: np.ndarray,
    sr: int,
    lyric_lines: List[LyricLine],
    model_name: str = "small",
) -> Optional[dict]:
    if not lyric_lines:
        return None

    try:
        rough = transcribe_clip_with_whisperx(audio, sr, model_name=model_name)
    except Exception:
        return None

    chosen_lines = _find_best_lyric_window(rough["text"], lyric_lines)
    if not chosen_lines:
        return None

    duration = len(audio) / float(sr)
    transcript_segments = _build_initial_segments(chosen_lines, duration)
    if not transcript_segments:
        return None

    language = rough.get("language") or "auto"
    try:
        import librosa
        import whisperx

        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
        align_model, align_metadata, device = _load_align_model(language)
        aligned = whisperx.align(
            transcript_segments,
            align_model,
            align_metadata,
            audio_16k,
            device,
            return_char_alignments=False,
        )
        aligned_segments = aligned.get("segments", [])
        words = []
        for seg in aligned_segments:
            for word in seg.get("words", []):
                if word.get("word"):
                    words.append(
                        {
                            "word": word["word"].strip(),
                            "start": float(word.get("start", seg.get("start", 0.0))),
                            "end": float(word.get("end", seg.get("end", 0.0))),
                        }
                    )
        sentences = _aligned_words_to_sentences(aligned_segments)
        if not sentences:
            sentences = [{"text": seg["text"], "start": seg["start"], "end": seg["end"]} for seg in transcript_segments]
        return {
            "engine": "whisperx-lyrics",
            "language": language,
            "text": " ".join(line["text"] for line in sentences).strip(),
            "words": words,
            "sentences": sentences,
            "confidence": rough.get("confidence", 0.0),
        }
    except Exception:
        return {
            "engine": "lyrics-fallback",
            "language": language,
            "text": " ".join(seg["text"] for seg in transcript_segments).strip(),
            "words": [],
            "sentences": [{"text": seg["text"], "start": seg["start"], "end": seg["end"]} for seg in transcript_segments],
            "confidence": rough.get("confidence", 0.0),
        }


def build_lyric_subtitle_result(
    audio_segment: np.ndarray,
    sr: int,
    model_name: str = "tiny",
    song_hint: Optional[dict | SongHint] = None,
    lyrics_text: str = "",
    lyric_start_index: int = 0,
    lyric_end_index: Optional[int] = None,
    clip_start_sec: Optional[float] = None,
    offset_sec: float = 0.0,
    auto_locate: bool = True,
    allow_song_identification: bool = False,
) -> Optional[dict]:
    import soundfile as sf

    temp_audio = None
    try:
        song_candidates = []
        hint_song = build_song_from_hint(song_hint)
        if hint_song:
            song_candidates.append(hint_song)

        if not lyrics_text and isinstance(song_hint, SongHint):
            lyrics_text = song_hint.lyrics_text
            lyric_start_index = song_hint.start_line
            lyric_end_index = song_hint.end_line
            clip_start_sec = song_hint.clip_start_sec
            offset_sec = song_hint.offset_sec
        elif not lyrics_text and isinstance(song_hint, dict):
            lyrics_text = str(song_hint.get("lyrics_text") or "")
            lyric_start_index = int(song_hint.get("start_line") or lyric_start_index or 0)
            raw_end = song_hint.get("end_line", lyric_end_index)
            lyric_end_index = None if raw_end in (None, "") else int(raw_end)
            raw_clip_start = song_hint.get("clip_start_sec", clip_start_sec)
            clip_start_sec = None if raw_clip_start in (None, "") else float(raw_clip_start)
            offset_sec = float(song_hint.get("offset_sec") or offset_sec or 0.0)

        if lyrics_text:
            song = hint_song or SongMatch(title="", artist="", provider="manual", confidence=1.0)
            lyrics_provider = "manual"
            if isinstance(song_hint, dict):
                lyrics_provider = str(song_hint.get("lyrics_provider") or lyrics_provider)
            location = None
            if auto_locate and clip_start_sec is None:
                location = locate_lyric_window_in_audio(
                    audio=audio_segment,
                    sr=sr,
                    lyrics_text=lyrics_text,
                    model_name=model_name,
                )
                if location:
                    lyric_start_index = int(location["start_line"])
                    lyric_end_index = int(location["end_line"])
                    if location.get("clip_start_sec") is not None:
                        clip_start_sec = float(location["clip_start_sec"])

            result = build_standard_lyric_subtitle_result(
                audio_segment=audio_segment,
                sr=sr,
                lyrics_text=lyrics_text,
                title=song.title,
                artist=song.artist,
                provider=lyrics_provider,
                start_line=lyric_start_index,
                end_line=lyric_end_index,
                clip_start_sec=clip_start_sec,
                offset_sec=offset_sec,
            )
            if result and location:
                result["auto_location"] = location
            return result

        if allow_song_identification:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_audio = tmp.name
            sf.write(temp_audio, audio_segment, sr)
            identified_song = identify_song_from_file(temp_audio)
            if identified_song:
                song_candidates.append(identified_song)
        if not song_candidates:
            return None

        song = None
        lyric_payload = None
        for candidate in song_candidates:
            lyric_payload = fetch_lyrics(candidate)
            if lyric_payload:
                song = candidate
                break
        if not song or not lyric_payload:
            return None

        lyric_lines = parse_lrc_text(lyric_payload["lyrics"])
        if not lyric_lines:
            return None

        selected_lines = slice_lyric_lines(lyric_lines, lyric_start_index, lyric_end_index)
        aligned = build_standard_lyric_subtitle_result(
            audio_segment=audio_segment,
            sr=sr,
            lyrics_text="\n".join(
                (
                    f"[{int(line.start // 60):02d}:{line.start % 60:05.2f}]{line.text}"
                    if line.start is not None else line.text
                )
                for line in selected_lines
            ),
            title=song.title,
            artist=song.artist,
            provider=lyric_payload.get("provider", ""),
            start_line=0,
            end_line=None,
            clip_start_sec=clip_start_sec,
            offset_sec=offset_sec,
        )
        if not aligned:
            return None

        aligned["song"] = {"title": song.title, "artist": song.artist, "provider": song.provider}
        aligned["lyrics_provider"] = lyric_payload.get("provider", "")
        return aligned
    finally:
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)

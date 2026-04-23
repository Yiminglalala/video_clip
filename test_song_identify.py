# -*- coding: utf-8 -*-
"""
Test: Song Recognition + Lyrics Fetch Pipeline
Input: E:\BaiduNetdiskDownload\video_20260412_210649.mp4 (224s, 4K)
"""

import os
import sys
import json
import time
import requests

VIDEO_PATH = r"E:\BaiduNetdiskDownload\video_20260412_210649.mp4"
OUTPUT_DIR = r"D:\video_clip\output\song_id_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Extract audio (first 30 seconds for song recognition)
# ============================================================
print("=" * 60)
print("Step 1: Extract first 30s of audio")
print("=" * 60)

audio_30s = os.path.join(OUTPUT_DIR, "audio_30s.wav")
cmd = [
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-t", "30",
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "44100",
    "-ac", "2",
    audio_30s,
]
os.system(" ".join(cmd))

if os.path.exists(audio_30s):
    size_mb = os.path.getsize(audio_30s) / 1024 / 1024
    print(f"[OK] Audio extracted: {audio_30s} ({size_mb:.1f} MB)")
else:
    print("[FAIL] Audio extraction failed")
    sys.exit(1)

# ============================================================
# Step 2: ShazamAPI song recognition
# ============================================================
print("\n" + "=" * 60)
print("Step 2: ShazamAPI song recognition")
print("=" * 60)

song_info = None

try:
    from shazamio import Shazam
    
    async def recognize():
        shazam = Shazam()
        result = await shazam.recognize(audio_30s)
        return result
    
    import asyncio
    result = asyncio.run(recognize())
    
    if hasattr(result, 'json'):
        data = result.json() if callable(result.json) else result.json
    else:
        data = result
    
    # Save raw result for debugging
    with open(os.path.join(OUTPUT_DIR, "shazam_result.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Raw result saved to shazam_result.json")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
    
    try:
        track = data.get('track', {})
        title = track.get('title', '')
        subtitle_raw = track.get('subtitle', '')
        subtitle = subtitle_raw.get('text', str(subtitle_raw)) if isinstance(subtitle_raw, dict) else str(subtitle_raw)
        
        if title:
            song_info = {'title': title, 'artist': subtitle}
            print(f"\n[RESULT] Song: {title} - {subtitle}")
        else:
            print("\n[WARN] No track info in result")
            # Check for other fields
            print(f"Available keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    except Exception as e:
        print(f"[WARN] Parse error: {e}")
        import traceback; traceback.print_exc()

except ImportError:
    print("[WARN] shazamio not installed, trying ShazamAPI...")
    
    try:
        from ShazamAPI import ShazamAPI
        mp3_file_to_shazam = open(audio_30s, 'rb').read()
        shazam = ShazamAPI(mp3_file_to_shazam)
        result = shazam.top_results()[0]  # best match
        
        song_info = {
            'title': getattr(result, 'title', 'unknown'),
            'artist': getattr(result, 'artist', 'unknown'),
        }
        print(f"[RESULT] Song: {song_info['title']} - {song_info['artist']}")
        
    except ImportError:
        print("[SKIP] ShazamAPI not installed either")
    except Exception as e:
        print(f"[FAIL] ShazamAPI error: {e}")

except Exception as e:
    print(f"[FAIL] Shazam error: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# Step 3: NetEase Music lyrics fetch (LRC format)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: NetEase Cloud Music - fetch LRC lyrics")
print("=" * 60)

lrc_lyrics = None

if song_info and song_info.get('title'):
    search_terms = [f"{song_info['title']} {song_info['artist']}"]
else:
    # Fallback candidates based on previous test context
    search_terms = [
        "A-Lin sound dream",
        "A-Lin voice dream concert",
        "A-Lin",
    ]

headers = {
    "Referer": "https://music.163.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

for term in search_terms:
    print(f"\n[SEARCH] Query: {term}")
    
    try:
        import urllib.parse
        encoded_term = urllib.parse.quote(term)
        search_url = f"http://music.163.com/api/search/get?s={encoded_term}&type=1&limit=5"
        resp = requests.get(search_url, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            print(f"   [WARN] HTTP {resp.status_code}")
            continue
            
        data = resp.json()
        
        songs = data.get('result', {}).get('songs', [])
        if not songs:
            print(f"   [SKIP] No results found")
            continue
            
        print(f"   Found {len(songs)} songs:")
        for i, s in enumerate(songs[:5]):
            artists_list = s.get('artists', [])
            artist_name = artists_list[0].get('name', '?') if artists_list else '?'
            song_name = s.get('name', '?')
            song_id = s.get('id')
            print(f"   [{i+1}] {song_name} - {artist_name} (id={song_id})")
            
            # Get lyrics
            lrc_url = f"http://music.163.com/api/song/lyric?id={song_id}&lv=1"
            try:
                lrc_resp = requests.get(lrc_url, headers=headers, timeout=10)
                
                if lrc_resp.status_code == 200:
                    lrc_data = lrc_resp.json()
                    lrc_text = lrc_data.get('lrc', {}).get('lyric', '')
                    
                    if lrc_text:
                        lrc_file = os.path.join(OUTPUT_DIR, f"lyrics_{song_id}.lrc")
                        with open(lrc_file, 'w', encoding='utf-8') as f:
                            f.write(lrc_text)
                        
                        text_lines = [l for l in lrc_text.split('\n') if l.strip() and not l.startswith('[')]
                        print(f"      [OK] Lyrics saved ({len(text_lines)} text lines)")
                        lrc_lyrics = lrc_text
                        break
                    else:
                        print(f"      [WARN] Empty lyrics")
                else:
                    print(f"      [WARN] Lyrics HTTP {lrc_resp.status_code}")
            except Exception as le:
                print(f"      [ERROR] Lyrics fetch failed: {le}")
        
        if lrc_lyrics:
            break
            
    except Exception as e:
        print(f"   [ERROR] {e}")

if lrc_lyrics:
    print("\n[LYRICS PREVIEW]")
    print("-" * 40)
    preview_lines = lrc_lyrics.split('\n')[:25]
    for line in preview_lines:
        print(line)
    print("-" * 40)
    total_chars = len(lrc_lyrics)
    print(f"[Total: {total_chars} chars]")
else:
    print("\n[WARN] No lyrics obtained from any source")

# ============================================================
# Step 4: Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Input video : {VIDEO_PATH}")
print(f"Video length: 224s (~3.7 min)")
print(f"Song recog  : {'YES - ' + str(song_info) if song_info else 'NO'}")
print(f"Lyrics      : {'YES' if lrc_lyrics else 'NO'}")
print(f"Output dir  : {OUTPUT_DIR}")

# List output files
files = os.listdir(OUTPUT_DIR)
for f in files:
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath) / 1024
    print(f"  -> {f} ({size:.1f} KB)")

print("\n[DONE]")

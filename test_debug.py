# -*- coding: utf-8 -*-
"""
调试：看看网易云搜索结果
"""

import requests
import json

url = "http://music.163.com/api/cloudsearch/pc"
params = {
    "s": "周杰伦 天青色等烟雨",
    "type": 1,
    "limit": 10
}
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://music.163.com/"
}

resp = requests.get(url, params=params, headers=headers, timeout=10)
data = resp.json()
print(f"Status: {data.get('code')}")
songs = data.get("result", {}).get("songs", [])
print(f"\n找到 {len(songs)} 首歌:\n")
for song in songs:
    name = song.get("name", "")
    artists = [ar.get("name", "") for ar in song.get("ar", [])]
    song_id = song.get("id", "")
    print(f"  - {name} - {artists} (id: {song_id})")

# -*- coding: utf-8 -*-
"""
测试：歌词+歌手 网页搜索 歌名
"""

import requests
import re


def get_lyric_by_song_id(song_id):
    """获取网易云音乐歌词"""
    url = f"https://music.163.com/api/song/lyric?id={song_id}&lv=1&kv=1&tv=-1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://music.163.com/"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            lyric = data.get("lrc", {}).get("lyric", "")
            if lyric:
                lyric = re.sub(r"\[\d+:\d+\.\d+\]", "", lyric)
                lyric = re.sub(r"\[\d+:\d+\]", "", lyric)
                return lyric.strip()
    except Exception as e:
        pass
    return ""


def calculate_lyrics_similarity(lyric1, lyric2):
    """简单计算歌词相似度"""
    def clean(l):
        return re.sub(r"\s+", "", l)
    
    l1 = clean(lyric1)
    l2 = clean(lyric2)
    
    max_common = 0
    for i in range(len(l1)):
        for j in range(i + 1, len(l1) + 1):
            substr = l1[i:j]
            if len(substr) > max_common and substr in l2:
                max_common = len(substr)
    
    lines1 = [line.strip() for line in lyric1.splitlines() if line.strip()]
    lines2 = [line.strip() for line in lyric2.splitlines() if line.strip()]
    
    line_matches = 0
    for line in lines1:
        if any(line in l2 for l2 in lines2):
            line_matches += 1
    
    return max_common + line_matches * 10


def test_search_netease(lyrics: str, singer: str):
    """
    用网易云音乐搜索API + 歌词相似度匹配
    """
    print("=" * 60)
    print(f"歌手: {singer}")
    print(f"歌词:\n{lyrics}")
    print("-" * 60)

    text = lyrics.strip()
    singer_name = singer.strip()
    
    if not text or not singer_name:
        print("缺少歌词或歌手")
        return None

    try:
        lines = text.splitlines()
        search_lyrics_list = []
        for i in range(min(3, len(lines))):
            if lines[i].strip():
                search_lyrics_list.append(lines[i].strip())

        if not search_lyrics_list:
            search_lyrics_list = [text[:100]]

        print(f"将尝试 {len(search_lyrics_list)} 组搜索词:")
        for i, line in enumerate(search_lyrics_list, 1):
            print(f"  {i}. {line}")

        song_candidates = {}

        for i, search_lyric in enumerate(search_lyrics_list, 1):
            print(f"\n--- 第 {i} 次搜索 (网易云音乐) ---")
            search_query = f"{search_lyric}"
            print(f"搜索: {search_query}")

            url = "http://music.163.com/api/cloudsearch/pc"
            params = {
                "s": search_query,
                "type": 1,
                "limit": 15
            }
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://music.163.com/"
            }

            try:
                resp = requests.get(url, params=params, headers=headers, timeout=10)
                if resp.status_code != 200:
                    print(f"搜索失败: {resp.status_code}")
                    continue

                data = resp.json()
                if data.get("code") != 200:
                    print(f"API错误: {data}")
                    continue

                songs = data.get("result", {}).get("songs", [])
                print(f"找到 {len(songs)} 首歌")

                for song in songs:
                    song_name = song.get("name", "")
                    song_id = song.get("id", "")
                    artists = [ar.get("name", "") for ar in song.get("ar", [])]
                    
                    print(f"  🔍 检查: {song_name} - {artists}")

                    full_lyric = get_lyric_by_song_id(song_id)
                    if full_lyric:
                        score = calculate_lyrics_similarity(text, full_lyric)
                        
                        print(f"    得分: {score}")
                        
                        if song_name not in song_candidates:
                            song_candidates[song_name] = {"score": 0, "count": 0}
                        song_candidates[song_name]["score"] += score
                        song_candidates[song_name]["count"] += 1
                    else:
                        print(f"    ⚠️  未获取到歌词")

            except Exception as e:
                print(f"搜索出错: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("最终得分排行:")
        
        sorted_songs = sorted(
            song_candidates.items(),
            key=lambda x: (x[1]["score"] / x[1]["count"], x[1]["count"]),
            reverse=True
        )

        for song_name, info in sorted_songs[:10]:
            avg_score = info["score"] / info["count"]
            print(f"  {song_name}: avg={avg_score:.1f}, total={info['score']}, count={info['count']}")

        if sorted_songs:
            best_song = None
            for song_name, info in sorted_songs:
                avg_score = info["score"] / info["count"]
                if avg_score > 10:
                    best_song = song_name
                    break
            
            if best_song:
                print(f"\n✅ 最可能的歌名: {best_song}")
                return best_song
            else:
                print("\n⚠️  没有找到足够匹配的歌曲")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

    return None


if __name__ == "__main__":
    test_singer = "小阿七"
    test_lyrics = """我做不了的梦醒不来的梦
寻不到的天堂医不好的痛
点不着的香烟松不开的手
忘不了的某某某
是我寻觅不到的风说不完的红"""
    
    result = test_search_netease(test_lyrics, test_singer)
    print(f"\n最终结果: {result}")

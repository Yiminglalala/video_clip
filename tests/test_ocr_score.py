#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试 OCR 评分函数"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import LiveVideoProcessor, ProcessingConfig

p = LiveVideoProcessor()

tests = [
    '《一夜》',
    '有没有人曾告诉你我很爱你',
    '看不见雪的冬天不夜的 城市',
    '忽然感到无比的思念',
    '来吧来吧去跳舞',
    'Blue Note OCUIC 有没有人曾告诉你我很爱你',
    '陈 婪 Concett Live',
    '就忘了所有的痛苦',       # 歌词，应该被拦截
    '却无法忘记你的脸',        # 歌词，应该被拦截
    '滴答滴答',               # 可能是歌名或歌词
]

print('=' * 70)
print(f'  {"文本":35s} | {"分数":>5s} | 通过 | 原因')
print('-' * 70)
for t in tests:
    score, reason = p._score_ocr_as_title(t)
    passed = p._is_likely_song_title_text(t)
    print(f'  {t[:33]:33s} | {score:5.1f} | {"YES" if passed else " NO"} | {reason[:40]}')
print('=' * 70)

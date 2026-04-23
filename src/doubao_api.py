# -*- coding: utf-8 -*-
"""
豆包语音API封装模块

实现豆包语音的音视频字幕API调用
支持音频提交和结果查询功能
"""

import time
import requests
from typing import Dict, Optional


def _ends_with_punctuation(text: str) -> bool:
    return bool(text) and text[-1] in "。！？!?,，、；;：:"


def _needs_space_between(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum()


def _merge_sentence_fragments(sentences: list) -> list:
    """Merge overly fragmented utterances into sentence-like subtitle lines."""
    if not sentences:
        return []

    ordered = sorted(sentences, key=lambda x: float(x.get("start", 0.0)))
    merged = []
    current = dict(ordered[0])

    for nxt in ordered[1:]:
        cur_text = (current.get("text") or "").strip()
        nxt_text = (nxt.get("text") or "").strip()
        if not nxt_text:
            continue

        cur_start = float(current.get("start", 0.0))
        cur_end = float(current.get("end", cur_start))
        nxt_start = float(nxt.get("start", cur_end))
        nxt_end = float(nxt.get("end", nxt_start))

        gap = max(0.0, nxt_start - cur_end)
        cur_len = len(cur_text)
        nxt_len = len(nxt_text)
        merged_len = cur_len + nxt_len
        merged_dur = max(cur_end, nxt_end) - cur_start

        should_merge = False
        if gap <= 0.45 and merged_len <= 15 and merged_dur <= 5.0:
            if cur_len <= 2 or nxt_len <= 2:
                should_merge = True
            elif not _ends_with_punctuation(cur_text):
                should_merge = True

        if should_merge:
            joiner = " " if _needs_space_between(cur_text, nxt_text) else ""
            current["text"] = f"{cur_text}{joiner}{nxt_text}"
            current["end"] = max(cur_end, nxt_end)
        else:
            if cur_text:
                current["text"] = cur_text
                merged.append(current)
            current = dict(nxt)

    last_text = (current.get("text") or "").strip()
    if last_text:
        current["text"] = last_text
        merged.append(current)

    if len(merged) <= 1:
        return merged

    # Second pass: fold ultra-short (1-char) residual fragments to neighbors when time-adjacent.
    folded = []
    i = 0
    while i < len(merged):
        cur = dict(merged[i])
        cur_text = (cur.get("text") or "").strip()
        cur_start = float(cur.get("start", 0.0))
        cur_end = float(cur.get("end", cur_start))

        if len(cur_text) <= 1 and folded:
            prev = folded[-1]
            prev_end = float(prev.get("end", prev.get("start", 0.0)))
            if cur_start - prev_end <= 0.5 and len((prev.get("text") or "").strip()) <= 24:
                joiner = " " if _needs_space_between((prev.get("text") or ""), cur_text) else ""
                prev["text"] = f"{(prev.get('text') or '').strip()}{joiner}{cur_text}"
                prev["end"] = max(prev_end, cur_end)
                folded[-1] = prev
                i += 1
                continue

        if len(cur_text) <= 1 and i + 1 < len(merged):
            nxt = dict(merged[i + 1])
            nxt_text = (nxt.get("text") or "").strip()
            nxt_start = float(nxt.get("start", cur_end))
            if nxt_start - cur_end <= 0.5 and len(nxt_text) <= 24:
                joiner = " " if _needs_space_between(cur_text, nxt_text) else ""
                nxt["text"] = f"{cur_text}{joiner}{nxt_text}"
                nxt["start"] = min(cur_start, nxt_start)
                merged[i + 1] = nxt
                i += 1
                continue

        folded.append(cur)
        i += 1

    return folded


class DoubaoASR:
    """豆包语音API封装类"""
    
    def __init__(self, appid: str, access_token: str):
        """
        初始化豆包ASR
        
        Args:
            appid: 应用标识
            access_token: 访问令牌
        """
        self.appid = appid
        self.access_token = access_token
        self.base_url = "https://openspeech.bytedance.com/api/v1/vc"
        self.headers = {
            "Authorization": f"Bearer; {access_token}",
            "Content-Type": "application/json"
        }
    
    def submit_audio(self, audio_data: Optional[bytes] = None, audio_url: Optional[str] = None, 
                     language: str = "zh-CN", caption_type: str = "auto",
                     words_per_line: int = 15, max_lines: int = 1) -> Dict:
        """
        提交音频文件
        
        Args:
            audio_data: 音频二进制数据（WAV格式，16kHz采样率）
            audio_url: 音频文件URL
            language: 字幕语言类型，默认zh-CN
            caption_type: 字幕识别类型，默认auto（同时识别说话和唱歌部分）
            words_per_line: 每行最多展示字数，默认15
            max_lines: 每屏最多展示行数，默认1
            
        Returns:
            包含任务ID的响应字典
        """
        if not (audio_data or audio_url):
            raise ValueError("必须提供audio_data或audio_url")
        
        params = {
            "appid": self.appid,
            "language": language,
            "caption_type": caption_type,
            "words_per_line": words_per_line,
            "max_lines": max_lines
        }
        
        if audio_url:
            # 使用音频URL方式
            data = {"url": audio_url}
            response = requests.post(
                f"{self.base_url}/submit",
                params=params,
                json=data,
                headers=self.headers,
                timeout=30
            )
        else:
            # 使用音频二进制方式
            headers = self.headers.copy()
            headers["Content-Type"] = "audio/wav"
            response = requests.post(
                f"{self.base_url}/submit",
                params=params,
                data=audio_data,
                headers=headers,
                timeout=60
            )
        
        response.raise_for_status()
        return response.json()
    
    def query_result(self, job_id: str, blocking: int = 1) -> Dict:
        """
        查询识别结果
        
        Args:
            job_id: 任务ID
            blocking: 查询结果时是否阻塞，0表示非阻塞，1表示阻塞（默认）
            
        Returns:
            包含识别结果的响应字典
        """
        params = {
            "appid": self.appid,
            "id": job_id,
            "blocking": blocking
        }
        
        response = requests.get(
            f"{self.base_url}/query",
            params=params,
            headers=self.headers,
            timeout=60
        )
        
        response.raise_for_status()
        return response.json()
    
    def recognize(self, audio_data: Optional[bytes] = None, audio_url: Optional[str] = None, 
                  language: str = "zh-CN", caption_type: str = "auto",
                  words_per_line: int = 15, max_lines: int = 1, 
                  max_retries: int = 5, retry_interval: int = 3) -> Dict:
        """
        完整的识别流程
        
        Args:
            audio_data: 音频二进制数据（WAV格式，16kHz采样率）
            audio_url: 音频文件URL
            language: 字幕语言类型，默认zh-CN
            caption_type: 字幕识别类型，默认auto
            words_per_line: 每行最多展示字数，默认15
            max_lines: 每屏最多展示行数，默认1
            max_retries: 最大重试次数，默认5
            retry_interval: 重试间隔（秒），默认3
            
        Returns:
            包含识别结果的响应字典
        """
        # 提交音频获取任务ID
        submit_retries = 0
        while submit_retries < max_retries:
            try:
                submit_response = self.submit_audio(
                    audio_data=audio_data,
                    audio_url=audio_url,
                    language=language,
                    caption_type=caption_type,
                    words_per_line=words_per_line,
                    max_lines=max_lines
                )
                
                if submit_response.get("code") != 0:
                    error_message = f"提交音频失败: {submit_response.get('message', '未知错误')}"
                    if submit_retries >= max_retries - 1:
                        raise Exception(error_message)
                    submit_retries += 1
                    time.sleep(retry_interval)
                    continue
                
                job_id = submit_response.get("id")
                if not job_id:
                    error_message = "未获取到任务ID"
                    if submit_retries >= max_retries - 1:
                        raise Exception(error_message)
                    submit_retries += 1
                    time.sleep(retry_interval)
                    continue
                
                break
            except requests.exceptions.RequestException as e:
                # 网络错误，进行重试
                if submit_retries >= max_retries - 1:
                    raise Exception(f"网络错误：{str(e)}")
                submit_retries += 1
                time.sleep(retry_interval)
                continue
            except Exception as e:
                # 其他错误，直接抛出
                raise
        
        # 查询结果
        retries = 0
        while retries < max_retries:
            try:
                result = self.query_result(job_id)
                if result.get("code") == 0:
                    return result
                elif result.get("code") == 2000:
                    # 任务处理中，继续重试
                    time.sleep(retry_interval)
                    retries += 1
                else:
                    error_message = f"查询结果失败: {result.get('message', '未知错误')}"
                    if retries >= max_retries - 1:
                        raise Exception(error_message)
                    retries += 1
                    time.sleep(retry_interval)
                    continue
            except requests.exceptions.RequestException as e:
                # 网络错误，进行重试
                if retries >= max_retries - 1:
                    raise Exception(f"网络错误：{str(e)}")
                retries += 1
                time.sleep(retry_interval)
                continue
            except Exception as e:
                # 其他错误，直接抛出
                raise
        
        raise Exception("查询结果超时")


def format_result(doubao_result: Dict) -> Dict:
    """
    将豆包API返回的结果格式转换为现有ASR结果格式
    
    Args:
        doubao_result: 豆包API返回的结果
        
    Returns:
        转换后的ASR结果格式
    """
    if doubao_result.get("code") != 0:
        return {
            "engine": "doubao",
            "text": "",
            "words": [],
            "sentences": [],
            "language": "zh",
            "error": doubao_result.get("message", "未知错误")
        }
    
    utterances = doubao_result.get("utterances", [])
    words = []
    sentences = []
    text = ""
    
    for utterance in utterances:
        # 处理句子
        sentence_text = utterance.get("text", "")
        if sentence_text:
            start_time = utterance.get("start_time", 0) / 1000.0  # 毫秒转秒
            end_time = utterance.get("end_time", 0) / 1000.0
            sentences.append({
                "text": sentence_text,
                "start": start_time,
                "end": end_time
            })
            text += sentence_text
        
        # 处理词
        utterance_words = utterance.get("words", [])
        for word in utterance_words:
            word_text = word.get("text", "")
            if word_text:
                start_time = word.get("start_time", 0) / 1000.0  # 毫秒转秒
                end_time = word.get("end_time", 0) / 1000.0
                words.append({
                    "word": word_text,
                    "start": start_time,
                    "end": end_time
                })
    
    merged_sentences = _merge_sentence_fragments(sentences)

    return {
        "engine": "doubao",
        "text": text,
        "words": words,
        "sentences": merged_sentences,
        "language": "zh",
        "duration": doubao_result.get("duration", 0)
    }

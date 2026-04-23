#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只看界面布局"""
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import time
import os

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_video_60s_final.mp4"


def screenshot(page, name):
    path = os.path.join(r"D:\video_clip\output\playwright", f"debug_layout_{name}.png")
    page.screenshot(path=path, full_page=True)
    print(f"  [截图] {path}")
    return path


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    page.goto(APP_URL, wait_until="load", timeout=30000)
    time.sleep(3)

    # 切换到Tab1
    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            break
    time.sleep(2)

    # 上传视频
    file_input = page.query_selector("input[type='file']")
    if file_input:
        file_input.set_input_files(TEST_VIDEO)
    time.sleep(5)

    # 往下滚动
    print("往下滚动...")
    page.evaluate("window.scrollTo(0, window.document.body.scrollHeight)")
    time.sleep(2)
    screenshot(page, "full")

    browser.close()
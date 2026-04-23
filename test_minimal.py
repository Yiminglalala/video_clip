#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""极简测试 - 只跑关键步骤"""
from playwright.sync_api import sync_playwright
import time
import os

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_video_60s_final.mp4"
OUTPUT_DIR = r"D:\video_clip\output"


def screenshot(page, name):
    path = os.path.join(r"D:\video_clip\output\playwright", f"test_final_{name}.png")
    page.screenshot(path=path, full_page=True)
    print(f"  [screenshot] {path}")
    return path


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    print("\n=== [1] 打开页面 & 切换Tab1 ===")
    page.goto(APP_URL, wait_until="load", timeout=30000)
    time.sleep(3)

    # 切换Tab1
    page.click('text=🎬 视频切片')
    time.sleep(2)
    screenshot(page, "01_tab1")

    # 上传视频
    print("\n=== [2] 上传视频 ===")
    page.set_input_files('input[type="file"]', TEST_VIDEO)
    time.sleep(6)
    screenshot(page, "02_uploaded")

    # 找到"生成字幕"复选框 - 直接遍历所有checkbox
    print("\n=== [3] 勾选生成字幕 ===")
    checkboxes = page.query_selector_all('input[type="checkbox"]')
    for cb in checkboxes:
        parent_text = cb.evaluate('(el) => el.parentElement.textContent')
        print(f"  - checkbox parent: {repr(parent_text[:80])}")
        if '生成字幕' in parent_text:
            cb.click()
            print("  ✓ 已勾选'生成字幕'")
            break
    time.sleep(1)
    screenshot(page, "03_checked")

    # 点击"下一步"
    print("\n=== [4] 下一步到Step1 ===")
    page.click('button:has-text("下一步")')
    time.sleep(3)
    screenshot(page, "04_step1")

    # 点击"开始处理"
    print("\n=== [5] 开始处理 ===")
    page.click('button:has-text("开始处理")')
    screenshot(page, "05_processing")

    # 等待处理完成（最多3分钟）
    print("\n=== [6] 等待处理完成 ===")
    waited = 0
    while waited < 180:
        time.sleep(10)
        waited += 10
        print(f"  {waited}s...")
        body = page.inner_text("body")
        if "下一步：预览与编辑" in body:
            print("  ✓ 分析完成！")
            screenshot(page, "06_done")
            break
        screenshot(page, f"06_wait_{waited}s")

    # 直接查看最新的输出目录
    print("\n=== [7] 检查最近的输出目录 ===")
    if os.path.exists(OUTPUT_DIR):
        dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))],
                      key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True)
        if dirs:
            latest_dir = dirs[0]
            print(f"  ✓ 最新目录: {latest_dir}")
            full_path = os.path.join(OUTPUT_DIR, latest_dir)
            for f in os.listdir(full_path):
                print(f"    - {f}")

    print("\n=== 测试结束，浏览器保持打开 ===")
    # 保持浏览器打开
    time.sleep(60)

    browser.close()
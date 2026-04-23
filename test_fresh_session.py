# -*- coding: utf-8 -*-
"""Fresh session test: verify the complete workflow from step 0 to step 3."""
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_chenchusheng_10min.mp4"

def screenshot(page, name):
    path = f"D:/video_clip/output/playwright/e2e_fresh_{name}.png"
    page.screenshot(path=path)
    print(f"  [screenshot] {path}")
    return path

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    print("=== [1] Open ===")
    page.goto(APP_URL, wait_until="load", timeout=30000)
    page.wait_for_timeout(3000)

    print("=== [2] Tab1 ===")
    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            break
    page.wait_for_timeout(2000)
    screenshot(page, "01_tab1")

    print("=== [3] Upload ===")
    file_inputs = page.query_selector_all("input[type='file']")
    file_inputs[0].set_input_files(TEST_VIDEO)
    page.wait_for_timeout(3000)
    screenshot(page, "02_uploaded")

    body = page.inner_text("body")
    if "时长" in body:
        print("  ✓ Video info shown")
    else:
        print("  ✗ No video info")

    print("=== [4] Next to step1 ===")
    for btn in page.query_selector_all("button"):
        if "下一步" in btn.inner_text() and "开始处理" in btn.inner_text():
            btn.click()
            break
    page.wait_for_timeout(3000)
    screenshot(page, "03_step1")

    body = page.inner_text("body")
    if "步骤 2" in body:
        print("  ✓ Step1 reached")
    else:
        print("  ✗ Step1 not reached")

    print("=== [5] Start processing ===")
    for btn in page.query_selector_all("button"):
        if "开始处理" in btn.inner_text():
            btn.click()
            print("  ✓ Clicked start")
            break
    page.wait_for_timeout(2000)
    screenshot(page, "04_processing")

    print("=== [6] Wait for completion (max 600s) ===")
    import time
    for waited in range(0, 600, 20):
        time.sleep(20)
        screenshot(page, f"05_wait_{waited+20}s")

        if page.query_selector_all("text=步骤 3: 预览编辑"):
            print(f"  ✓ Reached step 2 at ~{waited+20}s!")
            break

        alerts = page.query_selector_all("[data-testid='stAlert']")
        for al in alerts:
            txt = al.inner_text()
            if "错误" in txt or "失败" in txt:
                print(f"  ⚠ Error: {txt[:80]}")
                break

        spinners = page.query_selector_all("[data-testid='stSpinner']")
        progress = page.query_selector_all("[data-testid='stProgress']")
        print(f"  ... {waited+20}s (spinners={len(spinners)}, progress={len(progress)})")

    screenshot(page, "06_final")

    body = page.inner_text("body")
    print("\n=== Final page key lines ===")
    for line in body.split("\n"):
        line = line.strip()
        if any(k in line for k in ["步骤", "时间轴", "预览", "导出", "错误", "失败"]):
            print(f"  {repr(line[:80])}")

    print("\n=== Checking server output folder ===")
    import os, glob
    base = r"D:\video_clip\output"
    folders = sorted([d for d in os.listdir(base) if d.startswith("upload_")], reverse=True)
    if folders:
        latest = folders[0]
        print(f"  Latest folder: {latest}")
        files = glob.glob(os.path.join(base, latest, "**", "*.mp4"), recursive=True)
        print(f"  MP4 files: {len(files)}")

    browser.close()

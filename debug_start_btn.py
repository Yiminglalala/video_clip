# -*- coding: utf-8 -*-
"""Verify start button click and capture state."""
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_chenchusheng_10min.mp4"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(APP_URL, wait_until="load", timeout=30000)
    page.wait_for_timeout(3000)

    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            break
    page.wait_for_timeout(2000)

    file_inputs = page.query_selector_all("input[type='file']")
    file_inputs[0].set_input_files(TEST_VIDEO)
    page.wait_for_timeout(3000)

    buttons = page.query_selector_all("button")
    for btn in buttons:
        txt = btn.inner_text().strip()
        if "下一步" in txt and "开始处理" in txt:
            btn.click()
            break
    page.wait_for_timeout(3000)

    page.screenshot(path="D:/video_clip/output/playwright/debug_step1_ready.png")

    print("=== All buttons on step1 ===")
    for btn in page.query_selector_all("button"):
        txt = btn.inner_text().strip()
        if txt:
            # Check if visible and enabled
            visible = btn.is_visible()
            enabled = btn.is_enabled()
            print(f"  [{txt[:60]}] visible={visible}, enabled={enabled}")

    print("\n=== Looking for 开始处理 button ===")
    for btn in page.query_selector_all("button"):
        txt = btn.inner_text().strip()
        if "开始处理" in txt:
            print(f"  FOUND: [{txt[:60]}]")
            print(f"  is_visible={btn.is_visible()}, is_enabled={btn.is_enabled()}")
            # Try clicking it
            btn.click()
            print("  Clicked!")
            page.wait_for_timeout(2000)
            page.screenshot(path="D:/video_clip/output/playwright/debug_after_click.png")
            body = page.inner_text("body")
            lines = [l.strip() for l in body.split("\n") if l.strip()]
            print("\n=== Key lines after click ===")
            for l in lines[:30]:
                if any(k in l for k in ["步骤", "处理", "分析", "错误", "GPU", "CUDA", "开始", "完成"]):
                    print(f"  {repr(l[:80])}")
            break

    browser.close()

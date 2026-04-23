# -*- coding: utf-8 -*-
"""Debug step2 page."""
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

    for btn in page.query_selector_all("button"):
        if "下一步" in btn.inner_text() and "开始处理" in btn.inner_text():
            btn.click()
            break
    page.wait_for_timeout(3000)

    for btn in page.query_selector_all("button"):
        if "开始处理" in btn.inner_text():
            btn.click()
            break
    page.wait_for_timeout(1000)

    # Wait for "下一步：预览与编辑" to appear
    import time
    for waited in range(0, 300, 20):
        time.sleep(20)
        if page.query_selector_all("text=下一步：预览与编辑"):
            print(f"Found at ~{waited}s!")
            break
        print(f"Waiting... {waited}s")

    # Click it
    for btn in page.query_selector_all("button"):
        if "下一步" in btn.inner_text() and "预览" in btn.inner_text():
            btn.click()
            print("Clicked!")
            break
    page.wait_for_timeout(3000)
    page.screenshot(path="D:/video_clip/output/playwright/debug_step2.png")

    body = page.inner_text("body")
    print("\n=== All text on step2 page ===")
    for line in body.split("\n"):
        line = line.strip()
        if line:
            print(repr(line[:80]))

    print("\n=== All buttons ===")
    for btn in page.query_selector_all("button"):
        txt = btn.inner_text().strip()
        if txt:
            print(f"  [{txt[:60]}]")

    browser.close()

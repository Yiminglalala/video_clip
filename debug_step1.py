# -*- coding: utf-8 -*-
"""Debug step1 page content."""
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

    page.screenshot(path="D:/video_clip/output/playwright/debug_step1_check.png")

    body = page.inner_text("body")
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    print("All text on step1 page:")
    for l in lines[:80]:
        print(repr(l[:80]))

    print("\n\nAll buttons:")
    for btn in page.query_selector_all("button"):
        txt = btn.inner_text().strip()
        if txt:
            print(f"  [{txt[:60]}]")

    browser.close()

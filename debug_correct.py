# -*- coding: utf-8 -*-
"""Focused debug - correct tab1 input selection."""
from playwright.sync_api import sync_playwright
import os

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_chenchusheng_10min.mp4"

print("File exists:", os.path.exists(TEST_VIDEO))

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

    ti_inputs = page.query_selector_all("[data-testid='stTextInput'] input")
    print(f"Total text inputs: {len(ti_inputs)}")
    tab1_path = None
    for inp in ti_inputs:
        al = inp.get_attribute("aria-label")
        print(f"  aria-label={repr(al)}")
        if al and "或粘贴本地路径" in al:
            tab1_path = inp
            break

    if tab1_path:
        print("Found tab1 path input!")
        tab1_path.fill(TEST_VIDEO)
        page.wait_for_timeout(500)
        val = tab1_path.get_attribute("value")
        print(f"Filled value: {repr(val[:60])}")

        buttons = page.query_selector_all("button")
        clicked = False
        for btn in buttons:
            txt = btn.inner_text()
            if "使用本地文件" in txt:
                print(f"Found button: {repr(txt[:30])}")
                btn.click()
                clicked = True
                page.wait_for_timeout(5000)
                break

        if clicked:
            page.screenshot(path="D:/video_clip/output/playwright/debug_correct.png")
            body = page.inner_text("body")
            lines = [l.strip() for l in body.split("\n") if l.strip()]
            print("\nKey lines after click:")
            for l in lines:
                if any(k in l for k in ["时长", "分辨率", "请上传", "文件不存在", "下一步", "步骤", "歌手"]):
                    print(f"  KEY: {repr(l[:80])}")
    else:
        print("ERROR: Could not find tab1 path input!")

    browser.close()

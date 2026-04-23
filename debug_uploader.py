# -*- coding: utf-8 -*-
"""Debug: test file uploader vs text path approach."""
from playwright.sync_api import sync_playwright
import os

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_chenchusheng_10min.mp4"

print("File exists:", os.path.exists(TEST_VIDEO))

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    print("\n=== [1] Open app ===")
    page.goto(APP_URL, wait_until="load", timeout=30000)
    page.wait_for_timeout(3000)

    # Click tab1
    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            break
    page.wait_for_timeout(2000)
    page.screenshot(path="D:/video_clip/output/playwright/debug_a1.png")

    # === APPROACH 1: Text input + use_local ===
    print("\n=== [2] Test text input approach ===")
    ti_inputs = page.query_selector_all("[data-testid='stTextInput'] input")
    tab1_path = None
    for inp in ti_inputs:
        if "或粘贴本地路径" in (inp.get_attribute("aria-label") or ""):
            tab1_path = inp
            break

    if tab1_path:
        tab1_path.fill(TEST_VIDEO)
        page.wait_for_timeout(1000)
        print(f"Filled: {tab1_path.get_attribute('value')[:50]}")

        buttons = page.query_selector_all("button")
        clicked = False
        for btn in buttons:
            if "使用本地文件" in btn.inner_text():
                btn.click()
                clicked = True
                page.wait_for_timeout(5000)
                break

        if clicked:
            page.screenshot(path="D:/video_clip/output/playwright/debug_a2.png")
            body = page.inner_text("body")
            if "时长" in body:
                print("SUCCESS: Video info shown!")
            else:
                print("FAILURE: No video info. Checking why...")
                for line in body.split("\n"):
                    if any(k in line for k in ["请上传", "文件不存在", "时长", "步骤"]):
                        print(f"  {repr(line[:80])}")

    # === APPROACH 2: File uploader ===
    print("\n=== [3] Test file uploader approach ===")
    # Go back to step 0 by reloading
    page.goto(APP_URL, wait_until="load", timeout=30000)
    page.wait_for_timeout(2000)

    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            break
    page.wait_for_timeout(2000)

    # Find all file inputs
    file_inputs = page.query_selector_all("input[type='file']")
    print(f"File inputs found: {len(file_inputs)}")
    for i, fi in enumerate(file_inputs):
        al = fi.get_attribute("aria-label")
        print(f"  [{i}] aria-label={repr(al)}")

    # Try to set files on the first file input (tab1 uploader)
    if file_inputs:
        # The tab1 upload button has aria-label containing "upload" or similar
        # Let's try setting files on all file inputs and see
        for i, fi in enumerate(file_inputs):
            try:
                fi.set_input_files(TEST_VIDEO)
                print(f"  Set files on input [{i}]")
            except Exception as e:
                print(f"  Failed on [{i}]: {e}")

        page.wait_for_timeout(5000)
        page.screenshot(path="D:/video_clip/output/playwright/debug_a3.png")
        body = page.inner_text("body")
        if "时长" in body:
            print("SUCCESS: Video info shown via file upload!")
        else:
            print("FAILURE: No video info from file upload")
            for line in body.split("\n"):
                if any(k in line for k in ["请上传", "文件不存在", "时长", "步骤"]):
                    print(f"  {repr(line[:80])}")

    browser.close()

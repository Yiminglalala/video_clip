# -*- coding: utf-8 -*-
"""Debug file upload failure."""
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_chenchusheng_10min.mp4"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(APP_URL, wait_until="load", timeout=30000)
    page.wait_for_timeout(3000)

    # First click tab1
    tabs = page.query_selector_all("[data-testid='stTab']")
    print(f"Tabs found: {len(tabs)}")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            print("Clicked tab1")
    page.wait_for_timeout(2000)

    # Find file inputs
    file_inputs = page.query_selector_all("input[type='file']")
    print(f"File inputs after tab1 click: {len(file_inputs)}")
    for i, fi in enumerate(file_inputs):
        al = fi.get_attribute("aria-label")
        print(f"  [{i}] aria-label={repr(al)}, visible={fi.is_visible()}")

    # Try uploading
    if file_inputs:
        try:
            file_inputs[0].set_input_files(TEST_VIDEO)
            print("Upload successful!")
        except Exception as e:
            print(f"Upload failed: {e}")
            # Try with second input
            if len(file_inputs) > 1:
                try:
                    file_inputs[1].set_input_files(TEST_VIDEO)
                    print("Upload on input[1] successful!")
                except Exception as e2:
                    print(f"Upload on input[1] also failed: {e2}")

    page.wait_for_timeout(3000)
    page.screenshot(path="D:/video_clip/output/playwright/debug_upload_check.png")

    body = page.inner_text("body")
    if "时长" in body:
        print("SUCCESS: Video info shown!")
    else:
        print("FAIL: No video info")
        for line in body.split("\n"):
            line = line.strip()
            if line and any(k in line for k in ["请上传", "步骤", "时长"]):
                print(f"  {repr(line[:80])}")

    browser.close()

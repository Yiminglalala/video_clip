# -*- coding: utf-8 -*-
"""Focused debug: trace exactly what happens at each step."""
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_chenchusheng_10min.mp4"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    print("=== [1] Open ===")
    page.goto(APP_URL, wait_until="load", timeout=30000)
    page.wait_for_timeout(3000)
    page.screenshot(path="D:/video_clip/output/playwright/debug_a1_home.png")

    print("=== [2] Click tab1 ===")
    tabs = page.query_selector_all("[data-testid='stTab']")
    print(f"  Found {len(tabs)} tabs")
    for t in tabs:
        print(f"  Tab: {repr(t.inner_text()[:30])}")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            print("  Clicked tab1")
    page.wait_for_timeout(2000)
    page.screenshot(path="D:/video_clip/output/playwright/debug_a2_after_tab_click.png")

    print("=== [3] Query inputs ===")
    all_inputs = page.query_selector_all("input")
    print(f"  Total inputs: {len(all_inputs)}")
    for i, inp in enumerate(all_inputs):
        print(f"  [{i}] type={inp.get_attribute('type')}, aria-label={repr(inp.get_attribute('aria-label'))}, value={repr(inp.get_attribute('value') or '')[:20]}")

    print("=== [4] Query text inputs ===")
    ti_inputs = page.query_selector_all("[data-testid='stTextInput'] input")
    print(f"  Text inputs: {len(ti_inputs)}")
    for i, inp in enumerate(ti_inputs):
        al = inp.get_attribute("aria-label")
        val = inp.get_attribute("value")
        print(f"  [{i}] aria-label={repr(al)}, value={repr(val or '')[:30]}")

    print("=== [5] Fill first text input ===")
    if ti_inputs:
        inp = ti_inputs[0]
        inp.fill(TEST_VIDEO)
        page.wait_for_timeout(1000)
        print(f"  Filled: {repr(inp.get_attribute('value') or '')[:50]}")
        page.screenshot(path="D:/video_clip/output/playwright/debug_a5_after_fill.png")

    print("=== [6] Query buttons ===")
    buttons = page.query_selector_all("button")
    print(f"  Total buttons: {len(buttons)}")
    for i, btn in enumerate(buttons):
        txt = btn.inner_text().replace('\n', ' ').strip()
        if txt:
            print(f"  [{i}] {repr(txt[:50])}")

    print("=== [7] Click first '使用本地文件' ===")
    clicked = False
    for btn in buttons:
        if "使用本地文件" in btn.inner_text():
            btn.click()
            clicked = True
            print("  Clicked!")
            break
    if not clicked:
        print("  NOT FOUND!")
    page.wait_for_timeout(4000)
    page.screenshot(path="D:/video_clip/output/playwright/debug_a7_after_use_local.png")

    print("=== [8] Check page text ===")
    body_text = page.inner_text("body")
    lines = [l.strip() for l in body_text.split('\n') if l.strip()]
    for l in lines[:50]:
        print(f"  {repr(l[:80])}")

    browser.close()

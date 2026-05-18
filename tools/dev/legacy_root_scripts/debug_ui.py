# -*- coding: utf-8 -*-
"""Debug script to inspect tab1 UI elements."""
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8507"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(APP_URL, wait_until="networkidle", timeout=30000)
    import time; time.sleep(2)

    # Click tab1
    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            break
    time.sleep(2)

    # Get all inputs
    inputs = page.query_selector_all("input")
    print("=== INPUTS ===")
    for inp in inputs:
        print(f"  type={inp.get_attribute('type')}, placeholder={inp.get_attribute('placeholder')}, aria-label={inp.get_attribute('aria-label')}, value={repr(inp.get_attribute('value'))[:30]}")

    # Get all buttons
    buttons = page.query_selector_all("button")
    print("\n=== BUTTONS ===")
    for btn in buttons:
        txt = btn.inner_text()
        print(f"  text={repr(txt[:60])}")

    # Get all text inputs by looking for stTextInput
    text_inputs = page.query_selector_all("[data-testid='stTextInput'] input")
    print(f"\n=== TEXT INPUTS (stTextInput) ===")
    for inp in text_inputs:
        print(f"  placeholder={inp.get_attribute('placeholder')}, aria-label={inp.get_attribute('aria-label')}")

    browser.close()

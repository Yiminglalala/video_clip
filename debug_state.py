# -*- coding: utf-8 -*-
"""Quick check: what's on the page after 5 minutes of processing?"""
from playwright.sync_api import sync_playwright
import os, time

APP_URL = "http://localhost:8507"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    print("Opening app...")
    page.goto(APP_URL, wait_until="load", timeout=30000)
    page.wait_for_timeout(3000)

    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            break
    page.wait_for_timeout(2000)

    # Check current step indicator
    body = page.inner_text("body")
    lines = [l.strip() for l in body.split("\n") if l.strip()]
    print("\n=== Page text (key lines) ===")
    for l in lines:
        if any(k in l for k in ["步骤", "步骤", "处理", "分析", "错误", "失败", "完成", "预览", "导出", "时间轴"]):
            print(f"  {repr(l[:80])}")

    # Screenshot
    page.screenshot(path="D:/video_clip/output/playwright/e2e_state_check.png")
    print("Screenshot saved.")

    # Check for any error alerts
    alerts = page.query_selector_all("[data-testid='stAlert']")
    print(f"\nAlerts found: {len(alerts)}")
    for al in alerts:
        print(f"  Alert: {al.inner_text()[:100]}")

    # Check for spinner
    spinners = page.query_selector_all("[data-testid='stSpinner']")
    print(f"Spinners: {len(spinners)}")

    # Check session state via URL
    print(f"\nURL: {page.url}")

    browser.close()

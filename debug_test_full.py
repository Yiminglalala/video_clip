#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速调试测试 - 跑一次完整流程看ASR和字幕"""
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import time
import os

APP_URL = "http://localhost:8507"
TEST_VIDEO = r"D:\video_clip\test_video_60s_final.mp4"


def screenshot(page, name):
    path = os.path.join(r"D:\video_clip\output\playwright", f"debug_e2e_{name}.png")
    page.screenshot(path=path)
    print(f"  [截图] {path}")
    return path


def find_elem_by_text(page, text, timeout=10000):
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except PlaywrightTimeout:
        return False


with sync_playwright() as p:
    browser = p.chromium.launch(headless=False, args=["--disable-gpu", "--no-sandbox"])
    page = browser.new_page(viewport={"width": 1920, "height": 1080})

    print("\n=== [1] 打开页面 ===")
    page.goto(APP_URL, wait_until="load", timeout=30000)
    time.sleep(3)
    screenshot(page, "01_start")

    # 切换到Tab1
    print("\n=== [2] 切换到视频切片 ===")
    tabs = page.query_selector_all("[data-testid='stTab']")
    for t in tabs:
        if "视频切片" in t.inner_text():
            t.click()
            print("  ✓ 已点击'视频切片'")
            break
    time.sleep(2)
    screenshot(page, "02_tab1")

    # 上传视频
    print("\n=== [3] 上传视频 ===")
    file_input = page.query_selector("input[type='file']")
    if file_input:
        file_input.set_input_files(TEST_VIDEO)
        print("  ✓ 已上传视频")
        time.sleep(5)
        screenshot(page, "03_uploaded")
    else:
        print("  ✗ 未找到文件上传控件")
        browser.close()
        exit(1)

    # 确认视频信息
    print("\n=== [4] 确认视频信息 ===")
    page.wait_for_timeout(2000)
    body = page.inner_text("body")
    if "时长" in body:
        print("  ✓ 视频信息已加载")
        screenshot(page, "04_info")
    else:
        print("  ✗ 视频信息未加载")

    # 勾选"生成字幕"
    print("\n=== [5] 勾选生成字幕 ===")
    checkboxes = page.query_selector_all("input[type='checkbox']")
    for cb in checkboxes:
        parent = cb.query_selector("xpath=..")
        if parent and "生成字幕" in parent.inner_text():
            cb.click()
            print("  ✓ 已勾选'生成字幕'")
            break
    time.sleep(1)
    screenshot(page, "05_subtitle_checked")

    # 进入步骤1
    print("\n=== [6] 进入步骤1 ===")
    buttons = page.query_selector_all("button")
    for btn in buttons:
        txt = btn.inner_text().strip()
        if "下一步" in txt and "步骤 1" in txt:
            btn.click()
            print("  ✓ 已点击'下一步：步骤 1'")
            break
    time.sleep(3)
    screenshot(page, "06_step1")

    # 开始处理
    print("\n=== [7] 开始处理 ===")
    time.sleep(2)
    buttons = page.query_selector_all("button")
    for btn in buttons:
        txt = btn.inner_text().strip()
        if "开始处理" in txt:
            btn.click()
            print("  ✓ 已点击'开始处理'")
            break
    screenshot(page, "07_processing_start")

    # 等待处理完成（最多3分钟）
    print("\n=== [8] 等待处理完成 ===")
    waited = 0
    done = False
    while waited < 180:
        time.sleep(5)
        waited += 5
        print(f"  等待了 {waited} 秒...")
        body = page.inner_text("body")
        if "步骤 2" in body or "下一步：预览与编辑" in body:
            done = True
            print("  ✓ 处理完成！")
            screenshot(page, f"08_done_{waited}s")
            break
        screenshot(page, f"08_waiting_{waited}s")

    if not done:
        print("  ⚠ 超时，检查当前状态")
        screenshot(page, "08_timeout")
        browser.close()
        exit(1)

    # 进入步骤2
    print("\n=== [9] 进入步骤2 ===")
    buttons = page.query_selector_all("button")
    for btn in buttons:
        txt = btn.inner_text().strip()
        if "下一步：预览与编辑" in txt:
            btn.click()
            print("  ✓ 已进入步骤2")
            break
    time.sleep(3)
    screenshot(page, "09_step2")

    # 进入步骤3
    print("\n=== [10] 进入步骤3 ===")
    buttons = page.query_selector_all("button")
    for btn in buttons:
        txt = btn.inner_text().strip()
        if "下一步：导出视频" in txt:
            btn.click()
            print("  ✓ 已进入步骤3")
            break
    time.sleep(3)
    screenshot(page, "10_step3")

    # 导出
    print("\n=== [11] 导出视频 ===")
    # 先截图看一下Step3的调试信息
    screenshot(page, "11_before_export")

    buttons = page.query_selector_all("button")
    for btn in buttons:
        txt = btn.inner_text().strip()
        if "导出视频" in txt:
            print(f"  点击'导出视频'")
            btn.click()
            time.sleep(5)
            screenshot(page, "11_exporting")

            # 等待导出完成（最多2分钟）
            exp_waited = 0
            while exp_waited < 120:
                time.sleep(5)
                exp_waited += 5
                body = page.inner_text("body")
                if "导出完成" in body or "成功" in body:
                    print("  ✓ 导出完成！")
                    screenshot(page, f"11_export_done_{exp_waited}s")
                    break
                print(f"  导出中... {exp_waited}秒")
                screenshot(page, f"11_exporting_{exp_waited}s")

            break

    time.sleep(5)
    screenshot(page, "12_final")

    print("\n=== 测试完成 ===")
    browser.close()
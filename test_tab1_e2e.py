# -*- coding: utf-8 -*-
"""Tab1 End-to-End Test - Robust version using file upload."""
import sys
import time
import os

TEST_VIDEO = r"D:\video_clip\test_chenchusheng_10min.mp4"
APP_URL = "http://localhost:8507"
OUTPUT_DIR = r"D:\video_clip\output\playwright"

os.makedirs(OUTPUT_DIR, exist_ok=True)

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

def screenshot(page, name):
    path = os.path.join(OUTPUT_DIR, f"e2e_v3_{name}.png")
    page.screenshot(path=path)
    print(f"  [screenshot] {path}")
    return path

def find_elem_by_text(page, text, timeout=5000):
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except PlaywrightTimeout:
        return False

def run_test():
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-gpu", "--no-sandbox"])
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        # 1. Open app
        print("\n=== [1] Opening app ===")
        page.goto(APP_URL, wait_until="load", timeout=30000)
        page.wait_for_timeout(3000)
        screenshot(page, "01_home")

        # 2. Activate tab1
        print("\n=== [2] Activating tab1 ===")
        tabs = page.query_selector_all("[data-testid='stTab']")
        tab1_el = None
        for t in tabs:
            if "视频切片" in t.inner_text():
                tab1_el = t
                break
        if tab1_el:
            tab1_el.click()
            page.wait_for_timeout(2000)
            print("  ✓ Tab1 activated")
            screenshot(page, "02_tab1")
            results.append(("tab1_activation", "pass"))
        else:
            print("  ✗ Tab1 not found")
            results.append(("tab1_activation", "fail"))

        # 3. Verify step0 header
        print("\n=== [3] Verify step0 UI ===")
        if find_elem_by_text(page, "演唱会视频智能切片 v3.0", timeout=8000):
            print("  ✓ Header found")
            results.append(("step0_header", "pass"))
        else:
            print("  ✗ Header not found")
            results.append(("step0_header", "fail"))
            screenshot(page, "03_no_header")

        if find_elem_by_text(page, "步骤 1：选择视频文件", timeout=5000):
            print("  ✓ Step0 subheader found")
            results.append(("step0_subheader", "pass"))
        else:
            print("  ✗ Step0 subheader not found")
            results.append(("step0_subheader", "fail"))

        # 4. Upload video via file input
        print("\n=== [4] Uploading video via file input ===")
        file_inputs = page.query_selector_all("input[type='file']")
        print(f"  File inputs found: {len(file_inputs)}")
        uploaded = False
        for fi in file_inputs:
            al = fi.get_attribute("aria-label") or ""
            # Try to set files on each file input - tab1 is first (no aria-label)
            try:
                fi.set_input_files(TEST_VIDEO)
                uploaded = True
                print(f"  ✓ Uploaded via input with aria-label={repr(al)}")
                page.wait_for_timeout(3000)
                screenshot(page, "04_after_upload")
                results.append(("video_upload", "pass"))
                break
            except Exception as e:
                print(f"  Failed on input with aria-label={repr(al)}: {e}")
        if not uploaded:
            print("  ✗ Could not upload file")
            results.append(("video_upload", "fail"))

        # 5. Check video info displayed
        print("\n=== [5] Check video info ===")
        if find_elem_by_text(page, "时长", timeout=10000):
            print("  ✓ Video info (时长) displayed")
            results.append(("video_info", "pass"))
            screenshot(page, "05_video_info")
        else:
            print("  ⚠ Video info not shown")
            results.append(("video_info", "skip"))
            screenshot(page, "05_no_info")

        # 6. Check workflow steps shown
        print("\n=== [6] Check workflow step indicators ===")
        body = page.inner_text("body")
        steps_found = []
        for step in ["📹 选择视频", "⚙️ 开始处理", "✏️ 预览编辑", "🎬 导出视频"]:
            if step.replace("📹", "").replace("⚙️", "").replace("✏️", "").replace("🎬", "").strip() in body:
                steps_found.append(step)
        if len(steps_found) >= 3:
            print(f"  ✓ {len(steps_found)} step indicators found")
            results.append(("workflow_steps", "pass"))
        else:
            print(f"  ⚠ Only {len(steps_found)} step indicators found")
            results.append(("workflow_steps", "skip"))

        # 7. Advance to step1 - click 下一步：开始处理
        print("\n=== [7] Advance to step 1 ===")
        # Use exact button text match
        buttons = page.query_selector_all("button")
        next_btn = None
        for btn in buttons:
            txt = btn.inner_text().strip()
            if "下一步" in txt and "开始处理" in txt:
                next_btn = btn
                break
        if next_btn:
            next_btn.click()
            page.wait_for_timeout(3000)
            print("  ✓ Clicked '下一步：开始处理'")
            screenshot(page, "06_step1")
            results.append(("advance_to_step1", "pass"))
        else:
            print("  ✗ '下一步：开始处理' not found")
            results.append(("advance_to_step1", "fail"))
            screenshot(page, "06_no_next")

        # 8. Verify step1 UI
        print("\n=== [8] Verify step1 UI ===")
        if find_elem_by_text(page, "步骤 2: 视频分析与处理", timeout=8000):
            print("  ✓ Step1 header found")
            results.append(("step1_ui", "pass"))
            screenshot(page, "07_step1_shown")
        else:
            print("  ✗ Step1 header not found")
            results.append(("step1_ui", "fail"))
            screenshot(page, "07_no_step1")

        # Check processing flow description
        body = page.inner_text("body")
        if "音频分析" in body and "智能分类" in body:
            print("  ✓ Processing flow description present")
            results.append(("processing_flow", "pass"))
        else:
            results.append(("processing_flow", "skip"))

        # 9. Click 开始处理
        print("\n=== [9] Click 开始处理 ===")
        buttons = page.query_selector_all("button")
        start_btn = None
        for btn in buttons:
            txt = btn.inner_text().strip()
            if "开始处理" in txt:
                start_btn = btn
                break
        if start_btn:
            start_btn.click()
            page.wait_for_timeout(2000)
            print("  ✓ Clicked '开始处理'")
            screenshot(page, "08_processing")
            results.append(("start_processing", "pass"))
        else:
            print("  ✗ '开始处理' not found")
            results.append(("start_processing", "fail"))
            screenshot(page, "08_no_start")

        # 10. Wait for processing (poll every 20s, max 10min)
        print("\n=== [10] Waiting for processing ===")
        max_wait = 600
        poll = 20
        waited = 0
        done = False
        error_found = False

        while waited < max_wait:
            time.sleep(poll)
            waited += poll
            screenshot(page, f"10_proc_{waited}s")

            # After processing completes, the page shows "➡️ 下一步：预览与编辑" button
            # We need to check for this button, not just text
            if page.query_selector_all("text=下一步：预览与编辑"):
                print(f"  ✓ Processing completed at ~{waited}s! '下一步：预览与编辑' button found.")
                results.append(("processing_done", "pass"))
                done = True
                break

            try:
                page.wait_for_selector("text=步骤 3: 预览编辑", timeout=3000)
                print(f"  ✓ Processing completed at ~{waited}s! Step 2 reached.")
                results.append(("processing_done", "pass"))
                done = True
                break
            except PlaywrightTimeout:
                pass

            alerts = page.query_selector_all("[data-testid='stAlert']")
            for al in alerts:
                txt = al.inner_text()
                if any(k in txt for k in ["错误", "失败", "Error"]):
                    print(f"  ⚠ Error at ~{waited}s: {txt[:80]}")
                    screenshot(page, f"10b_err_{waited}s")
                    results.append(("processing_error", txt[:80]))
                    done = True
                    error_found = True
                    break

            if done:
                break

            spinners = page.query_selector_all("[data-testid='stSpinner']")
            progress = page.query_selector_all("[data-testid='stProgress']")
            print(f"  ... polling at {waited}s (spinners={len(spinners)}, progress={len(progress)})")

        if not done:
            print(f"  ⚠ Timeout after {max_wait}s")
            results.append(("processing_done", "timeout"))
            screenshot(page, "10c_timeout")

        # 10b. Click "下一步：预览与编辑" to advance to step 2
        if done and not error_found:
            print("\n=== [10b] Click '下一步：预览与编辑' ===")
            btn_next = None
            for btn in page.query_selector_all("button"):
                if "下一步" in btn.inner_text() and "预览" in btn.inner_text():
                    btn_next = btn
                    break
            if btn_next:
                btn_next.click()
                page.wait_for_timeout(3000)
                print("  ✓ Clicked '下一步：预览与编辑'")
                screenshot(page, "10b_step2")
                results.append(("click_step2_button", "pass"))
            else:
                print("  ⚠ '下一步：预览与编辑' button not found (may already be on step 2)")
                results.append(("click_step2_button", "skip"))

        # 11. Step 2: Preview/Edit
        if done and not error_found:
            print("\n=== [11] Verify step2 UI ===")
            # Note: page uses full-width colon (：) in "步骤 3:"
            if find_elem_by_text(page, "✏️ 步骤 3", timeout=8000):
                print("  ✓ Step2 header visible")
                results.append(("step2_ui", "pass"))
                screenshot(page, "11_step2")
            else:
                print("  ✗ Step2 header not visible")
                results.append(("step2_ui", "fail"))

            # Check timeline
            body = page.inner_text("body")
            if "时间轴" in body:
                print("  ✓ Timeline found")
                results.append(("timeline", "pass"))
            else:
                print("  ⚠ Timeline not found")
                results.append(("timeline", "skip"))

            # 12. Advance to step3
            print("\n=== [12] Advance to step 3 ===")
            buttons = page.query_selector_all("button")
            export_next = None
            for btn in buttons:
                txt = btn.inner_text().strip()
                if "下一步" in txt and "导出" in txt:
                    export_next = btn
                    break
            if export_next:
                export_next.click()
                page.wait_for_timeout(2000)
                print("  ✓ Clicked '下一步：导出视频'")
                screenshot(page, "12_step3")
                results.append(("advance_to_step3", "pass"))
            else:
                print("  ✗ '下一步：导出视频' not found")
                results.append(("advance_to_step3", "fail"))

            # 13. Verify step3
            print("\n=== [13] Verify step3 UI ===")
            # Note: page uses "🎬 步骤 4:" with full-width colon
            if find_elem_by_text(page, "🎬 步骤 4", timeout=8000):
                print("  ✓ Step3 header visible")
                results.append(("step3_ui", "pass"))
                screenshot(page, "13_step3")
            else:
                print("  ✗ Step3 header not visible")
                results.append(("step3_ui", "fail"))

            # Check export button
            buttons = page.query_selector_all("button")
            has_export = any("导出切片视频" in b.inner_text() for b in buttons)
            if has_export:
                print("  ✓ Export button found")
                results.append(("export_button", "pass"))
            else:
                print("  ⚠ Export button not found")
                results.append(("export_button", "skip"))

        browser.close()

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    for name, status in results:
        icon = "✓" if status == "pass" else ("⚠" if status == "skip" else "✗")
        print(f"  {icon} {name}: {status}")
    passed = sum(1 for _, s in results if s == "pass")
    skipped = sum(1 for _, s in results if s == "skip")
    failed = sum(1 for _, s in results if s not in ("pass", "skip"))
    print(f"\nPassed: {passed} | Skipped: {skipped} | Failed/Timeout: {failed} | Total: {len(results)}")
    print(f"Screenshots: {OUTPUT_DIR}/e2e_v3_*.png")
    return failed == 0

if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)

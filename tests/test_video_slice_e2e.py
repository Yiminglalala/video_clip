"""
视频切片四步工作流 - Playwright 自动化测试
"""
import asyncio
from playwright.async_api import async_playwright
import time

async def test_video_slice_workflow():
    """测试视频切片完整工作流程"""

    video_path = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (1).mp4"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        try:
            # ========== 步骤 0: 打开应用 ==========
            print("✅ 步骤 0: 打开应用...")
            await page.goto("http://localhost:8506")
            await page.wait_for_load_state("networkidle")
            await page.screenshot(path="output/playwright/test_01_home.png")

            # ========== 步骤 1: 点击【视频切片】Tab ==========
            print("✅ 步骤 1: 点击【视频切片】Tab...")
            await page.click("text=🎬 视频切片")
            await page.wait_for_timeout(1500)
            await page.screenshot(path="output/playwright/test_02_video_slice_tab.png")

            # ========== 步骤 2: 输入视频路径 ==========
            print(f"✅ 步骤 2: 输入视频路径...")
            text_input = page.locator("input[key='slice_step0_local_path']")
            await text_input.fill(video_path)
            await page.screenshot(path="output/playwright/test_03_video_path_input.png")

            # ========== 步骤 3: 点击"使用本地文件"按钮 ==========
            print("✅ 步骤 3: 点击'使用本地文件'按钮...")
            await page.click("text=使用本地文件")
            await page.wait_for_timeout(1000)
            await page.screenshot(path="output/playwright/test_04_use_local.png")

            # ========== 步骤 4: 点击"下一步：开始处理" ==========
            print("✅ 步骤 4: 点击'下一步：开始处理'...")
            await page.click("text=下一步：开始处理")
            await page.wait_for_timeout(1000)
            await page.screenshot(path="output/playwright/test_05_step1_page.png")

            # ========== 步骤 5: 点击"开始处理" ==========
            print("✅ 步骤 5: 点击'开始处理'...")
            await page.click("text=🚀 开始处理")
            await page.wait_for_timeout(1000)
            await page.screenshot(path="output/playwright/test_06_processing_started.png")

            # ========== 步骤 6: 等待处理完成 ==========
            print("⏳ 步骤 6: 等待处理完成（这可能需要几分钟）...")
            max_wait = 600  # 最多等待 10 分钟
            start_time = time.time()
            processing_done = False

            while time.time() - start_time < max_wait:
                try:
                    success_msg = page.locator("text=✅ 处理完成！")
                    if await success_msg.is_visible(timeout=2000):
                        print("✅ 处理完成！")
                        await page.screenshot(path="output/playwright/test_07_processing_done.png")
                        processing_done = True
                        break
                except:
                    pass

                try:
                    error_msg = page.locator("text=❌ 处理失败")
                    if await error_msg.is_visible(timeout=2000):
                        print("❌ 处理失败！")
                        await page.screenshot(path="output/playwright/test_07_processing_error.png")
                        processing_done = False
                        break
                except:
                    pass

                elapsed = int(time.time() - start_time)
                print(f"⏳ 仍在处理中... ({elapsed}秒)")
                await page.wait_for_timeout(10)

            if not processing_done:
                print("❌ 处理超时！")
                await page.screenshot(path="output/playwright/test_07_processing_timeout.png")
                return

            # ========== 步骤 7: 点击"下一步：预览编辑" ==========
            print("✅ 步骤 7: 点击'下一步：预览编辑'...")
            await page.click("text=下一步：预览编辑")
            await page.wait_for_timeout(1000)
            await page.screenshot(path="output/playwright/test_08_preview_page.png")

            # ========== 步骤 8: 测试时间轴交互（点击第一个片段）==========
            print("✅ 步骤 8: 测试时间轴交互...")
            segment_buttons = page.locator("button:has-text('主歌'), button:has-text('副歌'), button:has-text('前奏'), button:has-text('尾奏'), button:has-text('间奏')")
            if await segment_buttons.count() > 0:
                await segment_buttons.first.click()
                await page.wait_for_timeout(2000)
                print("✅ 已点击时间轴片段")
                await page.screenshot(path="output/playwright/test_09_timeline_click.png")
            else:
                print("⚠️ 未找到片段按钮，跳过时间轴测试")

            # ========== 步骤 9: 返回重新选择视频 ==========
            print("✅ 步骤 9: 点击'重新开始'...")
            restart_btn = page.locator("text=🔄 重新开始")
            if await restart_btn.is_visible():
                await restart_btn.click()
                await page.wait_for_timeout(1000)
                await page.screenshot(path="output/playwright/test_10_restart.png")
                print("✅ 已重新开始")

            # ========== 最终截图 ==========
            print("✅ 截取最终页面...")
            await page.screenshot(path="output/playwright/test_final.png")

            print("\n🎉 测试完成！所有截图已保存到 output/playwright/ 目录")

        except Exception as e:
            print(f"\n❌ 测试过程中出现错误: {e}")
            await page.screenshot(path="output/playwright/test_error.png")
            raise

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_video_slice_workflow())

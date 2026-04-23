"""详细调试脚本 - 检查处理完成后的页面内容"""
import asyncio
from playwright.async_api import async_playwright
import time

async def debug_processing():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        try:
            print("✅ 打开应用...")
            await page.goto("http://localhost:8505")
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2000)
            await page.screenshot(path="output/playwright/debug_01_home.png")
            
            print("✅ 点击【视频切片】Tab...")
            await page.click("text=🎬 视频切片")
            await page.wait_for_timeout(2000)
            await page.screenshot(path="output/playwright/debug_02_video_slice_tab.png")
            
            print("✅ 输入视频路径...")
            all_text_inputs = page.locator("div[data-testid='stTextInput'] input")
            await all_text_inputs.first.wait_for()
            count = await all_text_inputs.count()
            print(f"✅ 找到 {count} 个文本输入框")
            
            video_path = r"D:\个人资料\音乐测试\视频素材\live\周深\周深-VX-q97643800 (1).mp4"
            
            await all_text_inputs.nth(2).click()
            await all_text_inputs.nth(2).fill(video_path)
            await page.screenshot(path="output/playwright/debug_03_video_path_input.png")
            
            print("✅ 点击'使用本地文件'按钮...")
            all_use_local_buttons = page.locator("button", has_text="使用本地文件")
            await all_use_local_buttons.first.click()
            await page.wait_for_timeout(3000)
            await page.screenshot(path="output/playwright/debug_04_use_local.png")
            
            print("✅ 点击'下一步：开始处理'...")
            await page.wait_for_timeout(2000)
            all_next_buttons = page.locator("button", has_text="下一步：开始处理")
            await all_next_buttons.first.scroll_into_view_if_needed()
            await page.wait_for_timeout(1000)
            await all_next_buttons.first.click()
            await page.wait_for_timeout(1000)
            await page.screenshot(path="output/playwright/debug_05_next_to_processing.png")
            
            print("✅ 点击'开始处理'...")
            all_start_buttons = page.locator("button", has_text="🚀 开始处理")
            await all_start_buttons.first.click()
            await page.wait_for_timeout(1000)
            await page.screenshot(path="output/playwright/debug_06_start_processing.png")
            
            print("⏳ 等待处理完成（最多 300 秒）...")
            max_wait = 300
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0:
                    print(f"⏳ 仍在处理中... ({elapsed}s)")
                    try:
                        await page.screenshot(path=f"output/playwright/debug_processing_{elapsed}s.png")
                    except Exception as e:
                        print(f"⚠️ 截图失败: {e}")
                
                # 检查是否有任何"下一步"按钮出现
                try:
                    all_next_buttons = page.locator("button", has_text="下一步")
                    if await all_next_buttons.count() > 0:
                        print(f"🎉 发现 {await all_next_buttons.count()} 个'下一步'按钮！")
                        await page.screenshot(path="output/playwright/debug_processing_done_before_click.png")
                        break
                except Exception as e:
                    print(f"⚠️ 检查按钮失败: {e}")
                
                await page.wait_for_timeout(2)
            
            # 最后截图
            print("✅ 截取最终页面...")
            await page.screenshot(path="output/playwright/debug_final_before_click.png")
            
            # 打印页面完整内容，方便调试
            page_content = await page.content()
            print("\n✅ 页面内容前 2000 字符:")
            print(page_content[:2000])
            
            # 查找所有按钮文本
            all_buttons = page.locator("button")
            button_count = await all_buttons.count()
            print(f"\n✅ 页面上有 {button_count} 个按钮，它们的文本是:")
            for i in range(button_count):
                try:
                    text = await all_buttons.nth(i).text_content()
                    print(f"  按钮 {i}: {repr(text)}")
                except Exception as e:
                    print(f"  按钮 {i}: 无法读取文本 - {e}")
            
            print("\n✅ 调试脚本完成！所有截图已保存到 output/playwright/")
            
            # 等待 10 秒，让你看看页面
            await page.wait_for_timeout(10000)
            
        except Exception as e:
            print(f"\n❌ 调试脚本出错: {e}")
            await page.screenshot(path="output/playwright/debug_error.png")
            raise
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(debug_processing())

"""
简单测试 - 分析输入框布局
"""
import asyncio
from playwright.async_api import async_playwright

async def test_inputs():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await page.goto("http://localhost:8505")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(2000)
        
        # 点击视频切片 Tab
        await page.click("text=🎬 视频切片")
        await page.wait_for_timeout(2000)
        
        await page.screenshot(path="output/playwright/debug_tab.png")
        
        # 列出所有文本输入框的位置
        all_text_inputs = page.locator("div[data-testid='stTextInput']")
        count = await all_text_inputs.count()
        print(f"✅ 找到 {count} 个 stTextInput")
        
        for i in range(count):
            try:
                txt_input = all_text_inputs.nth(i)
                # 截图这个输入框
                await txt_input.scroll_into_view_if_needed()
                await page.wait_for_timeout(500)
                await txt_input.screenshot(path=f"output/playwright/debug_input_{i}.png")
                
                # 查找附近的标签
                parent = txt_input.locator("xpath=ancestor::div[contains(@class, 'stTextInput')]")
                nearby_text = await page.locator("div", has=txt_input).text_content()
                print(f"输入框 #{i} 附近文本: {nearby_text}")
            except Exception as e:
                print(f"输入框 #{i} 错误: {e}")
        
        # 填写所有的输入框，看哪个对
        for i in range(count):
            try:
                txt_input = all_text_inputs.nth(i).locator("input")
                await txt_input.scroll_into_view_if_needed()
                await txt_input.click()
                await txt_input.fill(f"TEST_INPUT_{i}")
            except Exception as e:
                print(f"填输入框 #{i} 错误: {e}")
        
        await page.screenshot(path="output/playwright/debug_filled.png")
        
        # 最后关闭
        await page.wait_for_timeout(5000)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_inputs())
"""
简单测试脚本 - 打开页面并截图
"""
import asyncio
from playwright.async_api import async_playwright

async def test_simple():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        await page.goto("http://localhost:8505")
        await page.wait_for_load_state("networkidle")
        await page.screenshot(path="output/playwright/current_page.png")
        
        print("✅ 截图已保存到 output/playwright/current_page.png")
        
        # 等待查看
        await page.wait_for_timeout(5000)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_simple())
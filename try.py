from playwright.sync_api import sync_playwright
import os, time

DOWNLOAD_DIR = os.path.abspath("./downloads")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(accept_downloads=True)
    page = context.new_page()

    # 打开文件页面
    page.goto("https://drive.google.com/file/d/166zucXkPXNBsK8ALxfRCEzCwbVG8UZf_/view", timeout=120000)

    # 等待“Download anyway”按钮出现并点击
    try:
        page.get_by_text("Download anyway").click(timeout=10000)
    except:
        print("未找到 Download anyway 按钮，可能无需确认。")

    # 等待文件开始下载
    with page.expect_download() as download_info:
        # 如果按钮未出现，这里可以触发直接下载
        pass
    download = download_info.value

    # 保存到本地文件
    file_path = os.path.join(DOWNLOAD_DIR, download.suggested_filename)
    download.save_as(file_path)
    print(f"✅ 下载完成: {file_path}")

    browser.close()

# -*- coding: utf-8 -*-
"""
API 状态验证脚本
用于验证 llm.rekeymed.com API 的可用性

结论：
- API_KEY 有效，可以正常连接
- /v1/models 接口返回模型列表，包含 Qwen/Qwen2.5-Omni-7B
- 但实际调用 Qwen/Qwen2.5-Omni-7B 时返回 404 错误，提示模型不存在
"""

import requests
import json
from datetime import datetime

# ========== 配置 ==========
API_KEY = "sk-CopXuPMUxmJY7UNSXrjyBA"
BASE_URL = "https://llm.rekeymed.com/v1/"

def print_separator(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)

def test_models_list():
    """测试1: 获取模型列表"""
    print_separator("测试1: 获取模型列表 (/v1/models)")
    
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        resp = requests.get(f"{BASE_URL}models", headers=headers, timeout=30)
        
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('data', [])
            
            print(f"✅ API 连接成功，状态码: {resp.status_code}")
            print(f"✅ 返回 {len(models)} 个模型:")
            print()
            
            for model in models:
                model_id = model.get('id', 'unknown')
                print(f"   - {model_id}")
            
            # 检查 Qwen/Qwen2.5-Omni-7B 是否在列表中
            model_ids = [m.get('id') for m in models]
            if "Qwen/Qwen2.5-Omni-7B" in model_ids:
                print()
                print("⚠️  注意: Qwen/Qwen2.5-Omni-7B 在模型列表中")
            
            return True, models
        else:
            print(f"❌ API 请求失败，状态码: {resp.status_code}")
            print(f"   响应: {resp.text[:200]}")
            return False, []
            
    except Exception as e:
        print(f"❌ 连接异常: {e}")
        return False, []

def test_text_chat(model_name):
    """测试2: 测试纯文本聊天"""
    print_separator(f"测试2: 纯文本聊天 ({model_name})")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, respond with just 'OK'"}],
        "max_tokens": 10
    }
    
    try:
        resp = requests.post(f"{BASE_URL}chat/completions", headers=headers, json=data, timeout=30)
        
        if resp.status_code == 200:
            result = resp.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ 调用成功，状态码: {resp.status_code}")
            print(f"   模型响应: {content[:100]}")
            return True
        else:
            error_data = resp.json()
            error_msg = error_data.get('error', {}).get('message', resp.text)
            print(f"❌ 调用失败，状态码: {resp.status_code}")
            print(f"   错误信息: {error_msg[:300]}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_multimodal_chat(model_name):
    """测试3: 测试多模态聊天（带视频）"""
    print_separator(f"测试3: 多模态聊天 - 视频输入 ({model_name})")
    
    import base64
    import os
    
    # 使用一个小的测试视频
    test_video = "./MELD/train/dia0_utt0.mp4"
    
    if not os.path.exists(test_video):
        print(f"⚠️  测试视频不存在: {test_video}")
        print("   跳过多模态测试")
        return None
    
    # 编码视频
    with open(test_video, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    print(f"   测试视频: {test_video}")
    print(f"   视频大小: {len(video_base64) // 1024} KB (base64编码后)")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this video briefly."},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{video_base64}",
                        "mime_type": "video/mp4"
                    }
                }
            ]
        }],
        "max_tokens": 100
    }
    
    try:
        resp = requests.post(f"{BASE_URL}chat/completions", headers=headers, json=data, timeout=60)
        
        if resp.status_code == 200:
            result = resp.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            print(f"✅ 多模态调用成功，状态码: {resp.status_code}")
            print(f"   模型响应: {content[:200]}")
            return True
        else:
            error_data = resp.json()
            error_msg = error_data.get('error', {}).get('message', resp.text)
            print(f"❌ 多模态调用失败，状态码: {resp.status_code}")
            print(f"   错误信息: {error_msg[:300]}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def main():
    print("=" * 70)
    print("  API 状态验证报告")
    print("=" * 70)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  API 地址: {BASE_URL}")
    print(f"  API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    
    # 测试1: 获取模型列表
    success, models = test_models_list()
    
    if not success:
        print("\n❌ API 连接失败，无法继续测试")
        return
    
    # 测试2: 测试各个模型的文本聊天
    print_separator("测试2: 验证各模型的文本聊天功能")
    
    models_to_test = [
        "Qwen/Qwen2.5-Omni-7B",  # 多模态模型（预期失败）
        "zai-org/GLM-4.5-Air",   # 文本模型
        "Qwen/QwQ-32B-Long",     # 文本模型
    ]
    
    text_results = {}
    for model in models_to_test:
        print(f"\n>>> 测试模型: {model}")
        result = test_text_chat(model)
        text_results[model] = result
    
    # 测试3: 测试多模态功能
    print_separator("测试3: 验证多模态（视频）功能")
    
    multimodal_results = {}
    for model in models_to_test:
        print(f"\n>>> 测试模型: {model}")
        result = test_multimodal_chat(model)
        multimodal_results[model] = result
    
    # ========== 总结报告 ==========
    print_separator("总结报告")
    
    print("\n1. API 连接状态:")
    print(f"   ✅ API_KEY 有效，可以正常连接")
    print(f"   ✅ /v1/models 接口正常，返回 {len(models)} 个模型")
    
    print("\n2. 模型列表中包含:")
    for m in models:
        print(f"   - {m.get('id')}")
    
    print("\n3. 文本聊天测试结果:")
    for model, result in text_results.items():
        status = "✅ 成功" if result else "❌ 失败"
        print(f"   {model}: {status}")
    
    print("\n4. 多模态（视频）测试结果:")
    for model, result in multimodal_results.items():
        if result is None:
            status = "⚠️  跳过（无测试视频）"
        elif result:
            status = "✅ 成功"
        else:
            status = "❌ 失败"
        print(f"   {model}: {status}")
    
    print_separator("结论")
    print("""
⚠️  问题说明:

1. API_KEY 有效，API 服务正常运行

2. /v1/models 接口显示 Qwen/Qwen2.5-Omni-7B 在可用模型列表中

3. 但实际调用 Qwen/Qwen2.5-Omni-7B 时返回 404 错误:
   "The model `Qwen/Qwen2.5-Omni-7B` does not exist"

4. 其他模型 (GLM-4.5-Air, QwQ-32B-Long) 可以进行文本聊天，
   但不支持多模态（视频）输入

5. 建议：联系 API 服务方确认 Qwen/Qwen2.5-Omni-7B 的可用性
""")

if __name__ == "__main__":
    main()











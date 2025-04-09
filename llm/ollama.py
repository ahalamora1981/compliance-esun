import requests
import json
import time
import tomllib
from pathlib import Path


# 读取配置
with open(Path(__file__).parent.parent / "config.toml", "rb") as f:
    config = tomllib.load(f)


OLLAMA_HOST = config["ollama"]["host"]
OLLAMA_PORT = config["ollama"]["port"]
MAX_TOKENS = config["ollama"]["max_tokens"]

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"函数 {func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper

def ollama_chat(
    prompt: str, 
    system_prompt: str = "", 
    history: list | None = None,
    temperature: float = 0.0
):
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/chat"  # Adjust the URL if needed
    
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "history": history or [],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()["content"]

def ollama_chat_stream(prompt, system_prompt="", history=None, temperature=0.0):
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/chat-stream"  # Adjust the URL if needed
    
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "history": history or [],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature
    }
    
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            yield chunk["content"]
            

# Example usage
if __name__ == "__main__":
    system_prompt = "你是一个AI助手"
    user_prompt = "写一篇200字的高中作文，关于信念。"
    
    print("Testing streaming chat:")
    full_response = ""

    start = time.time()

    for content in ollama_chat_stream(user_prompt, system_prompt):
        print(content, end='', flush=True)
        full_response += content
        
    duration = time.time() - start
    print(f"Duration: {duration:.2f}")

    # print("\n\nTesting non-streaming chat:")
    # non_stream_response = ollama_chat(user_prompt, system_prompt)
    # print("Non-streaming response:", non_stream_response)


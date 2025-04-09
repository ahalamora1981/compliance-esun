# server.py
import asyncio
from asyncio import Semaphore
import json
import re
import time
from pathlib import Path
# from tqdm import tqdm

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from transformers import AutoTokenizer
from llm.vllm import VllmClient

import tomllib


CHECK_OUTPUT_JSON_FORMAT = json.dumps(
    {
        "检查结论": "(直接陈述您的结论，确保简洁明了)",
        "风险等级": "(提供问题相关的风险等级，使用清晰的评级，如“无风险”、“低风险”、“中风险”、“高风险”)",
        "风险依据": "(解释支持您结论的风险依据，包括数据、研究或逻辑推理)",
        "优化建议": "(提供优化建议，包括更改方案、降低风险等等)"
    },
    ensure_ascii=False
)

# 加载配置
with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)

if config["check-scenario"]["concurrent_limit"] not in [1, 2, 3, 4]:
    raise ValueError("Concurrent limit must be 1, 2, 3, or 4") 

HOST = config["server"]["host"]
PORT = config["server"]["port"]

VLLM_HOST_QWEN = config["vllm-qwen25-72b"]["host"]
VLLM_PORT_QWEN = config["vllm-qwen25-72b"]["port"]
MAX_CONTEXT_LENGTH_QWEN = int(config["vllm-qwen25-72b"]["max_context_length"] * 0.95)
MAX_TOKENS_QWEN = config["vllm-qwen25-72b"]["max_tokens"]
TEMPERATURE_QWEN = config["vllm-qwen25-72b"]["temperature"]
MAX_INPUT_LENGTH_QWEN = MAX_CONTEXT_LENGTH_QWEN - MAX_TOKENS_QWEN

VLLM_HOST_DEEPSEEK = config["vllm-deepseek-r1-32b"]["host"]
VLLM_PORT_DEEPSEEK = config["vllm-deepseek-r1-32b"]["port"]
MAX_CONTEXT_LENGTH_DEEPSEEK = int(config["vllm-deepseek-r1-32b"]["max_context_length"] * 0.95)
MAX_TOKENS_DEEPSEEK = config["vllm-deepseek-r1-32b"]["max_tokens"]
TEMPERATURE_DEEPSEEK = config["vllm-deepseek-r1-32b"]["temperature"]
MAX_INPUT_LENGTH_DEEPSEEK = MAX_CONTEXT_LENGTH_DEEPSEEK - MAX_TOKENS_DEEPSEEK

vllm_qwen = VllmClient(
    VLLM_HOST_QWEN, 
    VLLM_PORT_QWEN, 
    MAX_TOKENS_QWEN,
    TEMPERATURE_QWEN
)

vllm_ds = VllmClient(
    VLLM_HOST_DEEPSEEK, 
    VLLM_PORT_DEEPSEEK, 
    MAX_TOKENS_DEEPSEEK,
    TEMPERATURE_DEEPSEEK
)

# 创建日志目录
LOG_DIR = "./logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# 配置loguru
logger.add(
    f"{LOG_DIR}/service.log",
    rotation="3 MB",       # 每3MB分割新文件
    retention=10,          # 保留最近10个文件
    compression="zip",     # 旧日志压缩保存
    enqueue=True,          # 线程安全
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    backtrace=True,        # 记录异常堆栈
    diagnose=True,         # 显示变量值
    level="INFO"
)

tokenizer_path = Path.cwd() / "model" / "qwen25-72b-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) 

app = FastAPI()

def parse_json(text):
    # 使用正则表达式提取JSON字符串
    json_match = re.search(r'```json(.*?)```', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            # 解析JSON字符串
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
    else:
        print("未找到JSON字符串")
        return None

def count_tokens(prompt: str) -> int:
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    return len(tokens)

class Request(BaseModel):
    prompt: str
    system_prompt: str
    temperature: float = 0.0

class Response(BaseModel):
    result: str

@app.post("/correction", response_model=Response)
async def correction(request: Request):
    try:
        prompt = request.prompt
        system_prompt = request.system_prompt
        tokens_count_system_prompt = count_tokens(system_prompt)
        temperature = request.temperature
        
        # 输入和输出的长度都是MAX_CONTEXT_LENGTH_QWEN的一半
        token_limit = MAX_CONTEXT_LENGTH_QWEN // 2
        prompt_list = prompt.split("\n")
        tokens_count_total = tokens_count_system_prompt
        prompt_total = ""
        result_total = ""
        
        for prompt in prompt_list:
            tokens_count_prompt = count_tokens(prompt)
            
            if tokens_count_total + tokens_count_prompt > token_limit:
                logger.info(f"Surpass token limit: {token_limit}")
                result = vllm_qwen.chat(
                    prompt=prompt_total, 
                    system_prompt=system_prompt
                )
                result_total += result + "\n"
                prompt_total = prompt + "\n"
                tokens_count_total = tokens_count_system_prompt + tokens_count_prompt
            else:
                prompt_total += prompt + "\n"
                tokens_count_total += tokens_count_prompt
        
        result = vllm_qwen.chat(
            prompt=prompt_total, 
            system_prompt=system_prompt,
            max_tokens=token_limit,
            temperature=temperature
        )
        result_total += result

        return Response(result=result_total.strip())
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary", response_model=Response)
async def summary(request: Request):
    try:
        prompt = request.prompt
        system_prompt = request.system_prompt
        tokens_count_system_prompt = count_tokens(system_prompt)
        temperature = request.temperature
        
        # 输入的最大长度是MAX_INPUT_LENGTH_QWEN
        token_limit = MAX_INPUT_LENGTH_QWEN
        prompt_list = prompt.split("\n")
        tokens_count_total = tokens_count_system_prompt
        prompt_accumulate = ""
        result = ""
        
        for prompt in prompt_list:
            tokens_count_prompt = count_tokens(prompt)
            
            if tokens_count_total + tokens_count_prompt > token_limit:
                logger.info(f"Surpass token limit: {token_limit}")
                prompt_total = f"[现有摘要]: \n{result}\n\n[额外通话内容]: \n{prompt_accumulate}"
                result = vllm_qwen.chat(
                    prompt=prompt_total, 
                    system_prompt=system_prompt,
                    temperature=temperature
                )
                prompt_accumulate = prompt + "\n"
                tokens_count_total = tokens_count_system_prompt + tokens_count_prompt
            else:
                prompt_accumulate += prompt + "\n"
                tokens_count_total += tokens_count_prompt
        
        prompt_total = f"[现有摘要]: \n{result}\n\n[额外内容]: \n{prompt_accumulate}"
        result = vllm_qwen.chat(
            prompt=prompt_total, 
            system_prompt=system_prompt,
            max_tokens=MAX_TOKENS_QWEN,
            temperature=temperature
        )
        return Response(result=result.strip())
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


class Scenario(BaseModel):  
    scenario_name: str
    scenario_description: str | None = None
    
class KeyPoint(BaseModel):
    key_point_name: str
    key_point_description: str | None = None
    scenarios: list[Scenario] | None = None

class Task(BaseModel):
    task_name: str
    task_description: str | None = None
    key_points: list[KeyPoint]

class CheckTaskRequest(BaseModel):
    task: Task
    content: str
    content_type: str
    system_prompt_scenario: str
    system_prompt_task: str
    
class CheckTaskResponse(BaseModel):
    检查结论: str
    风险等级: str
    风险依据: str
    优化建议: str
    场景检查结果: list[str]

@app.post("/check-task", response_model=CheckTaskResponse)
async def check_task(request: CheckTaskRequest):
    llm_for_scenario = config["check-scenario"]["llm_for_scenario"]
    
    if llm_for_scenario == "qwen":
        max_input_length = MAX_CONTEXT_LENGTH_QWEN
        max_tokens = MAX_TOKENS_QWEN
        temperature = TEMPERATURE_QWEN
        vllm_scenario = vllm_qwen
    elif llm_for_scenario == "deepseek":
        max_input_length = MAX_CONTEXT_LENGTH_DEEPSEEK
        max_tokens = MAX_TOKENS_DEEPSEEK
        temperature = TEMPERATURE_DEEPSEEK
        vllm_scenario = vllm_ds
    else:
        raise HTTPException(status_code=400, detail="Invalid LLM for scenario")
    
    system_prompt_list = []
    for i in range(len(request.task.key_points)):
        for j in range(len(request.task.key_points[i].scenarios)):
            try:
                system_prompt_scenario = request.system_prompt_scenario.format(
                    content_type = request.content_type,
                    key_point_name = request.task.key_points[i].key_point_name,
                    key_point_description = request.task.key_points[i].key_point_description,
                    scenario_name = request.task.key_points[i].scenarios[j].scenario_name,
                    scenario_description = request.task.key_points[i].scenarios[j].scenario_description
                )
            except Exception as e:
                logger.error(e)
                raise HTTPException(status_code=500, detail=str(e))
            
            if count_tokens(system_prompt_scenario + request.content) > max_input_length:
                error_scenario = request.task.key_points[i].scenarios[j].scenario_name
                error_msg = f"Scenario {error_scenario} exceeds token limit {max_input_length}"
                raise HTTPException(status_code=400, detail=error_msg)
            
            system_prompt_list.append(system_prompt_scenario)

    ##################################
    ### Syncronously call DeepSeek ###
    ##################################
    # check_result_scenarios = []
    # for system_prompt in tqdm(system_prompt_list):
    #     result = vllm_scenario.chat(
    #         prompt=request.content, 
    #         system_prompt=system_prompt,
    #         max_tokens=max_tokens,
    #         temperature=temperature
    #     )
        
    #     # 如果是R1模型，则去掉思考的部分
    #     if "</think>" in result:
    #         result = result.split("</think>")[-1]
            
    #     check_result_scenarios.append(result.strip())
    
    # check_result_scenarios_str = "\n\n--\n\n".join(check_result_scenarios)
    
    ###################################
    ### Asyncronously call DeepSeek ###
    ###################################
    concurrency_limit = config["check-scenario"]["concurrent_limit"]
    semaphore = Semaphore(concurrency_limit)
    
    # Create a list of coroutines to run concurrently
    async def process_prompt(system_prompt):
        try:
            async with semaphore:
                result = await vllm_scenario.async_chat(
                    prompt=request.content, 
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # 如果是R1模型，则去掉思考的部分
                if "</think>" in result:
                    result = result.split("</think>")[-1]
                    
                return result.strip()
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
    
    try:
        # Run all prompts concurrently and collect results
        start_time = time.time()
        tasks = [process_prompt(prompt) for prompt in system_prompt_list]
        check_result_scenarios = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        print(f"Async execution time: {elapsed_time:.2f} seconds")
        
        check_result_scenarios_str = "\n\n--\n\n".join(check_result_scenarios)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    ##################
    ### Check task ###
    ##################
    system_prompt_task = request.system_prompt_task.format(
        content_type = request.content_type,
        task_name = request.task.task_name,
        task_description = request.task.task_description,
        check_result_scenarios = check_result_scenarios_str,
        output_json_format = "```json\n" + CHECK_OUTPUT_JSON_FORMAT + "\n```"
    )
    
    if count_tokens(system_prompt_task + request.content) > MAX_INPUT_LENGTH_QWEN:
        raise HTTPException(status_code=400, detail="System prompt and content exceeds token limit")
    
    try:
        result = vllm_qwen.chat(
            prompt=request.content, 
            system_prompt=system_prompt_task,
            max_tokens=MAX_TOKENS_QWEN,
            temperature=TEMPERATURE_QWEN
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
        
    try:
        result_json = parse_json(result)
    except Exception as e:
        logger.info("Failed to parse JSON")
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    result_json['场景检查结果'] = check_result_scenarios
    
    return CheckTaskResponse.model_validate(
        result_json
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app", 
        host=config["server"]["host"], 
        port=config["server"]["port"], 
        reload=True,
        timeout_keep_alive=config["server"]["timeout_keep_alive"]
    )
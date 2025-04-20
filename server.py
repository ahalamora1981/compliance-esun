# server.py
import asyncio
from asyncio import Semaphore
import json
import re
import time
from pathlib import Path
# from tqdm import tqdm
from textwrap import dedent
import requests

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from transformers import AutoTokenizer
from llm.vllm import VllmClient

import tomllib


# 加载配置
with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)
    
with open(Path(__file__).parent / "prompts" / "check" / "system_prompt_task.txt", "r") as f:
    SYSTEM_PROMPT_TASK = f.read()
    
with open(Path(__file__).parent / "prompts" / "check" / "system_prompt_scenario.txt", "r") as f:
    SYSTEM_PROMPT_SCENARIO = f.read()
    
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

VLLM_HOST_QWEN_VL = config["vllm-qwen25-vl-32b"]["host"]
VLLM_PORT_QWEN_VL = config["vllm-qwen25-vl-32b"]["port"]
MAX_CONTEXT_LENGTH_QWEN_VL = int(config["vllm-qwen25-vl-32b"]["max_context_length"] * 0.95)
MAX_TOKENS_QWEN_VL = config["vllm-qwen25-vl-32b"]["max_tokens"]
TEMPERATURE_QWEN_VL = config["vllm-qwen25-vl-32b"]["temperature"]
MAX_INPUT_LENGTH_QWEN_VL = MAX_CONTEXT_LENGTH_QWEN_VL - MAX_TOKENS_QWEN_VL

VLLM_HOST_DEEPSEEK = config["vllm-deepseek-r1-32b"]["host"]
VLLM_PORT_DEEPSEEK = config["vllm-deepseek-r1-32b"]["port"]
MAX_CONTEXT_LENGTH_DEEPSEEK = int(config["vllm-deepseek-r1-32b"]["max_context_length"] * 0.95)
MAX_TOKENS_DEEPSEEK = config["vllm-deepseek-r1-32b"]["max_tokens"]
TEMPERATURE_DEEPSEEK = config["vllm-deepseek-r1-32b"]["temperature"]
MAX_INPUT_LENGTH_DEEPSEEK = MAX_CONTEXT_LENGTH_DEEPSEEK - MAX_TOKENS_DEEPSEEK

VECTOR_DB_HOST = config["vector-db"]["host"]
VECTOR_DB_PORT = config["vector-db"]["port"]
VECTOR_DB_COLLECTION_NAME = config["vector-db"]["collection_name"]

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

def parse_json(text: str) -> dict | None:
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
    scenario_reference: str | None = None
    
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
    media_type: str
    content: str
    content_type: str
    content_metadata: str
    
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
                system_prompt_scenario = SYSTEM_PROMPT_SCENARIO.format(
                    content_type = request.content_type,
                    content_metadata = request.content_metadata,
                    key_point_name = request.task.key_points[i].key_point_name,
                    key_point_description = request.task.key_points[i].key_point_description,
                    scenario_name = request.task.key_points[i].scenarios[j].scenario_name,
                    scenario_description = request.task.key_points[i].scenarios[j].scenario_description
                )
                
                if request.task.key_points[i].scenarios[j].scenario_reference:
                    system_prompt_scenario += f"\n\n## 参考资料\n{request.task.key_points[i].scenarios[j].scenario_reference}"
            except Exception as e:
                logger.error(e)
                raise HTTPException(status_code=500, detail=str(e))
            
            if request.media_type == "text":
                if count_tokens(system_prompt_scenario + request.content) > max_input_length:
                    error_scenario = request.task.key_points[i].scenarios[j].scenario_name
                    error_msg = f"Scenario {error_scenario} exceeds token limit {max_input_length}"
                    raise HTTPException(status_code=400, detail=error_msg)
            elif request.media_type == "image":
                if len(request.content) > 10_000_000:
                    raise HTTPException(status_code=400, detail="Image exceeds base64 size limit: 10MB")
            else:
                raise HTTPException(status_code=400, detail="Invalid media type")
            
            system_prompt_list.append(system_prompt_scenario)
            
    ###################################
    ### Syncronously Scenario Check ###
    ###################################
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
    
    ####################################
    ### Asyncronously Scenario Check ###
    ####################################
    concurrency_limit = config["check-scenario"]["concurrent_limit"]
    semaphore = Semaphore(concurrency_limit)
    
    # Create a list of coroutines to run concurrently
    async def check_scenario(system_prompt):
        try:
            async with semaphore:
                if request.media_type == "text":
                    result = await vllm_scenario.async_chat(
                        prompt=request.content or "执行系统指令",
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                elif request.media_type == "image":
                    result = await ask_image(
                        prompt="执行系统指令",
                        image_base64=request.content,
                        system_prompt=system_prompt
                    )
                else:
                    raise HTTPException(status_code=400, detail="Invalid media type")
                
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
        tasks = [check_scenario(prompt) for prompt in system_prompt_list]
        check_result_scenarios = await asyncio.gather(*tasks)
        elapsed_time = time.time() - start_time
        print(f"Async execution time: {elapsed_time:.2f} seconds")
        
        check_result_scenarios_str = "\n\n--\n\n".join(check_result_scenarios)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    print(check_result_scenarios_str)
    
    ##################
    ### Check task ###
    ##################
    output_json_format = "```json\n" + json.dumps(
        {
            "检查结论": "直接陈述您的结论，确保简洁明了(type: str)",
            "风险等级": "提供问题相关的风险等级，使用清晰的评级，如“无风险”、“低风险”、“中风险”、“高风险”(type: str)",
            "风险依据": "解释支持您结论的风险依据，包括数据、研究或逻辑推理(type: str)",
            "优化建议": "提供优化建议，包括更改方案、降低风险等等(type: str)"
        },
        ensure_ascii=False
    ) + "\n```"
    
    system_prompt_task = SYSTEM_PROMPT_TASK.format(
        content_type = request.content_type,
        content_metadata = request.content_metadata,
        task_name = request.task.task_name,
        task_description = request.task.task_description,
        output_json_format = output_json_format,
        check_result_scenarios = check_result_scenarios_str
    )
    
    if request.media_type == "text":
        if count_tokens(system_prompt_task + request.content) > MAX_INPUT_LENGTH_QWEN:
            raise HTTPException(status_code=400, detail="System prompt and content exceeds token limit")
    elif request.media_type == "image":
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid media type")
    
    try:
        if request.media_type == "text":
            result = vllm_qwen.chat(
                prompt=request.content or "执行系统指令", 
                system_prompt=system_prompt_task,
                max_tokens=MAX_TOKENS_QWEN,
                temperature=TEMPERATURE_QWEN
            )
        elif request.media_type == "image":
            result = vllm_qwen.chat(
                prompt="执行系统指令", 
                system_prompt=system_prompt_task,
                max_tokens=MAX_TOKENS_QWEN,
                temperature=TEMPERATURE_QWEN
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid media type")
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

async def ask_image(prompt: str, image_base64: str, system_prompt: str | None = None, history: list | None =None) -> str:
    url = f"http://{VLLM_HOST_QWEN_VL}:{VLLM_PORT_QWEN_VL}/ask-image"
    
    payload = {
        "prompt": prompt,
        "image_base64": image_base64,
        "system_prompt": system_prompt or "",
        "history": history or [],
        "max_tokens": MAX_TOKENS_QWEN_VL,
        "temperature": TEMPERATURE_QWEN_VL
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()["content"]

def query_vector_db(
    collection_name: str,
    query: str,
    n_results: int = 10,
    rerank: bool = True,
    metadata_filter: dict = {}
) -> dict:
    end_point = "query"
    
    url = f"http://{VECTOR_DB_HOST}:{VECTOR_DB_PORT}/{end_point}"
    
    payload = json.dumps({
        "collection_name": collection_name,
        "query": query,
        "n_results": n_results,
        "rerank": rerank,
        "metadata_filter": metadata_filter
    })
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()

    return response.json()

class GetFundNamesRequest(BaseModel):
    image_base64: str


class FundNameMatch(BaseModel):
    fund_name_raw: str
    fund_name_real: str
    is_same_fund: bool
    fund_id: str

class GetFundNamesResponse(BaseModel):
    fund_names: list[FundNameMatch]


@app.post("/get-fund-names", response_model=GetFundNamesResponse)
async def get_fund_names(request: GetFundNamesRequest):
    # 从图片中提取基金名称
    system_prompt = dedent("""
    # 任务：输出基金名称
    
    ## 指令
    - 从图片中提取所有基金名称
    - 注意不要遗漏任何一个基金名称
    - 以列表的形式列出基金名称，并输出到json格式
    
    ## 输出格式
    ```json
    {
        "fund_names": ["基金名称1", "基金名称2", ...]
    }
    ```
    """)
    prompt = "执行任务：输出基金名称"
    
    response = ask_image(prompt, request.image_base64, system_prompt)
    
    try:
        result_json = parse_json(response)
    except Exception as e:
        logger.info("Failed to parse JSON")
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

    fund_names = result_json["fund_names"]
    
    # 根据基金名称在向量数据库中匹配基金名称
    fund_names_matched = []
    
    for fund_name_raw in fund_names:
        result = query_vector_db(
            collection_name=VECTOR_DB_COLLECTION_NAME,
            query=fund_name_raw,
            n_results=1
        )
        fund_name_real = result['data']['metadatas'][0][0]['document_name']
        fund_id = result['data']['metadatas'][0][0]['document_id']
        
        response = vllm_qwen.chat(
            prompt=f"基金名称1：{fund_name_raw}\n基金名称2：{fund_name_real}",
            system_prompt=dedent("""
            # 任务：判断是否为同一支基金

            ## 指令
            - 判断基金名称1和基金名称2是否为同一支基金

            ## 输出格式
            True or False
            """),
            max_tokens=MAX_TOKENS_QWEN,
            temperature=TEMPERATURE_QWEN
        )
        
        print(response)
        
        if "true" not in response.lower():
            result = query_vector_db(
                collection_name=VECTOR_DB_COLLECTION_NAME,
                query=fund_name_raw,
                n_results=10
            )
            fund_name_real = result['data']['metadatas'][0][0]['document_name']
            fund_id = result['data']['metadatas'][0][0]['document_id']
            
            response = vllm_qwen.chat(
                prompt=f"基金名称1：{fund_name_raw}\n基金名称2：{fund_name_real}",
                system_prompt=dedent("""
                # 任务：判断是否为同一支基金

                ## 指令
                - 判断基金名称1和基金名称2是否为同一支基金

                ## 输出格式
                True or False
                """),
                max_tokens=MAX_TOKENS_QWEN,
                temperature=TEMPERATURE_QWEN
            )
        
        fund_names_matched.append(FundNameMatch.model_validate({
            "fund_name_raw": fund_name_raw,
            "fund_name_real": fund_name_real,
            "is_same_fund": "true" in response.lower(),
            "fund_id": fund_id
        }))
    
    return GetFundNamesResponse.model_validate({
        "fund_names": fund_names_matched
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app", 
        host=config["server"]["host"], 
        port=config["server"]["port"], 
        reload=True,
        timeout_keep_alive=config["server"]["timeout_keep_alive"]
    )
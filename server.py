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

import base64
from PIL import Image
from io import BytesIO


# 加载配置
with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)

with open(
    Path(__file__).parent / "prompts" / "check" / "system_prompt_task.txt", "r"
) as f:
    SYSTEM_PROMPT_TASK = f.read()

with open(
    Path(__file__).parent / "prompts" / "check" / "system_prompt_scenario.txt", "r"
) as f:
    SYSTEM_PROMPT_SCENARIO = f.read()

if config["check-scenario"]["concurrent_limit"] not in [1, 2, 3, 4]:
    raise ValueError("Concurrent limit must be 1, 2, 3, or 4")

HOST = config["server"]["host"]
PORT = config["server"]["port"]

### QWEN 2.5 72B ###
# VLLM_HOST_QWEN = config["vllm-qwen25-72b"]["host"]
# VLLM_PORT_QWEN = config["vllm-qwen25-72b"]["port"]
# MAX_CONTEXT_LENGTH_QWEN = int(config["vllm-qwen25-72b"]["max_context_length"] * 0.95)
# MAX_TOKENS_QWEN = config["vllm-qwen25-72b"]["max_tokens"]
# TEMPERATURE_QWEN = config["vllm-qwen25-72b"]["temperature"]

### QWEN 3 32B ###
VLLM_HOST_QWEN = config["vllm-qwen3-32b"]["host"]
VLLM_PORT_QWEN = config["vllm-qwen3-32b"]["port"]
MAX_CONTEXT_LENGTH_QWEN = int(config["vllm-qwen3-32b"]["max_context_length"] * 0.95)
MAX_TOKENS_QWEN = config["vllm-qwen3-32b"]["max_tokens"]
TEMPERATURE_QWEN = config["vllm-qwen3-32b"]["temperature"]

MAX_INPUT_LENGTH_QWEN = MAX_CONTEXT_LENGTH_QWEN - MAX_TOKENS_QWEN

VLLM_HOST_QWEN_VL = config["vllm-qwen25-vl-32b"]["host"]
VLLM_PORT_QWEN_VL = config["vllm-qwen25-vl-32b"]["port"]
MAX_CONTEXT_LENGTH_QWEN_VL = int(
    config["vllm-qwen25-vl-32b"]["max_context_length"] * 0.95
)
MAX_TOKENS_QWEN_VL = config["vllm-qwen25-vl-32b"]["max_tokens"]
TEMPERATURE_QWEN_VL = config["vllm-qwen25-vl-32b"]["temperature"]
MAX_INPUT_LENGTH_QWEN_VL = MAX_CONTEXT_LENGTH_QWEN_VL - MAX_TOKENS_QWEN_VL

VECTOR_DB_HOST = config["vector-db"]["host"]
VECTOR_DB_PORT = config["vector-db"]["port"]
VECTOR_DB_COLLECTION_NAME = config["vector-db"]["collection_name"]

vllm_qwen = VllmClient(
    VLLM_HOST_QWEN, VLLM_PORT_QWEN, MAX_TOKENS_QWEN, TEMPERATURE_QWEN
)

# 创建日志目录
LOG_DIR = "./logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# 配置loguru
logger.add(
    f"{LOG_DIR}/service.log",
    rotation="3 MB",  # 每3MB分割新文件
    retention=10,  # 保留最近10个文件
    compression="zip",  # 旧日志压缩保存
    enqueue=True,  # 线程安全
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    backtrace=True,  # 记录异常堆栈
    diagnose=True,  # 显示变量值
    level="INFO",
)

tokenizer_path = Path.cwd() / "model" / "qwen25-72b-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

app = FastAPI()


def remove_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text

def parse_json(text: str) -> dict | None:
    # 使用正则表达式提取JSON字符串
    json_match = re.search(r"```json(.*?)```", text, re.DOTALL)

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
                    prompt=prompt_total, system_prompt=system_prompt
                )
                result = remove_think(result)
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
            temperature=temperature,
        )
        result = remove_think(result)
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
                prompt_total = (
                    f"[现有摘要]: \n{result}\n\n[额外通话内容]: \n{prompt_accumulate}"
                )
                result = vllm_qwen.chat(
                    prompt=prompt_total,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                result = remove_think(result)
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
            temperature=temperature,
        )
        result = remove_think(result)
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
    max_input_length = MAX_CONTEXT_LENGTH_QWEN
    max_tokens = MAX_TOKENS_QWEN
    temperature = TEMPERATURE_QWEN
    vllm_scenario = vllm_qwen

    system_prompt_list = []
    for i in range(len(request.task.key_points)):
        for j in range(len(request.task.key_points[i].scenarios)):
            try:
                system_prompt_scenario_template = (
                    request.system_prompt_scenario or SYSTEM_PROMPT_SCENARIO
                )

                system_prompt_scenario = system_prompt_scenario_template.format(
                    content_type=request.content_type,
                    key_point_name=request.task.key_points[i].key_point_name,
                    key_point_description=request.task.key_points[
                        i
                    ].key_point_description,
                    scenario_name=request.task.key_points[i].scenarios[j].scenario_name,
                    scenario_description=request.task.key_points[i]
                    .scenarios[j]
                    .scenario_description,
                    scenario_reference=request.task.key_points[i]
                    .scenarios[j]
                    .scenario_reference,
                )
            except Exception as e:
                logger.error(e)
                raise HTTPException(status_code=500, detail=str(e))

            if (
                count_tokens(system_prompt_scenario + request.content)
                > max_input_length
            ):
                error_scenario = request.task.key_points[i].scenarios[j].scenario_name
                error_msg = (
                    f"Scenario {error_scenario} exceeds token limit {max_input_length}"
                )
                raise HTTPException(status_code=400, detail=error_msg)

            system_prompt_list.append(system_prompt_scenario)

    ####################################
    ### Asyncronously Scenario Check ###
    ####################################
    concurrency_limit = config["check-scenario"]["concurrent_limit"]
    semaphore = Semaphore(concurrency_limit)

    # Create a list of coroutines to run concurrently
    async def check_scenario(system_prompt):
        try:
            async with semaphore:
                result = await vllm_scenario.async_chat(
                    prompt=request.content or "执行系统指令",
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = remove_think(result)
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

    ##################
    ### Check task ###
    ##################
    output_json_format = (
        "```json\n"
        + json.dumps(
            {
                "检查结论": "直接陈述您的结论，确保简洁明了(type: str)",
                "风险等级": "提供问题相关的风险等级，使用清晰的评级，如“无风险”、“低风险”、“中风险”、“高风险”(type: str)",
                "风险依据": "解释支持您结论的风险依据，包括数据、研究或逻辑推理(type: str)",
                "优化建议": "提供优化建议，包括更改方案、降低风险等等(type: str)",
            },
            ensure_ascii=False,
        )
        + "\n```"
    )

    system_prompt_task_template = request.system_prompt_task or SYSTEM_PROMPT_TASK

    system_prompt_task = system_prompt_task_template.format(
        content_type=request.content_type,
        task_name=request.task.task_name,
        task_description=request.task.task_description,
        check_result_scenarios=check_result_scenarios_str,
        output_json_format=output_json_format,
    )

    if count_tokens(system_prompt_task + request.content) > MAX_INPUT_LENGTH_QWEN:
        raise HTTPException(
            status_code=400, detail="System prompt and content exceeds token limit"
        )

    try:
        result = vllm_qwen.chat(
            prompt=request.content or "执行系统指令",
            system_prompt=system_prompt_task,
            max_tokens=MAX_TOKENS_QWEN,
            temperature=TEMPERATURE_QWEN,
        )
        result = remove_think(result)
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result_json = parse_json(result)
    except Exception as e:
        logger.info("Failed to parse JSON")
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

    result_json["场景检查结果"] = check_result_scenarios

    return CheckTaskResponse.model_validate(result_json)


async def ask_image(
    prompt: str,
    image_base64: str,
    system_prompt: str | None = None,
    history: list | None = None,
) -> str:
    url = f"http://{VLLM_HOST_QWEN_VL}:{VLLM_PORT_QWEN_VL}/ask-image"
    
    width, height = get_image_size(image_base64)
    
    image_tokens = int((width * height) / 768)
    
    if image_tokens > MAX_INPUT_LENGTH_QWEN_VL:
        raise HTTPException(
            status_code=400, 
            detail=f"Image exceeds token limit {MAX_INPUT_LENGTH_QWEN_VL}"
        )
    
    payload = {
        "prompt": prompt,
        "image_base64": image_base64,
        "system_prompt": system_prompt or "",
        "history": history or [],
        "max_tokens": MAX_TOKENS_QWEN_VL,
        "temperature": TEMPERATURE_QWEN_VL,
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["content"]

def get_image_size(base64_string):
    # 分割base64字符串，去掉前缀
    base64_data = base64_string.split(',', 1)[1]
    
    # 将base64字符串解码为字节流
    image_bytes = base64.b64decode(base64_data)
    
    # 将字节流转换为图片对象
    try:
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError("无效的图片数据") from e
    
    # 获取图片的宽度和高度
    width, height = image.size
    
    return (width, height)


class QueryVectorDB(BaseModel):
    collection_name: str
    query: str
    n_results: int = 10
    rerank: bool = True
    metadata_filter: dict | None = None


def query_vector_db(request: QueryVectorDB) -> dict:
    end_point = "query"

    url = f"http://{VECTOR_DB_HOST}:{VECTOR_DB_PORT}/{end_point}"

    payload = json.dumps(
        {
            "collection_name": request.collection_name,
            "query": request.query,
            "n_results": request.n_results,
            "rerank": request.rerank,
            "metadata_filter": request.metadata_filter or {},
        }
    )

    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()

    return response.json()


class GetFundNamesRequest(BaseModel):
    image_base64: str


class FundNameMatch(BaseModel):
    fund_name_captured: str
    fund_name_official: str
    is_same_fund: bool
    fund_id: str


class GetFundNamesResponse(BaseModel):
    fund_names: list[FundNameMatch]


@app.post("/get-fund-names", response_model=GetFundNamesResponse)
async def get_fund_names(request: GetFundNamesRequest) -> GetFundNamesResponse:
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

    response = await ask_image(prompt, request.image_base64, system_prompt)

    try:
        result_json = parse_json(response)
    except Exception as e:
        logger.info("Failed to parse JSON")
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

    fund_names = result_json["fund_names"]

    # 根据基金名称在向量数据库中匹配基金名称
    fund_names_matched = []

    for fund_name_captured in fund_names:
        result = query_vector_db(
            QueryVectorDB(
                collection_name=VECTOR_DB_COLLECTION_NAME, 
                query=fund_name_captured, 
                n_results=1
            )
        )
        fund_name_official = result["data"]["metadatas"][0][0]["document_name"]
        fund_id = result["data"]["metadatas"][0][0]["document_id"]
        
        response = vllm_qwen.chat(
            prompt=f"基金名称1：{fund_name_captured}\n基金名称2：{fund_name_official}",
            system_prompt=dedent("""
            # 任务：判断是否为同一支基金

            ## 指令
            - 判断基金名称1和基金名称2是否为同一支基金

            ## 输出格式
            True or False
            """),
            max_tokens=MAX_TOKENS_QWEN,
            temperature=TEMPERATURE_QWEN,
        )

        print(response)
        
        # 如果基金名称不匹配，扩大范围重新匹配
        if "true" not in response.lower():
            result = query_vector_db(
                QueryVectorDB(
                    collection_name=VECTOR_DB_COLLECTION_NAME,
                    query=fund_name_captured,
                    n_results=10,
                )
            )
            fund_name_official = result["data"]["metadatas"][0][0]["document_name"]
            fund_id = result["data"]["metadatas"][0][0]["document_id"]

            response = vllm_qwen.chat(
                prompt=f"基金名称1：{fund_name_captured}\n基金名称2：{fund_name_official}",
                system_prompt=dedent("""
                # 任务：判断是否为同一支基金

                ## 指令
                - 判断基金名称1和基金名称2是否为同一支基金

                ## 输出格式
                True or False
                """),
                max_tokens=MAX_TOKENS_QWEN,
                temperature=TEMPERATURE_QWEN,
            )

        fund_names_matched.append(
            FundNameMatch.model_validate(
                {
                    "fund_name_captured": fund_name_captured,
                    "fund_name_official": fund_name_official,
                    "is_same_fund": "true" in response.lower(),
                    "fund_id": fund_id,
                }
            )
        )

    return GetFundNamesResponse.model_validate({"fund_names": fund_names_matched})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=True,
        timeout_keep_alive=config["server"]["timeout_keep_alive"],
    )

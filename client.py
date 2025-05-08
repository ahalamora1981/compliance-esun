# client.py
import requests
import tomllib
from pathlib import Path
from pydantic import BaseModel
import time
import json
import base64
from tqdm import tqdm


with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)


class Request(BaseModel):
    prompt: str
    system_prompt: str
    temperature: float = 0.0


class Response(BaseModel):
    result: str

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:0.4f} seconds")
        return result
    return wrapper

def load_correction_prompts(media_type: str) -> tuple[str, str]:
    system_prompt_path = Path(__file__).parent / "prompts" / "correction" / media_type / "system_prompt.txt"
    user_prompt_path = Path(__file__).parent / "prompts" / "correction" / media_type / "user_prompt.txt"
    
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_path, "r") as f:
        prompt = f.read()
    
    return system_prompt, prompt

def load_summary_prompts(media_type: str) -> tuple[str, str]:
    system_prompt_path = Path(__file__).parent / "prompts" / "summary" / media_type / "system_prompt.txt"
    user_prompt_path = Path(__file__).parent / "prompts" / "summary" / media_type / "user_prompt.txt"
    
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_path, "r") as f:
        prompt = f.read()
    
    return system_prompt, prompt

# def load_check_scenario_prompts() -> tuple[list[str], str]:
#     system_prompt_path = Path(__file__).parent / "prompts" / "check" / "system_prompt_scenario.txt"
#     content_path = Path(__file__).parent / "prompts" / "check" / "content.txt"
    
#     with open(system_prompt_path, "r") as f:
#         system_prompt_template = f.read()
#     with open(content_path, "r") as f:
#         content = f.read()
        
#     with open(Path(__file__).parent / "prompts" / "check" / "task.json", "r") as f:
#         task = json.load(f)['task']
        
#     system_prompt_list = []
        
#     for i in range(len(task['key_points'])):
#         for j in range(len(task['key_points'][i]['scenarios'])):
#             system_prompt = system_prompt_template.format(
#                 key_point_name = task['key_points'][i]['key_point_name'],
#                 key_point_description = task['key_points'][i]['key_point_description'],
#                 scenario_name=task['key_points'][i]['scenarios'][j]['scenario_name'],
#                 scenario_description=task['key_points'][i]['scenarios'][j]['scenario_description']
#             )
#             system_prompt_list.append(system_prompt)
    
#     return system_prompt_list, content

@timer
def test_correction():
    # media_type = "call"
    media_type = "meeting"
    # media_type = "stream_and_video"
    # media_type = "social_media"
    
    system_prompt, prompt = load_correction_prompts(media_type)
    url = f"{BASE_URL}/correction"
    
    request = Request(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.0
    )

    response = requests.post(
        url=url,
        json=request.model_dump(),
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    response_model = Response.model_validate(response.json())
    result = response_model.result
    
    # 如果是R1模型，则去掉思考的部分
    # if "</think>" in result:
    #     result = result.split("</think>")[-1]
    
    print(result)
    
@timer
def test_summary():
    # media_type = "call"
    media_type = "meeting"
    
    system_prompt, prompt = load_summary_prompts(media_type)
    url = f"{BASE_URL}/summary"
    
    request = Request(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.0
    )

    response = requests.post(
        url=url,
        json=request.model_dump(),
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    response_model = Response.model_validate(response.json())
    result = response_model.result
    
    # 如果是R1模型，则去掉思考的部分
    # if "</think>" in result:
    #     result = result.split("</think>")[-1]
    
    print(result)

@timer
def test_check():
    url = f"{BASE_URL}/check-task"
    
    file_name = "check_task_request_test.json"
    with open(Path(__file__).parent / "prompts" / "check" / file_name, "r") as f:
        request_data = json.load(f)
    
    # SEND REQUEST
    response = requests.post(
        url=url,
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    
    with open(f"output/check_task_output_{time.strftime('%y_%m_%dT%H_%M_%S')}.json", "w") as f:
        json.dump(response.json(), f, indent=2, ensure_ascii=False)


@timer
def test_get_fund_names(image_path: str):
    url = f"{BASE_URL}/get-fund-names"
    
    # load image from file and convert to base64
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
        image_base64 = f"data:image/jpeg;base64,{image_base64}"
        
    json_data = {
        "image_base64": image_base64
    }
    
    response = requests.post( 
        url=url,
        json=json_data,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def add_document(
    collection_name: str, 
    document_name: str,
    document_id: str,
    document: str,
    metadata: dict[str, str | int | float] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None
) -> dict:
    end_point = "add-document"
    url = f"http://{config['vector-db']['host']}:{config['vector-db']['port']}/{end_point}"

    payload = json.dumps({
        "collection_name": collection_name,
        "document_name": document_name,
        "document_id": document_id,
        "document": document,
        "metadata": metadata or {},
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response.raise_for_status()
    
    return response.json()


def test_add_fund_names():
    with open(Path(__file__).parent / "docs" / "all_public_fund.txt", "r") as f:
        content = f.read()
    
    funds = content.split("| 基金吧 | 档案")
    
    for fund in tqdm(funds):
        fund_id = fund[1:7]
        fund_name = fund[8:]
        add_document(
            collection_name=config["vector-db"]["collection_name"],
            document_name=fund_name,
            document_id=fund_id,
            document=fund_name
        )
    
    
if __name__ == "__main__":
    # BASE_URL = "http://localhost:8008"
    BASE_URL = "http://10.101.100.13:8010"
    
    # test_correction()
    
    test_summary()

    # test_check()
    
    # image_path = "image/万家01.jpg"
    # image_path = "image/fund_names_01.png"
    # test_get_fund_names(image_path=image_path)
    
    # test_add_fund_names()
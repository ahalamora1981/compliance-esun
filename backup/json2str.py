import json


with open("check_task_request_test.json", "r") as f:
    request_data = json.load(f)
    
with open("output_str.txt", "w") as f:
    # json_str = json.dumps(request_data, ensure_ascii=False)
    # json.dump(json_str, f, ensure_ascii=False)
    json.dump(request_data, f, ensure_ascii=False) 
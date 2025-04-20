from pathlib import Path


with open(Path(__file__).parent / "all_public_fund.txt", "r") as f:
    content = f.read()
    
funds = content.split("| 基金吧 | 档案")
funds = [(fund[1:7], fund[8:]) for fund in funds]

print(funds[-10:])
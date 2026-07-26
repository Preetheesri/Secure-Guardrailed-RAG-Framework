import json

with open("data/processed/splits_small/train.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 15:
            break
        record = json.loads(line)
        print(f"{i+1}. {record['question']}")
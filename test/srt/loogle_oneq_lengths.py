import ast
import json

from transformers import AutoTokenizer


dataset = "/mnt/data/dpser/datasets/loogle/longdep_qa_sglang_qa_pairs.jsonl"
model = "/mnt/data/models/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
lens = []
rows = []
with open(dataset) as fin:
    for index, line in enumerate(fin):
        row = json.loads(line)
        qa = ast.literal_eval(row["qa_pairs"])[0]
        text = "Input: " + row["input"] + " Question: " + qa["Q"]
        length = len(tokenizer.encode(text))
        lens.append(length)
        rows.append((index, row.get("doc_id"), length))
        if index >= 63:
            break

print("first_rows", rows[:24])
for count in [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]:
    print(count, sum(lens[:count]))

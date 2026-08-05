import json
from datasets import load_dataset

# 1. Load the raw dataset
dataset = load_dataset("openai/openai_humaneval")
data_points = dataset["test"]

# 2. Build your array of dictionary items
rag_documents = []
for item in data_points:
    rag_documents.append({
        "prompt": item["prompt"],
        "task_id": item["task_id"],
        "entry_point": item["entry_point"],
        "solution": item["canonical_solution"],
        "test_code": item["test"]
    })


with open("raw_humaneval_array.json", "w", encoding="utf-8") as f:
    json.dump(rag_documents, f, indent=4)

print(f"Saved {len(rag_documents)} items to raw_humaneval_array.json!")
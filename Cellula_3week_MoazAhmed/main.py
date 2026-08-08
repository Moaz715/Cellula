import os
import json

# Define the file path (adjust if you named it differently)
file_path = "raw_humaneval_array.json"

print("=== Starting Data Integrity Check ===")

# 1. Check if the file actually exists
if not os.path.exists(file_path):
    print(f"❌ Error: The file '{file_path}' was not found.")
    print("Please make sure you ran 'ingest.py' successfully first.")
    exit(1)
else:
    print(f"✅ Success: Found '{file_path}'.")

# 2. Check the file size
file_size_kb = os.path.getsize(file_path) / 1024
print(f"ℹ️ File Size: {file_size_kb:.2f} KB")

try:
    # 3. Try to parse the JSON data
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("✅ Success: JSON syntax is valid and readable.")
    
    # 4. Verify the number of problems (HumanEval contains exactly 164 problems)
    total_items = len(data)
    print(f"ℹ️ Total items found: {total_items}")
    if total_items == 164:
        print("✅ Success: All 164 HumanEval items are present.")
    else:
        print(f"⚠️ Warning: Expected 164 items, but found {total_items}.")

    # 5. Check structural completeness of the first item
    required_keys = ["prompt", "task_id", "entry_point", "solution", "test_code"]
    first_item = data[0]
    missing_keys = [key for key in required_keys if key not in first_item]
    
    if not missing_keys:
        print("✅ Success: All required fields are present in the dataset.")
    else:
        print(f"❌ Error: Missing keys {missing_keys} in the data structure.")

    # 6. Print a preview sample of the first problem
    print("\n=== Data Sample Preview ===")
    print(f"Task ID    : {first_item.get('task_id')}")
    print(f"Entry Point: {first_item.get('entry_point')}")
    print("-" * 30)
    print("Prompt Snippet:")
    # Print just the first few lines of the prompt
    prompt_lines = first_item.get('prompt', '').split('\n')[:5]
    print('\n'.join(prompt_lines) + "\n...")
    print("-" * 30)

except json.JSONDecodeError:
    print("❌ Error: The file exists but contains corrupted/invalid JSON format.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {str(e)}")

print("\n=== Validation Complete ===")
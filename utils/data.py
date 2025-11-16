from constants import *

def load_jsonl_data(file_path: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file line by line into a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Successfully loaded {len(data)} records from '{file_path}'.")
    except FileNotFoundError:
        print(f"ERROR: Could not find the file at '{file_path}'.")
    return data
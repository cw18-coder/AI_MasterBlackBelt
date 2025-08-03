import os
import json
from typing import List, Optional, Dict
import yaml

def rename_json_keys_in_batches(
    path: str,
    file_prefix: str,
    batch_num: Optional[List[int]] = None,
    rename_dict: Dict[str, str] = None
):
    """
    Renames keys in JSON files for specified batches.
    Args:
        path (str): Directory containing the sample files.
        file_prefix (str): Common file prefix (e.g., 'lss_cot_batch').
        batch_num (Optional[List[int]]): List of batch numbers to process. If None, process all files with prefix.
        rename_dict (dict): Dictionary mapping old key names to new key names.
    """
    if rename_dict is None:
        raise ValueError("rename_dict must be provided.")

    # Gather files to process
    files = []
    if batch_num is None:
        # Process all files with prefix
        for fname in os.listdir(path):
            if fname.startswith(file_prefix) and fname.endswith('.json'):
                files.append(os.path.join(path, fname))
    else:
        # Only process specified batch numbers
        for num in batch_num:
            fname = f"{file_prefix}{num}.json"
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                files.append(fpath)
            else:
                print(f"Warning: {fpath} not found.")

    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue

        # Rename keys in each sample
        changed = False
        for sample in data:
            for old_key, new_key in rename_dict.items():
                if old_key in sample:
                    sample[new_key] = sample.pop(old_key)
                    changed = True
        if changed:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Updated keys in {fpath}")
        else:
            print(f"No changes needed for {fpath}")

def insert_json_keys_in_batches(
    path: str,
    file_prefix: str,
    batch_num: Optional[List[int]] = None,
    insert_dict: Dict[str, any] = None
):
    """
    Inserts keys and their values into JSON files for specified batches.
    Args:
        path (str): Directory containing the sample files.
        file_prefix (str): Common file prefix (e.g., 'lss_cot_batch').
        batch_num (Optional[List[int]]): List of batch numbers to process. If None, process all files with prefix.
        insert_dict (dict): Dictionary of key-value pairs to insert.
    """
    if insert_dict is None:
        raise ValueError("insert_dict must be provided.")

    files = []
    if batch_num is None:
        for fname in os.listdir(path):
            if fname.startswith(file_prefix) and fname.endswith('.json'):
                files.append(os.path.join(path, fname))
    else:
        for num in batch_num:
            fname = f"{file_prefix}{num}.json"
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                files.append(fpath)
            else:
                print(f"Warning: {fpath} not found.")

    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue

        changed = False
        for sample in data:
            for key, value in insert_dict.items():
                if key not in sample:
                    sample[key] = value
                    changed = True
        if changed:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Inserted keys in {fpath}")
        else:
            print(f"No insertions needed for {fpath}")

def delete_json_keys_in_batches(
    path: str,
    file_prefix: str,
    batch_num: Optional[List[int]] = None,
    delete_keys: List[str] = None
):
    """
    Deletes specified keys from JSON files for specified batches.
    Args:
        path (str): Directory containing the sample files.
        file_prefix (str): Common file prefix (e.g., 'lss_cot_batch').
        batch_num (Optional[List[int]]): List of batch numbers to process. If None, process all files with prefix.
        delete_keys (List[str]): List of keys to delete from each sample.
    """
    if delete_keys is None:
        raise ValueError("delete_keys must be provided.")

    files = []
    if batch_num is None:
        for fname in os.listdir(path):
            if fname.startswith(file_prefix) and fname.endswith('.json'):
                files.append(os.path.join(path, fname))
    else:
        for num in batch_num:
            fname = f"{file_prefix}{num}.json"
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                files.append(fpath)
            else:
                print(f"Warning: {fpath} not found.")

    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue

        changed = False
        for sample in data:
            for key in delete_keys:
                if key in sample:
                    del sample[key]
                    changed = True
        if changed:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Deleted keys in {fpath}")
        else:
            print(f"No deletions needed for {fpath}")

def main_from_yaml(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    cfg = config.get('update_json_samples', {})

    path = cfg.get('path')
    file_prefix = cfg.get('file_prefix')
    batch_start = cfg.get('batch_start')
    batch_end = cfg.get('batch_end')
    operation = cfg.get('operation')
    rename_dict = cfg.get('rename_dict')
    insert_dict = cfg.get('insert_dict')
    delete_keys = cfg.get('delete_keys')

    batch_num = None
    if batch_start is not None and batch_end is not None:
        batch_num = list(range(batch_start, batch_end + 1))

    if operation == 'update':
        rename_json_keys_in_batches(
            path=path,
            file_prefix=file_prefix,
            batch_num=batch_num,
            rename_dict=rename_dict
        )
    elif operation == 'insert':
        insert_json_keys_in_batches(
            path=path,
            file_prefix=file_prefix,
            batch_num=batch_num,
            insert_dict=insert_dict
        )
    elif operation == 'delete':
        delete_json_keys_in_batches(
            path=path,
            file_prefix=file_prefix,
            batch_num=batch_num,
            delete_keys=delete_keys
        )
    else:
        print(f"Unknown operation: {operation}")

if __name__ == "__main__":
    # Read config from YAML and execute
    main_from_yaml("./config.yaml")

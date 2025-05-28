import json
from pathlib import Path
from dataset import NERDataset
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader
from typing import List, Dict

def load_json_dataset(json_path: str):
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    return data

def get_ner_dataset(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    label2id:Dict,
    id2label:Dict,
    label_type: str = "fine",
    max_length: int = 128,
    label_all_tokens: bool = False,
    json_base_path: str = "files"
):
    path = Path(json_base_path) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    raw_data = load_json_dataset(str(path))
    return NERDataset(
        data=raw_data,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        label_type=label_type,
        max_length=max_length,
        label_all_tokens=label_all_tokens
    )

def get_dataloader(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    id2label:Dict,
    label2id:Dict,
    batch_size: int = 16,
    max_length: int = 128,
    label_type: str = "fine",
    label_all_tokens: bool = False,
    json_base_path: str = "files",
    shuffle: bool = True
):

    dataset = get_ner_dataset(
        split=split,
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        label_type=label_type,
        max_length=max_length,
        label_all_tokens=label_all_tokens,
        json_base_path=json_base_path
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
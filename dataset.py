import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Dict

def tokenize_and_align_labels(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    label2id:Dict,
    id2label:Dict,
    label_all_tokens: bool = False,
    label_type: str = "fine" # fix "fine"
) -> Dict[str, List]:
    
    label_key = "fine_ner_tags_bio" if label_type == "fine" else "ner_tags"

    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    
    # 토큰 라벨 매핑
    labels = []
    for i, word_labels in enumerate(examples[label_key]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                continue

            label_id = word_labels[word_idx]
            label_str = id2label[label_id]

            if word_idx != previous_word_idx:
                label_ids.append(word_labels[word_idx])
            else:
                if label_all_tokens:
                    if label_str.startswith('B-'):
                        i_label_str = "I-" + label_str[2:]
                        i_label_id = label2id[i_label_str]
                        label_ids.append(i_label_id)
                    else:
                        label_ids.append(label_id)
                else:
                    label_ids.append(-100)
                                    
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, id2label, max_length, label_type="fine", label_all_tokens=False):
        self.data = data
        self.tokenizer = tokenizer
        self.label_type = label_type
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.label2id = label2id
        self.id2label = id2label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        label_key = "fine_ner_tags_bio" if self.label_type == "fine" else "ner_tags"
        example = {
            "tokens": [item["tokens"]],
            label_key: [item[label_key]],
        }

        tokenized = tokenize_and_align_labels(
            example,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            id2label=self.id2label,
            label_type=self.label_type,
            max_length=self.max_length,
            label_all_tokens=self.label_all_tokens,
        )

        return {
            key: torch.tensor(val[0], dtype=torch.long)
            for key, val in tokenized.items()
            if key in ["input_ids", "attention_mask", "labels"]
        }
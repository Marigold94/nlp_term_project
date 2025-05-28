import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils import get_dataloader
from tqdm import tqdm
import argparse
import json
import os
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from collections import Counter

def predict(model, dataloader, label_map, device):
    model.eval()
    all_results = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.argmax(dim=-1).cpu().tolist()
            label_ids = labels.tolist()

            for pred_ids, label_ids_row in zip(logits, label_ids):
                pred_labels, true_labels = [], []
                for p, l in zip(pred_ids, label_ids_row):
                    if l == -100:
                        continue
                    pred_labels.append(label_map[p])
                    true_labels.append(label_map[l])
                all_preds.append(pred_labels)
                all_labels.append(true_labels)
                all_results.append({"predicted_labels": pred_labels, "labels":true_labels})

    return all_results, all_labels, all_preds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.label_map_path, f"{args.label_type}_id2label.json"), "r") as f:
        id2label = json.load(f)

    with open(os.path.join(args.label_map_path, f"{args.label_type}_label2id.json"), "r") as f:
        label2id = json.load(f)
    
    id2label = {int(k): v for k, v in id2label.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)

    test_loader = get_dataloader(
        split="test",
        tokenizer=tokenizer,
        label2id=label2id,
        id2label=id2label,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label_type=args.label_type,
        label_all_tokens=False,
        json_base_path=args.data_dir,
        shuffle=False
    )

    results, all_labels, all_preds = predict(model, test_loader, id2label, device)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = {
        "precision": precision_score(all_labels, all_preds, average="micro").item(),
        "recall": recall_score(all_labels, all_preds, average="micro").item(),
        "f1": f1_score(all_labels, all_preds, average="micro").item(),
    }
    
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
        
    with open(os.path.join(args.output_dir,"metrics.json"), 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="data/files/")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--label_map_path", type=str, default="data/bio_label_ids")
    parser.add_argument("--label_type", type=str, default="fine")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    
    main(args)
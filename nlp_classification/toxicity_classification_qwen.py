#!/usr/bin/env python3
"""
Train toxicity classification model for Qwen tokenizer.
This script trains a SimpleTransformerClassifier on toxicity classification task
and generates token importance scores for debiasing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")
import json
from transformers import AutoTokenizer
import os
from datasets import load_dataset
from captum.attr import IntegratedGradients
from collections import defaultdict
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen', type=str)
parser.add_argument('--gpu_id', default='3', type=str)

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from model import SimpleTransformerClassifier

def process_text(text, target_class, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=model.max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    embeddings = model.token_embedding(input_ids)
    baseline = torch.zeros_like(embeddings).to(device)
    
    integrated_gradients = IntegratedGradients(model.forward_with_embeddings)
    attributions, delta = integrated_gradients.attribute(
        inputs=embeddings,
        baselines=baseline,
        additional_forward_args=(attention_mask, target_class),
        return_convergence_delta=True
    )
    
    token_attributions = attributions.sum(dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    return tokens, token_attributions

def train_model(model, train_loader, dev_loader, device, num_epochs=10, lr=2e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).long()
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        acc = evaluate_model(model, dev_loader, device)
        print(f"Dev Accuracy: {acc:.4f}")
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0

def run_experiment_pytorch(experiment_name, tokenizer, model_save_dir, imp_save_file,
                           num_train_epochs=10, batch_size=16, max_length=128,
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print(f"\n====== Running experiment: {experiment_name} ======")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    toxicity_data_url = ("https://github.com/conversationai/unintended-ml-bias-analysis/"
                        "raw/e02b9f12b63a39235e57ba6d3d62d8139ca5572c/data/")
    train_dataset = pd.read_csv(toxicity_data_url + "wiki_train.csv")
    dev_dataset = pd.read_csv(toxicity_data_url + "wiki_dev.csv")
    test_dataset = pd.read_csv(toxicity_data_url + "wiki_test.csv")
    
    Y_train = train_dataset["is_toxic"].values.astype(float)
    Y_val = dev_dataset["is_toxic"].values.astype(float)
    Y_test = test_dataset["is_toxic"].values.astype(float)

    class ToxicityDataset(Dataset):
        def __init__(self, dataframe, labels, tokenizer, max_length=128):
            self.data = dataframe["comment"].tolist()
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            encoding = self.tokenizer(self.data[idx], truncation=True, padding="max_length",
                                    max_length=self.max_length, return_tensors="pt")
            item = {key: val.squeeze(0) for key, val in encoding.items()}
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
            item["raw_text"] = self.data[idx]
            return item

    train_ds = ToxicityDataset(train_dataset, Y_train, tokenizer, max_length)
    dev_ds = ToxicityDataset(dev_dataset, Y_val, tokenizer, max_length)
    test_ds = ToxicityDataset(test_dataset, Y_test, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    vocab_size = tokenizer.vocab_size
    model = SimpleTransformerClassifier(vocab_size=vocab_size, max_length=max_length)
    model.to(device)
    
    print("Training model...")
    model = train_model(model, train_loader, dev_loader, device, num_epochs=num_train_epochs)
    
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_dir, "pytorch_model.bin"))
    print(f"Model saved to {model_save_dir}")
    
    print("Computing token importances...")
    importance_dict = {"toxic": defaultdict(float), "non-toxic": defaultdict(float)}
    
    model.eval()
    for split_name, dataset in [("train", train_ds), ("dev", dev_ds)]:
        print(f"Processing {split_name} set...")
        for i in range(min(1000, len(dataset))):  # Sample for efficiency
            example = dataset[i]
            text = example["raw_text"]
            label = example["label"].item()
            
            target_class = label  # 0=non-toxic, 1=toxic
            tokens, attributions = process_text(text, target_class, tokenizer, model, device)
            
            toxicity_key = "toxic" if label == 1 else "non-toxic"
            for token, attr in zip(tokens, attributions):
                importance_dict[toxicity_key][token] += attr.item()
    
    # Normalize importances
    for key in ["toxic", "non-toxic"]:
        total = sum(importance_dict[key].values())
        if total > 0:
            importance_dict[key] = {k: v/total for k, v in importance_dict[key].items()}
    
    with open(imp_save_file, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"Importance dictionary saved to {imp_save_file}")

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    model_save_dir = f"toxicity_model_{args.model}_pytorch"
    imp_save_file = f"importance_toxicity_dict_{args.model}_pytorch.json"
    
    run_experiment_pytorch(
        experiment_name=f"Qwen Toxicity Classification",
        tokenizer=tokenizer,
        model_save_dir=model_save_dir,
        imp_save_file=imp_save_file,
        num_train_epochs=10,
        batch_size=16,
        max_length=128,
        device=device
    )





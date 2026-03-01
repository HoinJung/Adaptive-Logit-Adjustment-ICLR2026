#!/usr/bin/env python3
"""
Train gender classification model for Qwen tokenizer.
This script trains a SimpleTransformerClassifier on gender classification task
and generates token importance scores for debiasing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen', type=str)
parser.add_argument('--gpu_id', default='3', type=str)

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from model import SimpleTransformerClassifier

def tokenize_and_format(examples, tokenizer, max_length=128):
    tokenized = tokenizer(examples["hard_text"], truncation=True, padding="max_length", max_length=max_length)
    tokenized["label"] = (1 - np.array(examples["gender"])).tolist()  # Convert back to list after inversion
    tokenized["raw_text"] = examples["hard_text"]
    return tokenized

def collate_fn(batch):
    out = {}
    tensor_keys = ["input_ids", "attention_mask", "label"]
    for key in tensor_keys:
        out[key] = default_collate([b[key] for b in batch])
    out["raw_text"] = [b["raw_text"] for b in batch]
    return out

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

def train_model(model, train_loader, dev_loader, device, num_epochs=3, lr=2e-5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
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
                           num_train_epochs=3, batch_size=16, max_length=128,
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print(f"\n====== Running experiment: {experiment_name} ======")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("LabHC/bias_in_bios", split="train")
    dev_dataset   = load_dataset("LabHC/bias_in_bios", split="dev")
    test_dataset  = load_dataset("LabHC/bias_in_bios", split="test")
    
    train_dataset = train_dataset.map(lambda ex: tokenize_and_format(ex, tokenizer, max_length), batched=True)
    dev_dataset   = dev_dataset.map(lambda ex: tokenize_and_format(ex, tokenizer, max_length), batched=True)
    test_dataset  = test_dataset.map(lambda ex: tokenize_and_format(ex, tokenizer, max_length), batched=True)
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
    dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

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
    importance_dict = {"male": defaultdict(float), "female": defaultdict(float)}
    
    model.eval()
    for split_name, dataset in [("train", train_dataset), ("dev", dev_dataset)]:
        print(f"Processing {split_name} set...")
        for i in range(min(1000, len(dataset))):  # Sample for efficiency
            example = dataset[i]
            text = example["raw_text"]
            label = example["label"].item()
            
            target_class = 1 - label  # 0=male, 1=female
            tokens, attributions = process_text(text, target_class, tokenizer, model, device)
            
            gender_key = "male" if label == 0 else "female"
            for token, attr in zip(tokens, attributions):
                importance_dict[gender_key][token] += attr.item()
    
    # Normalize importances
    for gender in ["male", "female"]:
        total = sum(importance_dict[gender].values())
        if total > 0:
            importance_dict[gender] = {k: v/total for k, v in importance_dict[gender].items()}
    
    with open(imp_save_file, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"Importance dictionary saved to {imp_save_file}")

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    model_save_dir = f"gender_model_{args.model}_pytorch_generated"
    imp_save_file = f"importance_dict_{args.model}_pytorch_generated.json"
    
    run_experiment_pytorch(
        experiment_name=f"Qwen Gender Classification",
        tokenizer=tokenizer,
        model_save_dir=model_save_dir,
        imp_save_file=imp_save_file,
        num_train_epochs=3,
        batch_size=16,
        max_length=128,
        device=device
    )













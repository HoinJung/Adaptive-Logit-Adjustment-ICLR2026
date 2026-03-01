#!/usr/bin/env python3
"""
Train bias classification model using McGill-NLP/stereoset dataset for Qwen tokenizer.
This script trains a SimpleTransformerClassifier on stereoset data and generates 
token importance scores for debiasing based on bias_type (gender, race, profession, etc.).
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
from datasets import load_dataset, Dataset
from captum.attr import IntegratedGradients
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='qwen', type=str, help='Model name suffix (qwen, llava, paligemma)')
parser.add_argument('--bias_type', default='gender', type=str, choices=['gender', 'race', 'profession'], 
                    help='Bias type to train on (gender, race, profession)')
parser.add_argument('--gpu_id', default='3', type=str)
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_length', default=128, type=int)

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

from model import SimpleTransformerClassifier

def collate_fn(batch):
    out = {}
    tensor_keys = ["input_ids", "attention_mask", "label"]
    for key in tensor_keys:
        if batch and key in batch[0]:
            out[key] = default_collate([b[key] for b in batch])
    out["raw_text"] = [b["raw_text"] for b in batch] if batch else []
    return out


def process_examples_batch(examples, bias_type):
    """
    Process examples in batches to extract contexts and binary labels.
    Label mapping:
        stereotype      -> 1
        antistereotype  -> 0
        unrelated/other -> excluded
    """
    contexts = []
    labels = []
    
    # Handle case where examples might be a list or dict
    bias_types = examples.get('bias_type', [])
    contexts_raw = examples.get('context', [])
    targets = examples.get('target', [])
    
    if isinstance(bias_types, list):
        bias_types_list = bias_types
    else:
        bias_types_list = [bias_types] if bias_types else []
    
    if isinstance(contexts_raw, list):
        contexts_list = contexts_raw
    else:
        contexts_list = [contexts_raw] if contexts_raw else []
    
    if isinstance(targets, list):
        targets_list = targets
    else:
        targets_list = [targets] if targets else []
    
    max_len = max(len(bias_types_list), len(contexts_list), len(targets_list))
    
    for i in range(max_len):
        # Check if this example matches the bias_type
        if i < len(bias_types_list) and bias_types_list[i] == bias_type:
            context = contexts_list[i] if i < len(contexts_list) else ""
            target = targets_list[i] if i < len(targets_list) else ""
            
            if context and str(context).strip():
                context_str = str(context).strip()
                target_str = str(target).lower().strip() if target else ""
                
                # Create binary labels: 1 for stereotype, 0 for antistereotype
                if 'antistereotype' in target_str:
                    labels.append(0)
                    contexts.append(context_str)
                elif 'stereotype' in target_str:
                    labels.append(1)
                    contexts.append(context_str)
                # Skip unrelated or undefined targets
    
    # Return empty dict structure if no valid examples
    if not contexts:
        return {"contexts": [], "labels": []}
    
    return {"contexts": contexts, "labels": labels}


def shuffle_split_data(data, rng):
    contexts = data.get("contexts", [])
    labels = data.get("labels", [])
    if len(contexts) <= 1:
        return data
    indices = rng.permutation(len(contexts))
    data["contexts"] = [contexts[i] for i in indices]
    data["labels"] = [labels[i] for i in indices]
    return data

def process_text(text, target_class, tokenizer, model, device):
    """Compute token attributions using Integrated Gradients."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=model.max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Clamp input_ids to valid vocabulary range
    vocab_size = model.token_embedding.num_embeddings
    input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    # Ensure attention_mask matches input_ids length
    if attention_mask.size(1) > model.max_length:
        attention_mask = attention_mask[:, :model.max_length]
    if input_ids.size(1) > model.max_length:
        input_ids = input_ids[:, :model.max_length]
    
    embeddings = model.token_embedding(input_ids)
    baseline = torch.zeros_like(embeddings).to(device)
    
    integrated_gradients = IntegratedGradients(model.forward_with_embeddings)
    try:
        attributions, delta = integrated_gradients.attribute(
            inputs=embeddings,
            baselines=baseline,
            additional_forward_args=(attention_mask, target_class),
            return_convergence_delta=True
        )
        
        token_attributions = attributions.sum(dim=-1).squeeze(0)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        return tokens, token_attributions
    except RuntimeError as e:
        print(f"Warning: Integrated Gradients failed for text: {text[:50]}... Error: {e}")
        # Return zero attributions on error
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
        zero_attributions = torch.zeros(len(tokens)).to(device)
        return tokens, zero_attributions

def train_model(model, train_loader, dev_loader, device, num_epochs=3, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    vocab_size = model.token_embedding.num_embeddings
    for epoch in range(num_epochs):
        total_loss = 0
        batch_count = 0
        for batch in train_loader:
            # Check if batch has input_ids and it's not empty
            if "input_ids" not in batch or batch["input_ids"] is None or batch["input_ids"].numel() == 0:
                continue
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Clamp input_ids to valid vocabulary range before passing to model
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            acc = evaluate_model(model, dev_loader, device)
            print(f"Dev Accuracy: {acc:.4f}")
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    vocab_size = model.token_embedding.num_embeddings
    with torch.no_grad():
        for batch in data_loader:
            # Check if batch has input_ids and it's not empty
            if "input_ids" not in batch or batch["input_ids"] is None or batch["input_ids"].numel() == 0:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            # Clamp input_ids to valid vocabulary range before passing to model
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs["logits"], dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0

def run_experiment_pytorch(experiment_name, tokenizer, model_save_dir, imp_save_file,
                           bias_type='gender', num_train_epochs=3, batch_size=16, max_length=128,
                           device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    print(f"\n====== Running experiment: {experiment_name} ======")
    print(f"Bias type: {bias_type}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load stereoset dataset
    print("Loading McGill-NLP/stereoset dataset...")
    try:
        dataset = load_dataset("McGill-NLP/stereoset", "intrasentence")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying without config...")
        dataset = load_dataset("McGill-NLP/stereoset")
    
    # Check available splits
    print(f"Available splits: {list(dataset.keys())}")
    
    print(f"Processing dataset for bias_type: {bias_type}...")
    rng = np.random.default_rng(42)

    def prepare_split(split_name):
        print(f"Collecting examples from '{split_name}' split...")
        # Process examples one by one to handle filtering correctly
        contexts = []
        labels = []
        
        split_data = dataset[split_name]
        
        # Debug: check first few examples to understand structure
        print(f"DEBUG: Checking first 3 examples from '{split_name}' split...")
        sample_count = 0
        
        # Process all examples (and debug first 3)
        for ex in split_data:
            # Debug first 3 examples
            if sample_count < 3:
                print(f"DEBUG: Example {sample_count} keys: {list(ex.keys())}")
                print(f"DEBUG: Example {sample_count} bias_type: {ex.get('bias_type', 'NOT_FOUND')}")
                print(f"DEBUG: Example {sample_count} context: {ex.get('context', 'NOT_FOUND')}")
                print(f"DEBUG: Example {sample_count} target: {ex.get('target', 'NOT_FOUND')}")
                print(f"DEBUG: Example {sample_count} sentences: {ex.get('sentences', 'NOT_FOUND')}")
                sample_count += 1
            
            # Get bias_type of this example
            ex_bias_type = ex.get("bias_type")
            
            # Get sentences dict
            sentences_dict = ex.get("sentences", {})
            if not sentences_dict or not isinstance(sentences_dict, dict):
                continue
            
            # Get sentence list and gold_label list
            sentence_list = sentences_dict.get("sentence", [])
            gold_labels = sentences_dict.get("gold_label", [])
            
            if not sentence_list:
                continue
            
            # Process each sentence
            for i, sent in enumerate(sentence_list):
                if not sent or not str(sent).strip():
                    continue
                
                context_str = str(sent).strip()
                
                # Get gold_label for this sentence (0=antistereotype, 1=stereotype, 2=unrelated)
                if i < len(gold_labels):
                    gold_label = gold_labels[i]
                else:
                    continue  # Skip if no label
                
                # For race bias_type: use gold_label directly (0=antistereotype, 1=stereotype)
                # For other bias_types: check if bias_type matches
                if bias_type == 'race':
                    # If this example's bias_type is 'race', use gold_label
                    # gold_label: 0=antistereotype, 1=stereotype, 2=unrelated
                    if ex_bias_type == 'race':
                        if gold_label == 1:  # stereotype
                            labels.append(1)
                            contexts.append(context_str)
                        elif gold_label == 0:  # antistereotype
                            labels.append(0)
                            contexts.append(context_str)
                        # Skip unrelated (gold_label == 2)
                    else:
                        # If bias_type is not 'race', label as 0
                        labels.append(0)
                        contexts.append(context_str)
                else:
                    # For other bias_types (gender, profession, etc.)
                    if ex_bias_type == bias_type:
                        if gold_label == 1:  # stereotype
                            labels.append(1)
                            contexts.append(context_str)
                        elif gold_label == 0:  # antistereotype
                            labels.append(0)
                            contexts.append(context_str)
                        # Skip unrelated (gold_label == 2)
                    else:
                        # If bias_type doesn't match, label as 0
                        labels.append(0)
                        contexts.append(context_str)
        
        data = {'contexts': contexts, 'labels': labels}
        data = shuffle_split_data(data, rng)
        print(f"Collected {len(data['contexts'])} usable examples from '{split_name}'.")
        return data

    def log_distribution(name, labels):
        if not labels:
            print(f"{name} split has no examples after filtering.")
            return
        counts = Counter(labels)
        print(f"{name} label distribution: {dict(counts)}")

    # Process available splits (stereoset might not have train/validation/test splits)
    available_splits = list(dataset.keys())
    print(f"Processing splits: {available_splits}")
    
    # Get the first split as train
    if len(available_splits) > 0:
        train_split = available_splits[0]
        train_data = prepare_split(train_split)
    else:
        print("No splits available!")
        return
    
    # Get the second split as dev (or use train if only one split)
    if len(available_splits) > 1:
        dev_split = available_splits[1]
        dev_data = prepare_split(dev_split)
    else:
        # If only one split, split it manually
        print("Only one split available, splitting manually...")
        total = len(train_data['contexts'])
        split_idx = int(total * 0.8)
        dev_data = {
            'contexts': train_data['contexts'][split_idx:],
            'labels': train_data['labels'][split_idx:]
        }
        train_data = {
            'contexts': train_data['contexts'][:split_idx],
            'labels': train_data['labels'][:split_idx]
        }
        dev_data = shuffle_split_data(dev_data, rng)
        train_data = shuffle_split_data(train_data, rng)
    
    # Get the third split as test (or use dev if only two splits)
    if len(available_splits) > 2:
        test_split = available_splits[2]
        test_data = prepare_split(test_split)
    else:
        # If only two splits, use part of dev as test
        print("Only two splits available, using part of dev as test...")
        total = len(dev_data['contexts'])
        split_idx = int(total * 0.5)
        test_data = {
            'contexts': dev_data['contexts'][split_idx:],
            'labels': dev_data['labels'][split_idx:]
        }
        dev_data = {
            'contexts': dev_data['contexts'][:split_idx],
            'labels': dev_data['labels'][:split_idx]
        }
        test_data = shuffle_split_data(test_data, rng)
        dev_data = shuffle_split_data(dev_data, rng)
    
    print(f"Train examples: {len(train_data['contexts'])}")
    print(f"Dev examples: {len(dev_data['contexts'])}")
    print(f"Test examples: {len(test_data['contexts'])}")

    log_distribution("Train", train_data["labels"])
    log_distribution("Dev", dev_data["labels"])
    log_distribution("Test", test_data["labels"])

    if len(set(train_data["labels"])) < 2:
        print("Warning: Training data lacks one of the classes (stereotype vs. antistereotype). Exiting.")
        return
    
    if len(train_data['contexts']) == 0:
        print(f"Warning: No examples found for bias_type '{bias_type}'. Please check the dataset structure.")
        return
    
    # Tokenize datasets
    def tokenize_dataset_batch(examples):
        contexts = examples['contexts']
        labels = examples['labels']
        # Ensure we truncate and pad to exactly max_length
        tokenized = tokenizer(
            contexts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )
        # Ensure all sequences are exactly max_length
        for i in range(len(tokenized["input_ids"])):
            if len(tokenized["input_ids"][i]) > max_length:
                tokenized["input_ids"][i] = tokenized["input_ids"][i][:max_length]
                tokenized["attention_mask"][i] = tokenized["attention_mask"][i][:max_length]
            elif len(tokenized["input_ids"][i]) < max_length:
                # Pad to max_length if needed (shouldn't happen with padding="max_length", but safety check)
                pad_length = max_length - len(tokenized["input_ids"][i])
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                tokenized["input_ids"][i] = tokenized["input_ids"][i] + [pad_token_id] * pad_length
                tokenized["attention_mask"][i] = tokenized["attention_mask"][i] + [0] * pad_length
        
        tokenized["label"] = labels
        tokenized["raw_text"] = contexts
        return tokenized
    
    print("Tokenizing datasets...")
    # Filter out empty datasets before tokenizing
    if len(train_data['contexts']) == 0:
        print("Warning: No training data found after filtering. Exiting.")
        return
    
    train_tokenized = Dataset.from_dict(train_data).map(tokenize_dataset_batch, batched=True)
    dev_tokenized = Dataset.from_dict(dev_data).map(tokenize_dataset_batch, batched=True) if len(dev_data['contexts']) > 0 else Dataset.from_dict({"contexts": [], "labels": []})
    test_tokenized = Dataset.from_dict(test_data).map(tokenize_dataset_batch, batched=True) if len(test_data['contexts']) > 0 else Dataset.from_dict({"contexts": [], "labels": []})
    
    # Filter out examples with empty tokenized data and ensure max_length constraint
    def filter_and_validate(examples):
        # Keep only examples where input_ids is not empty and length <= max_length
        filtered = {"input_ids": [], "attention_mask": [], "label": [], "raw_text": []}
        input_ids_list = examples.get("input_ids", [])
        attention_mask_list = examples.get("attention_mask", [])
        labels_list = examples.get("label", [])
        raw_text_list = examples.get("raw_text", [])
        
        for i in range(len(input_ids_list)):
            input_ids = input_ids_list[i]
            # Check if input_ids is valid and length is within max_length
            if input_ids and len(input_ids) > 0 and len(input_ids) <= max_length:
                # Truncate if necessary (shouldn't happen due to tokenizer, but safety check)
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    attention_mask = attention_mask_list[i][:max_length] if i < len(attention_mask_list) else [1] * max_length
                else:
                    attention_mask = attention_mask_list[i] if i < len(attention_mask_list) else [1] * len(input_ids)
                
                filtered["input_ids"].append(input_ids)
                filtered["attention_mask"].append(attention_mask)
                filtered["label"].append(labels_list[i] if i < len(labels_list) else 0)
                filtered["raw_text"].append(raw_text_list[i] if i < len(raw_text_list) else "")
        return filtered
    
    train_tokenized = train_tokenized.map(filter_and_validate, batched=True, remove_columns=train_tokenized.column_names)
    if len(dev_tokenized) > 0:
        dev_tokenized = dev_tokenized.map(filter_and_validate, batched=True, remove_columns=dev_tokenized.column_names)
    if len(test_tokenized) > 0:
        test_tokenized = test_tokenized.map(filter_and_validate, batched=True, remove_columns=test_tokenized.column_names)
    
    # Use tokenized datasets directly
    train_dataset = train_tokenized
    dev_dataset = dev_tokenized
    test_dataset = test_tokenized
    
    # Check if we have valid data
    if len(train_dataset) == 0:
        print("Warning: No valid training data after tokenization. Exiting.")
        return
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
    if len(dev_dataset) > 0:
        dev_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
    if len(test_dataset) > 0:
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab_size: {vocab_size}")
    print(f"Model max_length: {max_length}")
    
    # Check for any token IDs that exceed vocab_size in the dataset
    print("Checking for invalid token IDs in dataset...")
    max_token_id = 0
    for batch in train_loader:
        if "input_ids" in batch and batch["input_ids"] is not None:
            batch_max = batch["input_ids"].max().item()
            max_token_id = max(max_token_id, batch_max)
    print(f"Maximum token ID found in dataset: {max_token_id}")
    if max_token_id >= vocab_size:
        print(f"WARNING: Found token IDs >= vocab_size ({vocab_size}). These will be clamped.")
    
    model = SimpleTransformerClassifier(vocab_size=vocab_size, max_length=max_length)
    model.to(device)
    print(f"Model vocab_size: {model.token_embedding.num_embeddings}")
    
    print("Training model...")
    model = train_model(model, train_loader, dev_loader, device, num_epochs=num_train_epochs)
    
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_dir, "pytorch_model.bin"))
    print(f"Model saved to {model_save_dir}")
    
    print("Computing token importances...")
    importance_dict = defaultdict(float)
    
    model.eval()
    for split_name, dataset_split in [("train", train_dataset), ("dev", dev_dataset)]:
        print(f"Processing {split_name} set...")
        sample_size = min(1000, len(dataset_split))
        processed_count = 0
        for i in range(sample_size):
            try:
                example = dataset_split[i]
                text = example["raw_text"]
                if not text or not isinstance(text, str):
                    continue
                label = example["label"].item()
                
                target_class = int(label)
                tokens, attributions = process_text(text, target_class, tokenizer, model, device)
                
                # Move attributions to CPU for processing
                if isinstance(attributions, torch.Tensor):
                    attributions = attributions.cpu()
                
                for token, attr in zip(tokens, attributions):
                    attr_value = attr.item() if isinstance(attr, torch.Tensor) else float(attr)
                    importance_dict[token] += attr_value
                processed_count += 1
            except Exception as e:
                print(f"Warning: Failed to process example {i} in {split_name} set: {e}")
                continue
        print(f"Processed {processed_count}/{sample_size} examples from {split_name} set")
    
    # Normalize importances
    total = sum(importance_dict.values())
    if total > 0:
        importance_dict = {k: v/total for k, v in importance_dict.items()}
    
    with open(imp_save_file, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"Importance dictionary saved to {imp_save_file}")

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer based on model type
    if args.model == 'llava':
        tokenizer_name = "llava-hf/llava-1.5-7b-hf"
        print(f"Loading LLaVA tokenizer from {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif args.model == 'paligemma':
        from transformers import PaliGemmaProcessor
        model_id = "google/paligemma-3b-mix-224"
        print(f"Loading PaliGemma processor from {model_id}...")
        processor = PaliGemmaProcessor.from_pretrained(model_id)
        tokenizer = processor.tokenizer
    elif args.model == 'qwen':
        tokenizer_name = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"Loading Qwen tokenizer from {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        raise ValueError(f"Unsupported model: {args.model}. Choose from ['llava', 'paligemma', 'qwen']")
    
    # Determine model save paths based on bias_type
    if args.bias_type == 'gender':
        model_save_dir = f"gender_model_{args.model}_pytorch_generated"
        imp_save_file = f"importance_dict_{args.model}_pytorch_generated.json"
    elif args.bias_type == 'race':
        model_save_dir = f"race_model_{args.model}_pytorch_generated"
        imp_save_file = f"importance_race_dict_{args.model}_pytorch_generated.json"
    else:
        model_save_dir = f"{args.bias_type}_model_{args.model}_pytorch_generated"
        imp_save_file = f"importance_{args.bias_type}_dict_{args.model}_pytorch_generated.json"
    
    run_experiment_pytorch(
        experiment_name=f"{args.model.upper()} {args.bias_type.capitalize()} Classification (Stereoset)",
        tokenizer=tokenizer,
        model_save_dir=model_save_dir,
        imp_save_file=imp_save_file,
        bias_type=args.bias_type,
        num_train_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )


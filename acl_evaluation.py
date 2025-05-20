import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from sklearn.preprocessing import LabelEncoder
from time import time
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from scipy.stats import entropy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer safely
model = AutoModelForCausalLM.from_pretrained('model name').to(device)

def convert_labels_to_integers(labels):
    if isinstance(labels[0], str):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
    return labels

def calculate_intrinsic_metrics(features):
    total_tokens = 0
    total_words = 0
    unique_tokens = set()
    continued_words = 0
    token_lengths = []

    for sentence in features:
        tokens = str(sentence).split()
        total_words += len(tokens)
        total_tokens += sum(len(token) for token in tokens)
        token_lengths.extend([len(token) for token in tokens])
        unique_tokens.update(tokens)
        for i in range(1, len(tokens)):
            if tokens[i].startswith(tokens[i - 1]):
                continued_words += 1

    fertility = total_tokens / total_words if total_words > 0 else 0
    parity = min(total_words, total_tokens) / max(total_words, total_tokens) if max(total_words, total_tokens) > 0 else 0
    oov_rate = 1.0 - (len(unique_tokens) / total_tokens) if total_tokens > 0 else 0
    compression = total_tokens / sum(len(str(s)) for s in features) if features.size > 0 else 0
    lexical_overlap = len(unique_tokens) / total_words if total_words > 0 else 0
    token_dist = Counter(token_lengths)
    token_dist_probs = np.array(list(token_dist.values())) / sum(token_dist.values())
    uniform_dist = np.ones_like(token_dist_probs) / len(token_dist_probs)
    token_distribution_divergence = entropy(token_dist_probs, uniform_dist)

    return {
        'fertility': fertility,
        'parity': parity,
        'OOV_rate': oov_rate,
        'compression': compression,
        'lexical_overlap': lexical_overlap,
        'token_distribution_divergence': token_distribution_divergence,
    }

def calculate_extrinsic_metrics(model, tokenizer, tokenized_inputs, batch_size=8):
    start_time = time()
    total_tokens = 0
    total_words = 0
    total_memory = 0

    dataset = TensorDataset(
        tokenized_inputs['input_ids'],
        tokenized_inputs['attention_mask']
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256
            )

            total_tokens += torch.sum(attention_mask).item()
            decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            total_words += sum(len(sent.split()) for sent in decoded)
            total_memory += torch.cuda.max_memory_allocated(device)

    elapsed = time() - start_time
    return {
        'inference_speed': total_tokens / elapsed if elapsed > 0 else 0,
        'energy_consumption': total_memory / 1e6,
        'cost_per_token': total_memory / total_tokens if total_tokens > 0 else 0,
        'cost_per_word': total_memory / total_words if total_words > 0 else 0
    }

base_path = 'dataset path'
datasets = ["language categories"]
results = {}

for dataset in datasets:
    print(f"Processing dataset: {dataset}")
    dataset_results = {'samples_processed': 0, 'intrinsic_metrics': [], 'extrinsic_metrics': []}

    for split in ['train', 'dev', 'test']:
        features_file = os.path.join(base_path, dataset, f"{split}_features.npy")
        labels_file = os.path.join(base_path, dataset, f"{split}_labels.npy")

        if os.path.exists(features_file) and os.path.exists(labels_file):
            features = np.load(features_file, allow_pickle=True)
            labels = np.load(labels_file, allow_pickle=True)

            if features.size == 0 or labels.size == 0:
                print(f"  {split} is empty. Skipping.")
                continue

            labels = convert_labels_to_integers(labels)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            tokenized_inputs = tokenizer(
                ["summarize: " + str(feature) for feature in features],
                padding=True, truncation=True, max_length=256, return_tensors='pt'
            )
            tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

            intrinsic = calculate_intrinsic_metrics(features)
            extrinsic = calculate_extrinsic_metrics(model, tokenizer, tokenized_inputs)

            dataset_results['intrinsic_metrics'].append(intrinsic)
            dataset_results['extrinsic_metrics'].append(extrinsic)
            dataset_results['samples_processed'] += len(features)

        else:
            print(f"  {split} files not found.")

    results[dataset] = dataset_results

for dataset, dataset_results in results.items():
    print(f"\nResults for {dataset}:")
    print(f"  Total samples processed: {dataset_results['samples_processed']}")

    print("\n  Intrinsic Metrics:")
    for split, metrics in zip(['train', 'dev', 'test'], dataset_results['intrinsic_metrics']):
        print(f"    {split}: {metrics}")

    print("\n  Extrinsic Metrics:")
    for split, metrics in zip(['train', 'dev', 'test'], dataset_results['extrinsic_metrics']):
        print(f"    {split}: {metrics}")

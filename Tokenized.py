import os
import pandas as pd
import numpy as np
from subword_nmt import learn_bpe, apply_bpe
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shutil

# Paths for storing tokenized data
TOKENIZED_DATA_PATH = 'BPE_tokenized_data_numpy'

# List of relevant languages and tasks
LANGUAGES = ['Armenian', 'Belarusian', 'Bulgarian', 'Hebrew', 'Persian', 'Ukrainian']
TASKS = ['case_adj', 'case_noun', 'gender_adj', 'gender_noun', 'gender_verb', 
         'number_adj', 'number_noun', 'number_verb', 'tense_verb']

BASE_PATH = 'morphology-probes/data'

# Dictionary to store file paths for each language
language_files = {lang: [] for lang in LANGUAGES}
for task in TASKS:
    for lang in LANGUAGES:
        file_path = os.path.join(BASE_PATH, task, lang)
        if os.path.exists(file_path):
            for file_name in os.listdir(file_path):
                if file_name.endswith('.tsv'):
                    language_files[lang].append(os.path.join(file_path, file_name))

# Function to clean BPE codes file (ensure it has exactly two subword units per line)
def clean_bpe_codes_file(bpe_codes_file):
    cleaned_lines = []
    with open(bpe_codes_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:  # Only keep lines with exactly two subword units
                cleaned_lines.append(line)
            else:
                print(f"Skipping invalid line: {line.strip()}")  # Log the invalid line
    with open(bpe_codes_file, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)
    print(f"BPE codes file cleaned: {bpe_codes_file}")

# Function to apply BPE tokenization without fixed padding
def apply_bpe_tokenization(input_texts, bpe_model, vocab):
    tokenized_texts = [bpe_model.process_line(text).split() for text in input_texts]
    tokenized_indices = [[vocab.get(token, vocab['<unk>']) for token in text] for text in tokenized_texts]
    return tokenized_indices  # Variable-length sequences

# Function to process dataset and apply tokenization
def process_and_tokenize_data(file_paths, bpe_model, vocab):
    features, labels = [], []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', quotechar='"', escapechar='\\')
        except pd.errors.ParserError as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        
        features.extend(df.iloc[:, 0].tolist())  # Assuming first column is features
        labels.extend(df.iloc[:, 1].tolist())  # Assuming second column is labels
    
    # Tokenization without padding/truncation
    tokenized_features = apply_bpe_tokenization(features, bpe_model, vocab)
    return tokenized_features, np.array(labels)

# Function to build vocabulary from training text
def build_vocabulary(tokenized_texts, vocab_size=5000):
    unique_tokens = set(token for text in tokenized_texts for token in text)
    most_common_tokens = list(unique_tokens)[:vocab_size]
    vocab = {token: i+1 for i, token in enumerate(most_common_tokens)}  # 1-based index
    vocab['<unk>'] = len(vocab) + 1  # Unknown token index
    return vocab

# Function to learn BPE safely and handle errors in the process
def safe_learn_bpe(input_texts, num_symbols, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as out_file:
            learn_bpe.learn_bpe(input_texts, num_symbols=num_symbols, outfile=out_file)
    except Exception as e:
        print(f"Error during BPE learning: {e}")
        return None
    return output_file

# Iterate through languages
for lang, file_paths in language_files.items():
    train_file_paths = [file for file in file_paths if 'train.tsv' in file]
    all_train_texts = []
    
    for train_file in train_file_paths:
        try:
            df = pd.read_csv(train_file, sep='\t')
            all_train_texts.extend(df.iloc[:, 0].tolist())
        except pd.errors.ParserError as e:
            print(f"Error reading file {train_file}: {e}")
            continue
    
    # Learn BPE on training data safely
    bpe_model_path = f'bpe_model_{lang}.codes'
    safe_learn_bpe(all_train_texts, num_symbols=5000, output_file=bpe_model_path)
    
    # Clean the BPE codes file to ensure proper format
    clean_bpe_codes_file(bpe_model_path)
    
    # Load BPE model
    bpe_model = apply_bpe.BPE(open(bpe_model_path, 'r'))
    
    # Build vocabulary from tokenized training texts
    tokenized_train_texts = [bpe_model.process_line(text).split() for text in all_train_texts]
    vocabulary = build_vocabulary(tokenized_train_texts)
    
    # Calculate the max sequence length dynamically
    max_sequence_length = max(len(text) for text in tokenized_train_texts)
    print(f"Max sequence length for {lang} (train data): {max_sequence_length}")
    
    # Process train, dev, and test sets
    lang_data = {}
    for split in ['train', 'dev', 'test']:
        split_file_paths = [file for file in file_paths if split in file]
        tokenized_features, tokenized_labels = process_and_tokenize_data(split_file_paths, bpe_model, vocabulary)
        
        # Pad sequences to the dynamic maximum length
        padded_features = pad_sequences(tokenized_features, padding='post', truncating='post', maxlen=max_sequence_length)
        lang_data[split] = {'features': padded_features, 'labels': tokenized_labels}
        
        # Print shape of data for each split
        print(f"Shape of {split} data for {lang}: Features: {lang_data[split]['features'].shape}, Labels: {lang_data[split]['labels'].shape}")
    
    # Save tokenized data
    lang_output_path = os.path.join(TOKENIZED_DATA_PATH, lang)
    os.makedirs(lang_output_path, exist_ok=True)
    
    for split, data in lang_data.items():
        try:
            np.save(os.path.join(lang_output_path, f'{split}_features.npy'), data['features'], allow_pickle=True)
            np.save(os.path.join(lang_output_path, f'{split}_labels.npy'), data['labels'])
            print(f"Saved {split} features for {lang} with dynamic padded sequences.")
        except Exception as e:
            print(f"Error saving {split} features for {lang}: {e}")
    
    print(f"Tokenization complete for {lang}")
    
    # Zip the folder and create a download link
    zip_file_path = f'{TOKENIZED_DATA_PATH}_{lang}.zip'
    shutil.make_archive(zip_file_path.replace('.zip', ''), 'zip', lang_output_path)
    print(f"Created a zip archive: {zip_file_path}")

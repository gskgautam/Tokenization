import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shutil

# Paths for storing tokenized data
TOKENIZED_DATA_PATH = 'Path'

# List of relevant languages and tasks
LANGUAGES = ['Armenian', 'Belarusian', 'Bulgarian', 'Hebrew', 'Persian', 'Ukrainian']
TASKS = ['case_adj', 'case_noun', 'gender_adj', 'gender_noun', 'gender_verb', 
         'number_adj', 'number_noun', 'number_verb', 'tense_verb']

BASE_PATH = 'Path'

# Dictionary to store file paths for each language
language_files = {lang: [] for lang in LANGUAGES}
for task in TASKS:
    for lang in LANGUAGES:
        file_path = os.path.join(BASE_PATH, task, lang)
        if os.path.exists(file_path):
            for file_name in os.listdir(file_path):
                if file_name.endswith('.tsv'):
                    language_files[lang].append(os.path.join(file_path, file_name))

# Function to apply character-level tokenization
def apply_character_tokenization(input_texts):
    # Split input texts into individual characters
    tokenized_texts = [list(text) for text in input_texts]
    return tokenized_texts  # Variable-length sequences

# Function to process dataset and apply tokenization
def process_and_tokenize_data(file_paths):
    features, labels = [], []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip', quotechar='"', escapechar='\\')
        except pd.errors.ParserError as e:
            print(f"Error reading file {file_path}: {e}")
            continue
        
        features.extend(df.iloc[:, 0].tolist())  # Assuming first column is features
        labels.extend(df.iloc[:, 1].tolist())  # Assuming second column is labels
    
    # Character-level tokenization
    tokenized_features = apply_character_tokenization(features)
    return tokenized_features, np.array(labels)

# Function to build vocabulary from tokenized texts
def build_vocabulary(tokenized_texts, vocab_size=5000):
    # Flatten tokenized texts and get the most common characters
    all_chars = [char for text in tokenized_texts for char in text]
    unique_chars = list(set(all_chars))
    most_common_chars = unique_chars[:vocab_size]
    vocab = {char: i+1 for i, char in enumerate(most_common_chars)}  # 1-based index
    vocab['<unk>'] = len(vocab) + 1  # Unknown token index
    return vocab

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
    
    # Character-level tokenization
    tokenized_train_texts = apply_character_tokenization(all_train_texts)
    
    # Build vocabulary from tokenized training texts
    vocabulary = build_vocabulary(tokenized_train_texts)
    
    # Calculate the max sequence length dynamically
    max_sequence_length = max(len(text) for text in tokenized_train_texts)
    print(f"Max sequence length for {lang} (train data): {max_sequence_length}")
    
    # Process train, dev, and test sets
    lang_data = {}
    for split in ['train', 'dev', 'test']:
        split_file_paths = [file for file in file_paths if split in file]
        tokenized_features, tokenized_labels = process_and_tokenize_data(split_file_paths)
        
        # Convert characters to indices using vocabulary
        tokenized_features_indices = [[vocabulary.get(char, vocabulary['<unk>']) for char in text] for text in tokenized_features]
        
        # Pad sequences to the dynamic maximum length
        padded_features = pad_sequences(tokenized_features_indices, padding='post', truncating='post', maxlen=max_sequence_length)
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

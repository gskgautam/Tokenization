# Tokenization and Morphological Evaluation for Low-Resource and Non-Latin Languages

This repository supports experiments on tokenization strategies and model performance for several non-Latin, low-resource, and morphologically rich languages.

## 🌍 Languages and Datasets

We use datasets for the following languages:

- **Hebrew** and **Persian** — Non-Latin scripts
- **Armenian** and **Belarusian** — Low-resource
- **Bulgarian** and **Ukrainian** — Both non-Latin and low-resource

All datasets are publicly available at:  
🔗 [https://github.com/juditacs/morphology-probes](https://github.com/juditacs/morphology-probes)

## 🔤 Tokenization Strategies

This repository supports multiple tokenization methods:

- **BPE (Byte Pair Encoding)** — `BPE_Tokenized.py`
- **Character-level Tokenization** — `Character_Tokenized.py`
- **SAGE-based Tokenization** — `SAGE_Tokenized.py`

Each script processes the dataset and saves the resulting token arrays for later use.

## 🧠 Language Models Used

We evaluate the tokenized data using the following large language models:

- [LLaMA-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)

## 📊 Evaluation

To assess tokenization and model behavior, we support:

- **Intrinsic Metrics** 
- **Extrinsic Metrics** 

Use the provided `Evaluation.py` script to compute all metrics once the token arrays have been generated.

🖥️ Note on Performance Variability:
Evaluation results may vary by up to ±10% depending on your hardware configuration, especially GPU type, memory bandwidth, and compute environment. This margin reflects differences in numerical precision, runtime optimizations, and stability of training dynamics during task vector extraction and calibration.

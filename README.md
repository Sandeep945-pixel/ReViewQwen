# ReViewQwen: Explainable VLM for E-Commerce Discrepancy Detection

**Published at:** IEEE CBMI 2025

## What It Does
Detects discrepancies between seller claims and buyer experiences on e-commerce platforms by comparing product images, descriptions, and reviews using a Vision-Language Model.

## Approach
- Fine-tuned **Qwen2-VL** using **LoRA (rank 16)** on a curated dataset of 2,300 examples
- Compares seller images + descriptions against buyer reviews + images
- Determines who is misleading — the seller or the buyer

## Key Results
- **88% accuracy** (up from 51.15% baseline)
- **78.86% F1-score**
- Outperformed LLaMA 3.2, Phi-3.5, and PaLiGemma 2

## Tech Stack
Python, PyTorch, HuggingFace Transformers, PEFT/LoRA, Qwen2-VL

## Repo Structure
- `qwen/` — Qwen2-VL fine-tuning and inference
- `llama/` — LLaMA 3.2 baseline comparison
- `phi/` — Phi-3.5 baseline comparison
- `paligemma/` — PaLiGemma 2 baseline comparison
- `prompting/` — Prompt engineering experiments
- `data/` — Dataset curation scripts
- `accuracy.py` — Evaluation metrics

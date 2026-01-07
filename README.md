# Sahayok-LLM: Empathetic Bengali Mental Health Counselor

A fine-tuned Llama 3.1 8B model specialized in providing empathetic, culturally nuanced mental health support in Bengali. This project utilizes PEFT (LoRA) and 4-bit quantization to achieve high-performance conversational AI.

[![Model: Llama-3.1-8B-Instruct](https://img.shields.io/badge/Model-Llama--3.1--8B--Instruct-blue)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
[![Fine-tuning: Unsloth/PEFT](https://img.shields.io/badge/Fine--tuning-Unsloth%2FPEFT-orange)](https://github.com/unslothai/unsloth)

---

##  Technical Architecture

### 1. Choice of LoRA & Unsloth
For this solution, I implemented a strategy pattern supporting both standard **PEFT/LoRA** and **Unsloth**.
- **Unsloth Optimization**: Utilized to achieve faster training speeds and significant VRAM savings, allowing for a 2048 sequence length on a single T4 GPU.
- **LoRA Parameters**:
  - **Rank (r)**: 16 (Optimal for learning Bengali emotional nuances without high parameter overhead).
  - **Alpha**: 32 (Standard scaling factor).
  - **Target Modules**: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) to maximize the adaptation of the model's self-attention and MLP layers to Bengali.
  - **Quantization**: 4-bit NormalFloat (NF4) for efficient memory management.

### 2. Training Strategy
- **Prompt Template**: Llama 3.1 Instruct (Header-based engineering).
- **Learning Rate**: 2e-4.
- **Epochs**: 1
- **Batch Size** : 4 per device (effective 8 with gradient accumulation).
- **Optimizer**: AdamW 8-bit (lr=2e-4, weight decay=0.01).
- **Hardware**: Kaggle Dual T4 GPUs.
- **Sequence Length**: 2048 tokens (Max sequence length of the dataset is 2048).

### 3. Dataset
- **Source**: [Bengali Empathetic Conversations Corpus](https://www.kaggle.com/datasets/raseluddin/bengali-empathetic-conversations-corpus)
- **Preprocessing**: Formatted with Llama 3.1 special tokens (<|begin_of_text|>, <|eot_id|>, etc.) for system/user/assistant roles. Filtered sequences to ≤2048 tokens to fit GPU memory.
- **Size**: 20,000 entries. (Train/Val/Test split: 90%/9%/1%)
- **Key Feature**: Focuses on mental health topics like depression, anxiety, and loneliness, with empathetic Bengali responses.

---

##  Evaluation & Results

### Metrics Summary

| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Perplexity** | 1.8306 | **Outstanding**: A perplexity below 2.0 indicates the model is highly "certain" about its Bengali word choices. It has effectively internalized the grammar and vocabulary of the corpus. |
| **BLEU** | 0.2276 |Strong performance for a creative task. In empathetic dialogue, word-for-word matching (BLEU) is often low; 0.22+ shows significant lexical alignment with professional counsel.  |
| **ROUGE-L** | 0.3598 | This reflects high structural similarity. The model is capturing the "flow" of a counseling session (acknowledgment -> empathy -> advice) consistently.  |
| **Train Loss** | 0.4512| Indicates strong convergence. The model successfully minimized the cross-entropy loss between its predictions and the counselor's responses. |
| **Val Loss** | 0.3876 |  Since validation loss is lower than training loss, it suggests the model generalizes well and is not overfitting to the training samples. |
| **Human Evaluation** | 7.5 | Most of the response was clear and helpful. check [Generated Samples](https://github.com/Mazharul-Khan/-Llama-3.1-8B-Fine-tune-on-Bangla-empathy-dataset/blob/main/Results/gen_samp.xlsx) for more details |

### Sample Model Responses
| User Question | Ground Truth | Model Response |
| :--- | :--- | :--- |
|  বাচ্চারা চলে গেছে, আমার বাচ্চারা চলে যাওয়ার সময় আমি একা ছিলাম| ওহ না. বাচ্চারা কোথায় গেল? | আমি খুব দুঃখিত. আমি অনুমান করি এটি কঠিন হবে, কিন্তু আপনি অন্তত এই সময়ে আপনার স্বাধীনতা উপভোগ করতে পারেন! |

---

##  Challenges Faced & Solutions

1. **The "English Drift" Challenge**: Initially, the model frequently defaulted to English.
   - **Solution**: Implemented strict Llama 3.1 `<|start_header_id|>` tags in the preprocessor and increased `repetition_penalty` to 1.1 to anchor the model in the Bengali token space.
   
2. **Incomplete Sentence Cut-offs**: Responses were ending abruptly mid-thought.
   - **Solution**: Engineered a custom logic in the `Evaluator` class to check for Bengali full stops (`।`) and ensured `max_new_tokens` was dynamically calculated based on the dataset's average answer length.

3. **Multi-GPU Device Mismatch**: Faced `RuntimeError` due to tensors being created on `cuda:0` while the model was on `cuda:1`.
   - **Solution**: Re-wrote the inference loop to explicitly identify the primary model device (`next(model.parameters()).device`) and mapped all input/label tensors to it.
4. **Kaggle GPU VRAM & Session Time Limits**: Training an 8B model on Tesla T4 GPUs (16GB VRAM) required heavy memory optimization. Enabling **gradient checkpointing** (via Unsloth's `"unsloth"` mode) was essential to fit the model and long sequences (up to 2048 tokens), but it significantly slowed down training. Additionally, Kaggle enforces a **~12-hour session runtime limit** and a weekly GPU quota (~30–40 hours), making multi-epoch or full-dataset runs risky or impossible in a single session.  
   - **Solution**: Limited training to **20,000 sampled examples** (from the original larger corpus), used **1 epoch** only, and relied on Unsloth's highly efficient 4-bit LoRA implementation. This allowed the entire training + evaluation + Testing to complete comfortably within one Kaggle session without timeout.
---

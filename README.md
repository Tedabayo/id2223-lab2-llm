 **Live demo on Hugging Face Spaces**  
 https://huggingface.co/spaces/teeda-ml/llama3-lab2-finetuned-ui
 


# ID2223 – Lab 2: PEFT Fine-Tuning of a Llama 3 Model

This repo contains my solution for **Lab 2** of ID2223 (HT2025): parameter-efficient fine-tuning (PEFT) of a Llama-based large language model on the **FineTome** instruction dataset, and deployment of a CPU-only inference UI using **Gradio** on **Hugging Face Spaces**.

---

## 1. Project Structure

```text
.
├── app.py                # Gradio UI for inference (Hugging Face Space)
├── train_finetune.ipynb  # Colab notebook used to fine-tune the model
├── requirements.txt      # Python dependencies
└── README.md             # This file (Task 2 description + usage)

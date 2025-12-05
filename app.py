import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ----------------------------------------------------
# Load your fine-tuned model
# ----------------------------------------------------
MODEL_ID = "teeda-ml/llama3-lab2-finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,   # CPU only
    device_map="cpu",            # Run on CPU in Spaces
)


# ----------------------------------------------------
# Chat function for Gradio ChatInterface
# ----------------------------------------------------
def chat_fn(message, history):
    """
    message: new user message (string)
    history: list of dicts: {"role": "...", "content": "..."}
    """

    # Build text prompt from previous turns
    prompt = ""

    if history:
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

    # Add new user input
    prompt += f"User: {message}\nAssistant:"

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract only assistant reply (new tokens)
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = full_text[len(prompt):].strip()

    return reply


# ----------------------------------------------------
# Gradio app
# ----------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Llama 3 Lab 2 – Fine-tuned",
    description="Chat with my fine-tuned model.",
)

if __name__ == "__main__":
    demo.launch()

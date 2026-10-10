import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/Phi-3-mini-4k-instruct"

print("Loading Phi-3 onto GPU.")

# Load model and tokenizer using native transformers support (trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("model loaded successfully\n")

# Format prompt using Phi-3's official chat template
messages = [
    {"role": "user", "content": "Write a short 2-sentence story about a robot exploring Mars."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate response
outputs = model.generate(
    **inputs, 
    max_new_tokens=100, 
    do_sample=True, 
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id
)

# Decode only the newly generated text
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("--- AI Response ---")
print(response)
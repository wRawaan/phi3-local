import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation="eager"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

generator = pipeline( "text-generation", model=model, tokenizer=tokenizer, return_full_text= False ,max_new_tokens=50, do_sample= False , )

prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids 
input_ids = input_ids.to("cuda")

start_time = time.time()

output_with_cache = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    use_cache=True
)

time_with_cache = time.time() - start_time

start_time = time.time()

output_without_cache = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    use_cache=False
)

time_without_cache = time.time() - start_time

print(f"Time with use_cache=True:  {time_with_cache:.4f} seconds")
print(f"Time with use_cache=False: {time_without_cache:.4f} seconds")

import torch
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

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened." 

output = generator(prompt) 

print(output[0]['generated_text'])
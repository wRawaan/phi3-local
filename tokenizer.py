import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation="eager"
 )
tokenizer = AutoTokenizer.from_pretrained(model_id)
print("model loaded successfully\n")

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate the text
generation_output = model.generate( input_ids=input_ids, max_new_tokens=20 )

# Print the output
print(tokenizer.decode(generation_output[0]))

#Print input_ids
print(input_ids)

for id in input_ids[0]:
  print(tokenizer.decode(id))

print(generation_output)

print(tokenizer.decode(3323)) 
print(tokenizer.decode(622)) 
print(tokenizer.decode([3323, 622])) 
print(tokenizer.decode(29901))




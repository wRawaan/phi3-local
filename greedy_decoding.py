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

prompt = "The capital of France is"
# Tokenize the input prompt 
input_ids = tokenizer(prompt, return_tensors="pt").input_ids 
input_ids = input_ids.to("cuda") 

# Get the output of the model before the lm_head 
model_output = model.model(input_ids) 

# Get the output of the lm_head 
lm_head_output = model.lm_head(model_output[0])

token_id = lm_head_output[0,-1].argmax(-1)

output = tokenizer.decode(token_id)

print(output)

dimension = model_output[0].shape
print(dimension)

LMhead = lm_head_output.shape
print(LMhead)
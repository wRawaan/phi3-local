from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")

print("Model loaded")

# Tokenize the sentence
tokens = tokenizer('I am having fun with LLMs', return_tensors='pt')

# Process the tokens
output = model(**tokens)[0]

# Print dimensions
print(output.shape)

# Decode the generated vectors
for token in tokens['input_ids'][0]:
     print(tokenizer.decode(token))

# Print the generated raw output
print(output)
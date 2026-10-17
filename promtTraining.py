from datasets import load_dataset
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Load our data
print("Loading dataset...")
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# 2. Load model and tokenizer directly for Seq2Seq models
model_id = "google/flan-t5-small"
print(f"Loading model {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to("cuda:0")

# 3. Define prompt template
prompt = "Is the following movie review positive or negative? Answer with 'positive' or 'negative': "

# 4. Run inference loop
y_pred = []
y_true = data["test"]["label"]  # 0 for negative, 1 for positive

print("Running inference...")
for example in tqdm(data["test"], total=len(data["test"])):
  input_text = prompt + example["text"]

  # Tokenize input and move to GPU
  inputs = tokenizer(
      input_text, return_tensors="pt", truncation=True, max_length=512
  ).to("cuda:0")

  # Generate model response
  outputs = model.generate(**inputs, max_new_tokens=10)
  prediction = (
      tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
  )

  # Map string output to binary label (1 for positive, 0 for negative)
  if "positive" in prediction:
    y_pred.append(1)
  else:
    y_pred.append(0)


# 5. Evaluate Performance
def evaluate_performance(y_true, y_pred):
  """Create and print the classification report"""
  performance = classification_report(
      y_true, y_pred, target_names=["Negative Review", "Positive Review"]
  )
  print("\n--- Classification Report ---")
  print(performance)


evaluate_performance(y_true, y_pred)
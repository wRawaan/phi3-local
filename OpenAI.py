from dotenv import load_dotenv
from openai import OpenAI
import os
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm

# 1. Load environment variables from .env file and initialize client
load_dotenv()
client = OpenAI()  # Automatically detects OPENAI_API_KEY from the environment

# 2. Load the dataset
print("Loading dataset...")
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# 3. Define the OpenAI generation function
def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    """Generate an output based on a prompt and an input document."""
    messages = [
        {"role": "system", "content": "You are a helpful text classification assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages, 
        model=model, 
        temperature=0
    )
    return chat_completion.choices[0].message.content.strip()

# 4. Define the prompt template
prompt = "Predict whether the following document is a positive or negative movie review: [DOCUMENT] If it is positive return 1 and if it is negative return 0. Do not give any other answers."

# 5. Run inference loop 
# Use .select() to grab a small batch safely
test_subset = data["test"].select(range(50))
y_true = test_subset["label"]
y_pred = []

print("Running inference with OpenAI...")
for example in tqdm(test_subset, total=len(test_subset)):
  document_text = (
      example["text"]
  )  # Now example remains a dictionary, so this works perfectly!

  # Get prediction from OpenAI
  result = chatgpt_generation(prompt, document_text)

  # Clean and parse the response into an integer (0 or 1)
  try:
    pred_label = int(result)
  except ValueError:
    pred_label = 1 if "1" in result else 0

  y_pred.append(pred_label)

# 6. Evaluate Performance
def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred, 
        target_names=["Negative Review", "Positive Review"]
    )
    print("\n--- Classification Report ---")
    print(performance)

evaluate_performance(y_true, y_pred)
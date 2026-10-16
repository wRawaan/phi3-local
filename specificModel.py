from datasets import load_dataset
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

# Load our data
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model into pipeline (added batch_size=1 via KeyDataset iteration or call)
pipe = pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
    device="cuda:0",
)

# Run inference safely with label lookup
y_pred = []
for output in tqdm(
    pipe(KeyDataset(data["test"], "text"), batch_size=1),
    total=len(data["test"]),
):
  # Handle cases where output might be a list or a nested list/dict
  if isinstance(output, list):
    res_list = output[0] if isinstance(output[0], list) else output
  else:
    res_list = [output]  # fallback if it's a single dict

  # Map scores by their label name to avoid index mismatches
  scores = {item["label"].lower(): item["score"] for item in res_list}

  negative_score = scores.get("negative", 0.0)
  positive_score = scores.get("positive", 0.0)

  assignment = np.argmax([negative_score, positive_score])
  y_pred.append(assignment)


def evaluate_performance(y_true, y_pred):
  """Create and print the classification report"""
  performance = classification_report(
      y_true, y_pred, target_names=["Negative Review", "Positive Review"]
  )
  print(performance)


evaluate_performance(data["test"]["label"], y_pred)
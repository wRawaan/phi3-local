from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np

data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Load model 
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 
# Convert text to embeddings 
train_embeddings = model.encode(data["train"]["text"], 
show_progress_bar= True ) 
test_embeddings = model.encode(data["test"]["text"], 
show_progress_bar= True )

# Create embeddings for our labels
label_embeddings = model.encode(["A negative review",  "A positive review"])

#print(train_embeddings.shape)

clf = LogisticRegression(random_state=42) 
clf.fit(train_embeddings, data["train"]["label"])

# Find the best matching label for each document 
sim_matrix = cosine_similarity(test_embeddings, label_embeddings) 
y_pred = np.argmax(sim_matrix, axis=1)

def evaluate_performance(y_true, y_pred):
  """Create and print the classification report"""
  performance = classification_report(
      y_true, y_pred, target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

evaluate_performance(data["test"]["label"], y_pred)
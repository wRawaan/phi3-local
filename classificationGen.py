from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Load model 
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 
# Convert text to embeddings 
train_embeddings = model.encode(data["train"]["text"], 
show_progress_bar= True ) 
test_embeddings = model.encode(data["test"]["text"], 
show_progress_bar= True )

#print(train_embeddings.shape)

clf = LogisticRegression(random_state=42) 
clf.fit(train_embeddings, data["train"]["label"])

# Predict previously unseen instances 
y_pred = clf.predict(test_embeddings) 

def evaluate_performance(y_true, y_pred):
  """Create and print the classification report"""
  performance = classification_report(
      y_true, y_pred, target_names=["Negative Review", "Positive Review"]
  )
  print(performance)


evaluate_performance(data["test"]["label"], y_pred)

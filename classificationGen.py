from sentence_transformers import SentenceTransformer
from datasets import load_dataset

data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Load model 
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 
# Convert text to embeddings 
train_embeddings = model.encode(data["train"]["text"], 
show_progress_bar= True ) 
test_embeddings = model.encode(data["test"]["text"], 
show_progress_bar= True )

print(train_embeddings.shape)
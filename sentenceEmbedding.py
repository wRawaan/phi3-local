from sentence_transformers import SentenceTransformer 

# Load model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") 

print("model loaded")

# Convert text to text embeddings
vector = model.encode("Best movie ever!")

# Print Dimension
print(vector.shape)
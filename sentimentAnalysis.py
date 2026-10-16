from datasets import load_dataset 
# Load our data 
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")

#print(data)
 
print(data["train"][0, -1])
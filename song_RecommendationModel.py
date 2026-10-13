import pandas as pd 
from urllib import request
from gensim.models import Word2Vec
import numpy as np

# Get the playlist dataset file
data = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt')

# Parse the playlist dataset file. Skip the first two lines as they only contain metadata 
lines = data.read().decode("utf-8").split('\n')[2:]

# Remove playlists with only one song
playlists = [s.rstrip().split() for s in lines if len(s.split()) > 1]

# Load song metadata
songs_file = request.urlopen('https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt')
songs_file = songs_file.read().decode("utf-8").split('\n')
songs = [s.rstrip().split('\t') for s in songs_file]
songs_df = pd.DataFrame(data=songs, columns = ['id', 'title', 'artist'])
songs_df = songs_df.set_index('id')

#print( 'Playlist #1:\n ', playlists[0], '\n')
#print( 'Playlist #2:\n ', playlists[1])

# Train our Word2Vec model
model = Word2Vec( 
    playlists, vector_size=32, window=20, negative=50, 
    min_count=1, workers=4
    )

song_id = 2172
# Ask the model for songs similar to song #2172
output = model.wv.most_similar(positive=str(song_id))
#print("Songs similar to", song_id, "are:",output)

#print(songs_df.iloc[2172])

def print_recommendations(song_id):
     similar_songs = np.array( model.wv.most_similar(positive=str(song_id),topn=5) )[:,0]
     return  songs_df.iloc[similar_songs] 

# Extract and print recommendations 
print(print_recommendations(2172))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset (movie_id, title, genre)
movies = {
    'movie_id': ['A', 'B', 'C', 'D', 'E'],
    'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4', 'Movie 5'],
    'genre': ['Action, Adventure', 'Action, Sci-Fi', 'Adventure, Fantasy', 'Drama, Action', 'Sci-Fi, Drama']
}

# Convert to DataFrame
df_movies = pd.DataFrame(movies)

# Vectorize the movie genres using TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['genre'])

# Compute cosine similarity between movies based on their genres
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on a movie's title
def recommend_movie(title, cosine_sim=cosine_sim):
    idx = df_movies.index[df_movies['title'] == title].tolist()[0]  # Get index of the movie
    sim_scores = list(enumerate(cosine_sim[idx]))  # Get similarity scores for all movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity
    sim_scores = sim_scores[1:4]  # Get top 3 similar movies (exclude the movie itself)
    movie_indices = [i[0] for i in sim_scores]
    return df_movies['title'].iloc[movie_indices]

# Recommend movies similar to 'Movie 1'
recommended_movies = recommend_movie('Movie 1')
print("Movies similar to 'Movie 1':")
print(recommended_movies)

     
Movies similar to 'Movie 1':
2    Movie 3
3    Movie 4
1    Movie 2
Name: title, dtype: object

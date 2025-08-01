import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
movies = pd.read_csv('movies.csv')
movies['genres'] = movies['genres'].fillna('')

# TF-IDF and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def recommend_movies(title, num_recommendations=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Streamlit UI
st.title("🎬 Movie Recommendation System")
movie_list = movies['title'].sort_values().unique()
selected_movie = st.selectbox("Choose a movie to get recommendations", movie_list)

if st.button("Get Recommendations"):
    results = recommend_movies(selected_movie)
    if results:
        st.subheader("Recommended Movies:")
        for movie in results:
            st.write("👉", movie)
    else:
        st.error("Movie not found!")

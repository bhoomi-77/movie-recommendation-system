import streamlit as st
import pandas as pd
import pickle

# Load saved files
with open("movies.pkl", "rb") as file:
    movies = pickle.load(file)

with open("similarity.pkl", "rb") as file:
    similarity_matrix = pickle.load(file)


# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):

    movie_indices = movies[
        movies["Movie_Title"].str.lower() == movie_title.lower()
    ].index

    if len(movie_indices) == 0:
        return None

    movie_index = movie_indices[0]

    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1:num_recommendations + 1
    ]

    recommended_indices = [
        x[0] for x in similarity_scores
    ]

    return movies.iloc[
        recommended_indices
    ][["Movie_Title", "Genre", "Rating"]]


# Streamlit UI
st.title("🎬 Movie Recommendation System")

st.write("Select a movie and get similar movie recommendations.")


# Movie selection
movie_title = st.selectbox(
    "Select a Movie",
    movies["Movie_Title"].tolist()
)


# Number of recommendations
number = st.slider(
    "Number of Recommendations",
    1,
    10,
    5
)


# Recommendation button
if st.button("Recommend Movies"):

    recommendations = recommend_movies(
        movie_title,
        number
    )

    if recommendations is not None:

        st.subheader("Recommended Movies")

        for _, movie in recommendations.iterrows():

            st.write(
                f"### 🎬 {movie['Movie_Title']}"
            )

            st.write(
                f"Genre: {movie['Genre']}"
            )

            st.write(
                f"Rating: ⭐ {movie['Rating']}"
            )

            st.divider()
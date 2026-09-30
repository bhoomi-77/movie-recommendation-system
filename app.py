import os
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import pickle

# Folder where app.py lives (so poster paths work no matter where you run Streamlit from)
BASE_DIR = Path(__file__).parent

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0f19;
        color: white;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main title */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 5px;
        background: linear-gradient(
            90deg,
            #ff4b2b,
            #ff416c,
            #8e44ad
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #a8b0c0;
        font-size: 18px;
        margin-bottom: 40px;
    }

    /* Movie card */
    .movie-card {
        background: #151b2b;
        border-radius: 18px;
        padding: 15px;
        margin-bottom: 25px;
        border: 1px solid #252d40;
        transition: all 0.3s ease;
        height: 100%;
    }

    .movie-card:hover {
        transform: translateY(-6px);
        border: 1px solid #ff416c;
        box-shadow: 0px 10px 30px rgba(255, 65, 108, 0.18);
    }

    .movie-title {
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .movie-info {
        color: #aeb7c7;
        font-size: 14px;
        margin: 5px 0;
    }

    .rating {
        display: inline-block;
        background: #242b3d;
        color: #ffd166;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin-top: 8px;
    }

    .genre {
        display: inline-block;
        background: #252040;
        color: #c7b7ff;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 13px;
        margin-top: 8px;
    }

    /* Poster images */
    .poster-img {
        width: 100%;
        border-radius: 12px;
        display: block;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #101522;
        border-right: 1px solid #252d40;
    }

    section[data-testid="stSidebar"] h1 {
        color: white;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px;
        font-size: 16px;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            90deg,
            #ff416c,
            #ff4b2b
        );
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0px 5px 20px rgba(255, 65, 108, 0.35);
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background-color: #151b2b;
        border-radius: 10px;
        border: 1px solid #30394d;
    }

    /* Slider */
    div[data-testid="stSlider"] {
        padding-top: 10px;
    }

    /* Section heading */
    .section-title {
        font-size: 30px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 20px;
        color: white;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #151b2b,
            #20152c
        );
        padding: 35px;
        border-radius: 25px;
        margin-bottom: 30px;
        border: 1px solid #2b3448;
    }

    .hero h2 {
        font-size: 32px;
        margin-bottom: 10px;
    }

    .hero p {
        color: #aeb7c7;
        font-size: 16px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    with open(BASE_DIR / "movies.pkl", "rb") as file:
        movies = pickle.load(file)

    with open(BASE_DIR / "similarity.pkl", "rb") as file:
        similarity_matrix = pickle.load(file)

    return movies, similarity_matrix


movies, similarity_matrix = load_model()


# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "Movie_Title",
    "Genre",
    "Rating"
]

for column in required_columns:

    if column not in movies.columns:

        st.error(
            f"Required column '{column}' is missing from movies.pkl"
        )

        st.stop()


# --------------------------------------------------
# POSTER FUNCTIONS
# --------------------------------------------------

@st.cache_data
def image_to_data_uri(path):
    """Read a local image file and return it as a base64 data URI
    so it can be used inside an HTML <img> tag."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + encoded


def get_poster(movie):
    """Return something usable as an <img src=...> for this movie."""

    # 1) Local generated poster (poster_path column, e.g. posters/Beyond_the_Stars.png)
    if "poster_path" in movies.columns:

        poster = movie.get("poster_path")

        if pd.notna(poster) and str(poster).strip() != "":
            full_path = BASE_DIR / str(poster)

            if full_path.exists():
                return image_to_data_uri(str(full_path))

    # 2) Poster URL columns (if you ever add real poster links)
    for column in ["Poster_URL", "poster_url", "Poster"]:

        if column in movies.columns:

            poster = movie.get(column)

            if pd.notna(poster) and str(poster).strip() != "":
                return str(poster)

    # 3) Placeholder using the movie title
    title = str(movie.get("Movie_Title", "No Poster")).replace(" ", "+")
    return f"https://placehold.co/300x450/151b2b/ffffff?text={title}"


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_movies(movie_title, num_recommendations=5):

    movie_indices = movies[
        movies["Movie_Title"].astype(str).str.lower()
        == movie_title.lower()
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
    ]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🎬 CineMatch")

    st.markdown(
        "### Your Personal Movie Recommender"
    )

    st.markdown("---")

    st.markdown("### ⚙️ Recommendation Settings")

    number = st.slider(
        "Number of Movies",
        min_value=1,
        max_value=10,
        value=5
    )

    st.markdown("---")

    st.markdown("### 📊 Dataset")

    st.metric(
        "Total Movies",
        len(movies)
    )

    if "Genre" in movies.columns:

        st.metric(
            "Genres",
            movies["Genre"].nunique()
        )

    st.markdown("---")

    st.caption(
        "Powered by Content-Based Recommendation"
    )


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown(
    '<div class="hero">'
    '<h2>🍿 Discover Your Next Favorite Movie</h2>'
    '<p>'
    'Choose a movie you love and CineMatch will find '
    'similar movies based on their features.'
    '</p>'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find movies you will love based on what you already enjoy.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# MOVIE SELECTION
# --------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:

    movie_title = st.selectbox(
        "🔍 Select a Movie",
        movies["Movie_Title"].astype(str).tolist()
    )

with col2:

    st.write("")

    recommend_button = st.button(
        "✨ Recommend"
    )


# --------------------------------------------------
# SELECTED MOVIE PREVIEW
# --------------------------------------------------

selected_movie = movies[
    movies["Movie_Title"].astype(str)
    == movie_title
]

if not selected_movie.empty:

    selected_movie = selected_movie.iloc[0]

    st.markdown(
        '<div class="section-title">'
        '🎥 Selected Movie'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 3])

    with col1:

        st.markdown(
            f'<img class="poster-img" src="{get_poster(selected_movie)}" '
            f'alt="{selected_movie["Movie_Title"]}">',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <h2>{selected_movie["Movie_Title"]}</h2>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <span class="genre">
                🎭 {selected_movie["Genre"]}
            </span>

            <span class="rating">
                ⭐ {selected_movie["Rating"]}
            </span>
            """,
            unsafe_allow_html=True
        )


# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

if recommend_button:

    recommendations = recommend_movies(
        movie_title,
        number
    )

    if recommendations is not None:

        st.markdown(
            '<div class="section-title">'
            '🍿 Recommended For You'
            '</div>',
            unsafe_allow_html=True
        )

        # Create cards
        columns = st.columns(5)

        for i, (_, movie) in enumerate(
            recommendations.iterrows()
        ):

            with columns[i % 5]:

                poster_src = get_poster(movie)

                # Build the card HTML with NO indentation and NO blank lines,
                # otherwise markdown shows it as a code block.
                card_html = (
                    '<div class="movie-card">'
                    f'<img class="poster-img" src="{poster_src}" alt="poster">'
                    f'<div class="movie-title">{movie["Movie_Title"]}</div>'
                    f'<div class="movie-info">🎭 {movie["Genre"]}</div>'
                    f'<div class="rating">⭐ {movie["Rating"]}</div>'
                    '</div>'
                )

                st.markdown(card_html, unsafe_allow_html=True)

    else:

        st.warning(
            "Movie not found."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#737d91;
        padding:20px;
    ">
        🎬 CineMatch Movie Recommendation System
        <br>
        Built with Python, Pandas & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
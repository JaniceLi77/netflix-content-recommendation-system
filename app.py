import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content-Based Recommendation System")

st.write(
    "Enter a Netflix movie or TV show title, and the system will recommend similar titles based on genre, description, cast, director, and country."
)


@st.cache_data
def load_data():
    netflix = pd.read_csv("netflix_titles.csv")

    text_cols = ["director", "cast", "country", "listed_in", "description"]
    for col in text_cols:
        netflix[col] = netflix[col].fillna("")

    netflix["rating"] = netflix["rating"].fillna("Unknown")

    netflix["date_added"] = pd.to_datetime(netflix["date_added"], errors="coerce")
    if netflix["date_added"].notna().sum() > 0:
        most_common_date = netflix["date_added"].mode()[0]
        netflix["date_added"] = netflix["date_added"].fillna(most_common_date)

    netflix["year_added"] = netflix["date_added"].dt.year

    netflix = netflix.drop_duplicates()
    netflix = netflix.reset_index(drop=True)

    netflix["combined_features"] = (
        netflix["listed_in"] + " " +
        netflix["description"] + " " +
        netflix["cast"] + " " +
        netflix["director"] + " " +
        netflix["country"]
    )

    return netflix


@st.cache_resource
def build_model(netflix):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    tfidf_matrix = tfidf.fit_transform(netflix["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    title_indices = pd.Series(
        netflix.index,
        index=netflix["title"].str.lower()
    ).drop_duplicates()

    return cosine_sim, title_indices


def recommend_titles(title, netflix, cosine_sim, title_indices, top_n=10):
    title_lower = title.lower().strip()

    if title_lower not in title_indices:
        possible_titles = netflix[
            netflix["title"].str.lower().str.contains(title_lower, na=False)
        ][["title", "type", "release_year", "listed_in"]].head(10)

        return None, possible_titles

    idx = title_indices[title_lower]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n + 1]

    indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    result = netflix.loc[
        indices,
        ["title", "type", "listed_in", "release_year", "rating", "description"]
    ].copy()

    result["similarity_score"] = scores

    return result, None


netflix = load_data()
cosine_sim, title_indices = build_model(netflix)

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Titles", netflix.shape[0])

with col2:
    st.metric("Movies", netflix[netflix["type"] == "Movie"].shape[0])

with col3:
    st.metric("TV Shows", netflix[netflix["type"] == "TV Show"].shape[0])


st.subheader("Find Similar Netflix Titles")

title_input = st.text_input(
    "Enter a Netflix title:",
    value="Stranger Things"
)

top_n = st.slider(
    "Number of recommendations:",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("Recommend"):
    recommendations, possible_titles = recommend_titles(
        title_input,
        netflix,
        cosine_sim,
        title_indices,
        top_n
    )

    if recommendations is not None:
        st.success(f"Top {top_n} recommendations for '{title_input}'")
        for i, row in recommendations.iterrows():
            st.markdown(f"### 🎬 {row['title']}")
            st.write(f"Type: {row['type']}")
            st.write(f"Year: {row['release_year']}")
            st.write(f"Genre: {row['listed_in']}")
            st.write(f"Score: {round(row['similarity_score'], 3)}")
            st.write(f"Description: {row['description'][:150]}...")
            st.write("---")

    else:
        st.warning("Exact title not found. Did you mean one of these?")
        st.dataframe(possible_titles, use_container_width=True)


st.subheader("Sample Titles in Dataset")

st.dataframe(
    netflix[["title", "type", "release_year", "listed_in"]].head(20),
    use_container_width=True
)
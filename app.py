# use the venv "conda actiavet krish"
# Run with:  streamlit run app.py
import re
from pathlib import Path

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ----------------- Page config -----------------
st.set_page_config(
    page_title="IJPS Article Finder",
    page_icon="🩺",
    layout="centered",
)


# ----------------- Search / NLP logic -----------------

@st.cache_resource
def load_data():
    # Load the CSV
    df = pd.read_csv("ijps_articles.csv")

    # Safely handle missing columns/values
    title = df["title"].fillna("")
    abstract = df["abstract"].fillna("")
    keywords = df.get("keywords", "").fillna("")
    authors = df.get("authors", "").fillna("")

    # Boost title & keywords a bit
    title_boost = (title + " ") * 3
    keyword_boost = (keywords + " ") * 2

    df["text"] = title_boost + abstract + " " + keyword_boost

    # Build TF-IDF with 1–2 word n-grams
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    # Keep cleaned columns
    df["title"] = title
    df["abstract"] = abstract
    df["keywords"] = keywords
    df["authors"] = authors

    return df, vectorizer, tfidf_matrix


def split_into_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sentences if s.strip()]


def make_query_aware_teaser(query: str, abstract: str, max_sentences: int = 3) -> str:
    """
    Choose up to max_sentences from the abstract that are most
    relevant to the query, using TF-IDF over the sentences.
    """
    sentences = split_into_sentences(abstract)
    if not sentences:
        return ""

    sent_vectorizer = TfidfVectorizer(stop_words="english")
    sent_matrix = sent_vectorizer.fit_transform(sentences)

    query_vec = sent_vectorizer.transform([query])
    sims = linear_kernel(query_vec, sent_matrix).flatten()

    if sims.max() == 0:
        chosen_indices = list(range(min(max_sentences, len(sentences))))
    else:
        top_indices = sims.argsort()[::-1][:max_sentences]
        chosen_indices = sorted(top_indices)

    chosen_sentences = [sentences[i] for i in chosen_indices]
    teaser = " ".join(chosen_sentences)

    if len(teaser) > 400:
        teaser = teaser[:400].rsplit(" ", 1)[0] + "..."

    return teaser


def search_articles(query: str, top_k: int = 3):
    df, vectorizer, tfidf_matrix = load_data()
    query_vec = vectorizer.transform([query])
    sims = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = sims.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["score"] = sims[top_indices]
    return results


# ----------------- UI -----------------

# Hero section with cover image + title
col_img, col_title = st.columns([1, 2])

with col_img:
    cover_path = Path("ijps_cover.png")
    if not cover_path.exists():
        cover_path = Path("ijps_cover.jpg")
    if cover_path.exists():
        st.image(str(cover_path), width=260)
    else:
        st.write("")

with col_title:
    st.markdown(
        "<h1 style='margin-bottom:0.2rem;'>IJPS Article Finder</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Prototype search assistant for the Indian Journal of Plastic Surgery")

    st.write(
        "Ask a question about plastic surgery, hair transplant, hair care, "
        "or wound management, and this tool will suggest relevant IJPS articles "
        "with a short teaser and a link to read the full text on Thieme."
    )

st.markdown("---")

# Query input
st.subheader("Ask a question")

default_prompt = ""
query = st.text_input(
    "Type your question here:",
    value=default_prompt,
    placeholder="e.g. complications after hair transplant surgery",
)

# Example queries if box is empty
if not query:
    st.markdown("**Try questions like:**")
    st.markdown(
        """
        - *Complications after hair transplant surgery*  
        - *Hair care recommendations for alopecia patients using minoxidil*  
        - *Newer modalities for treatment of pressure ulcers*  
        - *Effect of caffeine on full-thickness skin graft healing*  
        """
    )

# Results
if query:
    results = search_articles(query, top_k=3)

    if results.empty:
        st.warning("No relevant articles found. Try rephrasing your question.")
    else:
        st.markdown("### Suggested articles")
        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            with st.container():
                st.markdown(f"#### {rank}. {row['title']}")
                meta_line = []

                if isinstance(row.get("authors"), str) and row["authors"].strip():
                    meta_line.append(row["authors"])

                if "score" in row:
                    meta_line.append(f"Match score: {row['score']:.3f}")

                if meta_line:
                    st.caption(" • ".join(meta_line))

                teaser = make_query_aware_teaser(
                    query=query,
                    abstract=str(row["abstract"]),
                    max_sentences=3,
                )
                st.write(teaser)

                if isinstance(row.get("article_url"), str) and row["article_url"]:
                    st.markdown(f"[Read full article on Thieme]({row['article_url']})")

                st.markdown("---")

# Footer / disclaimer
st.markdown(
    """
    <small>
    This prototype is for educational and exploratory use only and does not replace
    clinical judgement. Always consult the full article and current guidelines before
    making treatment decisions.
    </small>
    """,
    unsafe_allow_html=True,
)
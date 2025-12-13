# app.py
# Run with:  streamlit run app.py
# (use your venv / conda env that has streamlit, pandas, sklearn installed)

import re
import base64
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


# ----------------- Helpers -----------------
def img_to_base64(path: str) -> str:
    """Load an image file and return a base64 string for CSS embedding."""
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# this is your OT / theatre image; must sit next to app.py
hero_bg_b64 = img_to_base64("hero_bg.jpg")


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
    Choose up to max_sentences from the abstract that are
    most relevant to the query, using TF-IDF over the sentences.
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


# ----------------- Global CSS (full-page bg + hero + input) -----------------
st.markdown(
    f"""
<style>
/* FULL PAGE BACKGROUND */
.stApp {{
    background-image:
        linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.55)),
        url("data:image/jpg;base64,{hero_bg_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}

/* Center content a bit more and make background transparent so photo shows */
.main .block-container {{
    max-width: 1100px;
    padding-top: 2rem;
    background: transparent;
}}

/* HERO CARD (now just a translucent panel on top of the photo) */
.hero {{
    position: relative;
    width: 100%;
    min-height: 260px;
    margin: 2rem auto 3rem auto;
    border-radius: 2.5rem;
    background: radial-gradient(circle at top left, rgba(255,255,255,0.25), rgba(0,0,0,0.35));
    box-shadow: 0 22px 50px rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
}}

.hero-content {{
    color: white;
    padding: 3.2rem 3.6rem;
    max-width: 540px;
}}

.hero-kicker {{
    text-transform: uppercase;
    letter-spacing: 0.18em;
    font-size: 0.80rem;
    opacity: 0.9;
    margin-bottom: 0.6rem;
}}

.hero-title {{
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}}

.hero-subtitle {{
    font-size: 0.98rem;
    line-height: 1.5;
    opacity: 0.97;
}}

/* Rounded search bar look */
.stTextInput > div > div > input {{
    border-radius: 999px;
    padding: 0.7rem 1.1rem;
    border: 1px solid #dde1e7;
}}

.stTextInput > label {{
    font-weight: 500;
    color: white;
}}

h2, h3, h4, h5, h6, p, li, label, span {{
    color: #f5f5f5 !important;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ----------------- UI -----------------

# HERO
st.markdown(
    """
<div class="hero">
  <div class="hero-content">
    <div class="hero-kicker">Prototype • Indian Journal of Plastic Surgery</div>
    <div class="hero-title">IJPS Article Finder</div>
    <div class="hero-subtitle">
      Ask a question about plastic surgery, hair transplant, hair care, or wound management,
      and this tool will suggest relevant IJPS articles with a short teaser and a link
      to read the full text on Thieme.
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Query section
st.subheader("Ask a question")

query = st.text_input(
    "Type your question here:",
    value="",
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
                meta_bits = []

                if isinstance(row.get("authors"), str) and row["authors"].strip():
                    meta_bits.append(row["authors"])

                if "score" in row:
                    meta_bits.append(f"Match score: {row['score']:.3f}")

                if meta_bits:
                    st.caption(" • ".join(meta_bits))

                teaser = make_query_aware_teaser(
                query=query,
                abstract=str(row["abstract"]),
                max_sentences=3,
            )
            st.write(teaser)

            # Links: Thieme article (first line) + PDF (next line)
            article_url = row.get("article_url", "")
            pdf_url = row.get("pdf_url", "")

            if isinstance(article_url, str) and article_url.strip() and isinstance(pdf_url, str) and pdf_url.strip():
                # both links present -> two lines
                st.markdown(
                    f"[Read full article on Thieme]({article_url})  \n"
                    f"[Download PDF]({pdf_url})"
                )
            elif isinstance(article_url, str) and article_url.strip():
                st.markdown(f"[Read full article on Thieme]({article_url})")
            elif isinstance(pdf_url, str) and pdf_url.strip():
                st.markdown(f"[Download PDF]({pdf_url})")



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


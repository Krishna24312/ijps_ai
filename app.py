# app.py
# Run with: streamlit run app.py

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

# Session state for the query
if "query" not in st.session_state:
    st.session_state["query"] = ""


# ----------------- Helpers -----------------
def img_to_base64(path: str) -> str:
    """Load an image file and return a base64 string for CSS embedding."""
    p = Path(path)
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


hero_bg_b64 = img_to_base64("hero_bg.jpg")


# ----------------- Search / NLP logic -----------------
@st.cache_resource
def load_data():
    df = pd.read_csv("ijps_articles.csv")

    title = df["title"].fillna("")
    abstract = df["abstract"].fillna("")
    keywords = df.get("keywords", "").fillna("")
    authors = df.get("authors", "").fillna("")

    title_boost = (title + " ") * 3
    keyword_boost = (keywords + " ") * 2

    df["text"] = title_boost + abstract + " " + keyword_boost

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(df["text"])

    df["title"] = title
    df["abstract"] = abstract
    df["keywords"] = keywords
    df["authors"] = authors

    return df, vectorizer, tfidf_matrix


def split_into_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+", str(text))
    return [s.strip() for s in sentences if s.strip()]


def make_query_aware_teaser(query: str, abstract: str, max_sentences: int = 3) -> str:
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


# ----------------- Global CSS -----------------
st.markdown(
    f"""
<style>
:root {{
  --text: rgba(255,255,255,0.94);
  --muted: rgba(255,255,255,0.70);
  --glass: rgba(255,255,255,0.12);
  --glass-2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.20);
  --shadow: 0 26px 70px rgba(0,0,0,0.55);
}}

html, body, [class*="css"] {{
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}

* {{ box-sizing: border-box; }}

/* BACKGROUND (closer to your reference screenshot) */
.stApp {{
  background-image:
    radial-gradient(900px 520px at 18% 18%, rgba(255,255,255,0.10), rgba(0,0,0,0) 62%),
    linear-gradient(rgba(0,0,0,0.34), rgba(0,0,0,0.68)),
    url("data:image/jpg;base64,{hero_bg_b64}");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
  background-attachment: fixed;
}}

@media (max-width: 900px) {{
  .stApp {{ background-attachment: scroll; }}
}}

header[data-testid="stHeader"] {{ background: transparent; }}

.main .block-container {{
  max-width: 1120px;
  padding-top: 1.9rem;
  padding-bottom: 2.2rem;
  background: transparent;
}}

h1,h2,h3,h4,h5,h6,p,li,label,span {{
  color: var(--text) !important;
}}

div[data-testid="stCaptionContainer"], small {{
  color: var(--muted) !important;
}}

/* HERO (frosted glass like your reference) */
.hero {{
  position: relative;
  width: 100%;
  margin: 1.1rem auto 2.2rem auto;
  border-radius: 2.3rem;
  border: 1px solid var(--border);
  background:
    linear-gradient(135deg, rgba(255,255,255,0.18), rgba(255,255,255,0.06));
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  overflow: hidden;
}}

.hero::before {{
  content: "";
  position: absolute;
  inset: 0;
  /* subtle dark wash so text stays readable over any background */
  background: linear-gradient(90deg, rgba(0,0,0,0.28), rgba(0,0,0,0.12));
  pointer-events: none;
}}

.hero-content {{
  position: relative;
  padding: 3.1rem 3.25rem;
  max-width: 620px;
}}

.hero-kicker {{
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 0.78rem;
  opacity: 0.95;
  margin-bottom: 0.85rem;
  color: rgba(255,255,255,0.78) !important;
}}

/* TITLE: brighter + “marketing hero” feel like your screenshot */
.hero-title {{
  font-size: clamp(2.35rem, 3.8vw, 3.6rem);
  font-weight: 820;
  line-height: 1.05;
  margin-bottom: 0.95rem;
  color: rgba(255,255,255,1) !important;
  letter-spacing: -0.015em;
  text-shadow: 0 10px 30px rgba(0,0,0,0.42);
}}

.hero-subtitle {{
  font-size: 1.02rem;
  line-height: 1.58;
  color: rgba(255,255,255,0.86) !important;
  text-shadow: 0 8px 20px rgba(0,0,0,0.35);
  max-width: 62ch;
}}

@media (max-width: 700px) {{
  .hero {{ border-radius: 1.7rem; }}
  .hero-content {{ padding: 2.2rem 1.6rem; max-width: 100%; }}
}}

/* SUBHEADER */
div[data-testid="stSubheader"] h2 {{
  font-size: 1.28rem;
}}

/* CHIPS */
.stButton > button {{
  width: 100% !important;
  border-radius: 999px !important;
  padding: 0.52rem 0.95rem !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  background: rgba(0,0,0,0.42) !important;
  color: rgba(255,255,255,0.92) !important;
  font-size: 0.86rem !important;
  line-height: 1.2;
  white-space: normal !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.30);
  transition: transform 160ms ease, background 160ms ease, border-color 160ms ease;
}}
.stButton > button:hover {{
  background: rgba(255,255,255,0.12) !important;
  border-color: rgba(255,255,255,0.48) !important;
  transform: translateY(-1px);
}}

/* SEARCH BAR: make it stand out cleanly (white “card” like your reference) */
div[data-testid="stTextInput"] {{
  max-width: 860px;
  margin-top: 0.20rem;
}}

div[data-testid="stTextInput"] input {{
  border-radius: 999px !important;
  padding: 0.86rem 1.10rem !important;

  background: rgba(255,255,255,0.92) !important;
  color: rgba(15,15,15,0.92) !important;

  border: 1px solid rgba(255,255,255,0.55) !important;
  box-shadow: 0 16px 36px rgba(0,0,0,0.35);
  outline: none !important;

  transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
}}

div[data-testid="stTextInput"] input::placeholder {{
  color: rgba(20,20,20,0.55) !important;
}}

div[data-testid="stTextInput"] input:hover {{
  transform: translateY(-1px);
  box-shadow: 0 18px 42px rgba(0,0,0,0.40);
}}

div[data-testid="stTextInput"] input:focus {{
  border-color: rgba(255,255,255,0.85) !important;
  box-shadow: 0 20px 46px rgba(0,0,0,0.45);
}}
div[data-testid="stTextInput"] input:focus-visible {{
  outline: none !important;
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
      to read the full text on Thieme or to view the PDF.
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Query section
st.subheader("Ask a question")
st.caption("Try one of these:")

example_queries = [
    "Complications after hair transplant surgery",
    "Hair care recommendations for alopecia patients using minoxidil",
    "Newer modalities for treatment of pressure ulcers",
    "Effect of caffeine on full-thickness skin graft healing",
]

# Two rows of chips (buttons)
row1 = st.columns(2)
for i, col in enumerate(row1):
    ex = example_queries[i]
    with col:
        if st.button(ex, key=f"example_{i}"):
            st.session_state["query"] = ex

row2 = st.columns(2)
for j, col in enumerate(row2, start=2):
    ex = example_queries[j]
    with col:
        if st.button(ex, key=f"example_{j}"):
            st.session_state["query"] = ex

# Text input bound to session_state["query"]
query = st.text_input(
    "Type your question here:",
    key="query",
    placeholder="e.g. complications after hair transplant surgery",
)

# ----------------- Results -----------------
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

                article_url = row.get("article_url", "")
                pdf_url = row.get("pdf_url", "")

                if (
                    isinstance(article_url, str)
                    and article_url.strip()
                    and isinstance(pdf_url, str)
                    and pdf_url.strip()
                ):
                    st.markdown(
                        f"[Read full article on Thieme]({article_url})  \n"
                        f"[Download PDF]({pdf_url})"
                    )
                elif isinstance(article_url, str) and article_url.strip():
                    st.markdown(f"[Read full article on Thieme]({article_url})")
                elif isinstance(pdf_url, str) and pdf_url.strip():
                    st.markdown(f"[Download PDF]({pdf_url})")

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


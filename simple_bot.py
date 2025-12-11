import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ---------- 1. Load the CSV of articles ----------
df = pd.read_csv("ijps_articles.csv")

# Safely handle missing values
title = df["title"].fillna("")
abstract = df["abstract"].fillna("")
keywords = df.get("keywords", "").fillna("")
pdf_url = df.get("pdf_url", "").fillna("")

# ---------- 2. Build the searchable text with boosted keywords ----------

# Boost the keywords by repeating them a few times
# so TF-IDF gives them more weight in similarity
boost_factor = 3  # you can tune this later if needed
boosted_keywords = (keywords + " ") * boost_factor

# Combine into a single "text" field
df["text"] = title + " " + abstract + " " + boosted_keywords

# Keep cleaned columns (optional but neat)
df["title"] = title
df["abstract"] = abstract
df["keywords"] = keywords
df["pdf_url"] = pdf_url

# ---------- 3. Build TF-IDF matrix for article search ----------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),  # 1- and 2-word phrases
)
tfidf_matrix = vectorizer.fit_transform(df["text"])


def split_into_sentences(text: str):
    """
    Very simple sentence splitter: splits on . ! ? plus whitespace.
    Not perfect, but fine for abstracts.
    """
    sentences = re.split(r'(?<=[.!?])\s+', str(text))
    return [s.strip() for s in sentences if s.strip()]


def make_query_aware_teaser(query: str, abstract: str, max_sentences: int = 3) -> str:
    """
    Phase 2: choose up to max_sentences from the abstract that are
    most relevant to the query using TF-IDF cosine similarity.
    If everything scores 0, fall back to the first sentences.
    """
    sentences = split_into_sentences(abstract)
    if not sentences:
        return ""

    # Build tiny TF-IDF model over sentences of THIS abstract
    sent_vectorizer = TfidfVectorizer(stop_words="english")
    sent_matrix = sent_vectorizer.fit_transform(sentences)

    query_vec = sent_vectorizer.transform([query])
    sims = linear_kernel(query_vec, sent_matrix).flatten()

    # If no overlap at all, just use first sentences
    if sims.max() == 0:
        chosen_indices = list(range(min(max_sentences, len(sentences))))
    else:
        # indices of top-k most similar sentences
        top_indices = sims.argsort()[::-1][:max_sentences]
        # but keep them in original order in the abstract
        chosen_indices = sorted(top_indices)

    chosen_sentences = [sentences[i] for i in chosen_indices]
    teaser = " ".join(chosen_sentences)

    # Limit length
    if len(teaser) > 400:
        teaser = teaser[:400].rsplit(" ", 1)[0] + "..."

    return teaser


def find_top_articles(query: str, top_k: int = 3):
    """
    Given a user query, return the top_k most relevant articles
    and their similarity scores.
    """
    query_vec = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:top_k]

    results = df.iloc[top_indices].copy()
    results["score"] = cosine_similarities[top_indices]
    return results


def answer_query(query: str, top_k: int = 3) -> str:
    """
    Search for the best articles and return:
    - Title
    - similarity score
    - 2–3 query-aware teaser lines from the abstract
    - 'Read more' link
    - PDF link (if available)
    """
    articles = find_top_articles(query, top_k=top_k)

    if articles.empty:
        return "No relevant articles found."

    lines = ["Top matches:"]
    for _, article in articles.iterrows():
        title = article["title"]
        abstract = article["abstract"]
        url = article["article_url"]
        score = article["score"]
        pdf = article.get("pdf_url", "")

        teaser = make_query_aware_teaser(query, abstract, max_sentences=3)

        lines.append("")
        lines.append(f"Title : {title}")
        lines.append(f"Score : {score:.3f}")
        lines.append(teaser)
        lines.append(f"Read more: {url}")

        if isinstance(pdf, str) and pdf.strip():
            lines.append(f"PDF: {pdf}")

        lines.append("-" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    print("IJPS Article Finder (prototype) – Phase 2")
    print("----------------------------------------")
    user_query = input("Type your question and press Enter:\n> ")

    answer = answer_query(user_query, top_k=3)
    print("\n" + answer)

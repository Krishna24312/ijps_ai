# IJPS Article Finder (Prototype)

A lightweight semantic search assistant for the **Indian Journal of Plastic Surgery (IJPS)**.

Given a natural-language query (e.g., *“pressure ulcer newer modalities”*), the app suggests the **top 3 most relevant IJPS articles** from a local CSV dataset, along with:

- Article title  
- Authors  
- A short **query-aware teaser** extracted from the abstract  
- Link to the article landing page (Thieme)  
- (If available) direct PDF link

> This is **not** a general web search engine and **not** a medical advice bot.  
> It only searches within the IJPS articles present in `ijps_articles.csv`.

---

## Repo structure

ijps_ai/
├─ app.py # Streamlit web app
├─ simple_bot.py # CLI version (quick testing)
├─ ijps_articles.csv # Dataset (1 row per IJPS article)
├─ ijps_cover.png # UI asset (cover)
├─ hero_bg.jpg # UI background image
├─ requirements.txt
└─ .gitignore

yaml
Copy code

---

## How it works

### 1) Build a searchable text field per article
For each article row:

- Title is **boosted ×3**
- Keywords are **boosted ×2**
- Abstract is included normally

Combined searchable text:

text = (title + " ") * 3 + abstract + " " + (keywords + " ") * 2

yaml
Copy code

### 2) TF-IDF + cosine similarity ranking
We build a TF-IDF vector space using:

- English stopwords
- 1–2 gram features (unigrams + bigrams)

For a user query, we compute **cosine similarity** and return the **top 3** matches.

### 3) Query-aware teaser (Phase 2)
For each of the selected articles:

- Split abstract into sentences
- Build sentence-level TF-IDF
- Rank sentences by similarity to the user query
- Return up to top 3 sentences  
  (fallback to the first few if similarity scores are all zero)
- Trim to ~400 characters

---

## Local setup (Conda)

### Create/activate environment
```bash
conda create -n krish python=3.10 -y
conda activate krish
Install dependencies
bash
Copy code
pip install -r requirements.txt
Run the Streamlit app
bash
Copy code
streamlit run app.py
Open the local URL shown in your terminal (usually http://localhost:8501).

Demo prompts
Newer modalities for treatment of pressure ulcers

Complications after hair transplant surgery

Effect of caffeine on full-thickness skin graft healing

Run the CLI bot
simple_bot.py provides a command-line interface using the same TF-IDF + cosine similarity approach.

bash
Copy code
python simple_bot.py
(Enter a query when prompted.)

Dataset format (ijps_articles.csv)
Expected columns:

title

abstract

authors

keywords (sometimes blank)

article_url

pdf_url (sometimes blank)

Missing values are handled using .fillna("").

Notes / limitations
This is a prototype for demonstration and educational use.

Search quality depends entirely on the CSV content.

TF-IDF is fast and lightweight, but it does not fully capture meaning like embedding models.

Disclaimer
This tool is for educational and exploratory use only and does not replace clinical judgement.
Always consult the full article and current guidelines before making treatment decisions.

yaml
Copy code

---

## 2) Commands to add it to GitHub (copy-paste in terminal)

From your repo folder:

```bash
cd /path/to/ijps_ai
git status
git add README.md
git commit -m "Add README"
git push
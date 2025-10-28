# streamlit_app.py — Hair Product Recommender (Category + Curl + Porosity only)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("hair_products.csv")

# ---------------------------
# App config + styling
# ---------------------------
st.set_page_config(page_title="Hair Product Recommender", page_icon="💇🏽‍♀️", layout="wide")
st.markdown("""
<style>
.stApp {
  background: linear-gradient(135deg, #ffd6eb 0%, #ffeaf3 50%, #fff8fb 100%);
  font-family: 'Poppins', sans-serif;
}
.main-title {
  background: rgba(255,255,255,0.85);
  border: 2px solid rgba(255,255,255,0.8);
  border-radius: 25px; padding: 25px 40px; text-align: center;
  box-shadow: 0 6px 25px rgba(255, 182, 193, 0.35); margin-bottom: 25px;
}
.main-title h1 { color:#2c2c2c; font-weight:700; font-size:40px; }
.main-title p  { color:#4a4a4a; font-size:16px; }
[data-testid="stDataFrameContainer"] {
  background: rgba(255,255,255,0.85);
  border: 1.5px solid rgba(255,255,255,0.8);
  border-radius: 15px;
  box-shadow: 0 10px 25px rgba(255, 182, 193, 0.2);
  padding: 10px;
}
.stDataFrame td, .stDataFrame th {
  color:#2c2c2c !important;
  background: rgba(255,255,255,0.65) !important;
  border: 1.5px solid rgba(255,255,255,0.9) !important;
  border-radius: 10px !important;
  text-align: center !important;
  transition: all 0.3s ease;
}
.stDataFrame tbody tr:hover td {
  background: rgba(255,255,255,0.95) !important;
  border: 1.5px solid #ffb3d9 !important;
  box-shadow: 0 4px 15px rgba(255,105,180,0.25);
  transform: scale(1.01);
  color: #ff4da6 !important;
  font-weight: 600;
}
div.stButton > button {
  background: linear-gradient(90deg, #ff7eb9, #ff65a3);
  color: white; border-radius: 12px; font-weight: 600;
  border: none; padding: 0.6em 1.2em;
  box-shadow: 0 5px 12px rgba(255,105,180,0.25);
  transition: all 0.3s ease-in-out;
}
div.stButton > button:hover {
  transform: scale(1.05);
  background: linear-gradient(90deg, #ff65a3, #ff2d91);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #fff5fa 0%, #ffeaf3 100%);
  border-right: 2px solid #ffd6ea;
  box-shadow: 4px 0 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.markdown(
    '<div class="main-title"><h1>💇🏽‍♀️ Hair Product Recommender</h1>'
    '<p>Select your Category, Curl Pattern, and Porosity to get matches</p></div>',
    unsafe_allow_html=True
)

# ---------------------------
# Load & normalize CSV
# ---------------------------
@st.cache_data
def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename_map = {"product_name": "name", "curl_patterns": "curl_pattern", "price_usd": "price"}
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    for col in ["name", "category", "curl_pattern", "porosity", "ingredients"]:
        if col not in df.columns:
            df[col] = ""
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"].astype(str).str.replace("$", "", regex=False), errors="coerce")
    for col in ["category", "curl_pattern", "porosity"]:
        df[col] = df[col].astype(str).str.strip()
    return df

df = load_df(DATA)
st.caption(f"Products loaded: {len(df):,}")

# ---------------------------
# Build corpus
# ---------------------------
def build_corpus_row(row: pd.Series) -> str:
    tokens = []
    def add(prefix, value, times=2):
        v = str(value).strip()
        if v and v.lower() != "nan":
            tokens.extend([f"{prefix}-{v.replace(' ', '-').lower()}"] * times)
    add("category", row.get("category", ""))
    for c in str(row.get("curl_pattern", "")).split(","):
        c = c.strip()
        if c:
            tokens.append(f"curl-{c.lower()}")
    for p in str(row.get("porosity", "")).split(","):
        p = p.strip()
        if p:
            tokens.append(f"porosity-{p.lower()}")
    return " ".join(tokens)

@st.cache_data
def build_corpus(df_in: pd.DataFrame) -> pd.Series:
    return df_in.apply(build_corpus_row, axis=1)

corpus = build_corpus(df)

@st.cache_resource
def make_tfidf(corpus_series: pd.Series):
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(corpus_series.values)
    return vec, X

vec, X = make_tfidf(corpus)

# ---------------------------
# Query builder + recommender
# ---------------------------
def build_query(sel_cat, sel_curl, sel_por):
    toks = []
    if sel_cat and sel_cat != "(any)": toks.append(f"category-{sel_cat.lower()}")
    if sel_curl and sel_curl != "(any)": toks.append(f"curl-{sel_curl.lower()}")
    if sel_por and sel_por != "(any)": toks.append(f"porosity-{sel_por.lower()}")
    return " ".join(toks)

def recommend(query, candidate_df, top_n=12):
    if candidate_df.empty:
        return candidate_df
    if not query.strip():
        candidate_df["score"] = 0.0
        return candidate_df.head(top_n)
    qvec = vec.transform([query])
    sims = cosine_similarity(qvec, X[candidate_df.index, :]).ravel()
    out = candidate_df.copy()
    out["score"] = sims
    return out.sort_values("score", ascending=False).head(top_n)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Your hair profile")
cat_opts  = ["(any)"] + sorted(df["category"].dropna().unique().tolist())
curl_opts = ["(any)"] + sorted(df["curl_pattern"].dropna().unique().tolist())
por_opts  = ["(any)"] + sorted(df["porosity"].dropna().unique().tolist())

sel_cat  = st.sidebar.selectbox("Category", cat_opts)
sel_curl = st.sidebar.selectbox("Curl Pattern (hair type)", curl_opts)
sel_por  = st.sidebar.selectbox("Porosity Level", por_opts)

filtered = df.copy()
if sel_cat  != "(any)": filtered = filtered[filtered["category"] == sel_cat]
if sel_curl != "(any)": filtered = filtered[filtered["curl_pattern"].str.contains(sel_curl, na=False)]
if sel_por  != "(any)": filtered = filtered[filtered["porosity"].str.contains(sel_por, na=False)]

query = build_query(sel_cat, sel_curl, sel_por)

# ---------------------------
# Output
# ---------------------------
if st.button("Find Products"):
    recs = recommend(query, filtered, top_n=12)
    if recs.empty:
        st.info("No matches. Try loosening your selections.")
    else:
        st.subheader("Recommended for you")
        cols = ["name", "category", "curl_pattern", "porosity"]
        if "price" in recs.columns:
            cols.insert(2, "price")
        if "score" in recs.columns:
            cols.append("score")
        st.dataframe(recs[cols].reset_index(drop=True), use_container_width=True)

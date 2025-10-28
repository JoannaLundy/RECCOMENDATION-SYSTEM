# streamlit_app.py — Hair Product Recommender (Profile → Products)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("hair_products.csv")

# --------------------------------------------
# App config
# --------------------------------------------
st.set_page_config(
    page_title="Hair Product Recommender",
    page_icon="💇🏽‍♀️",
    layout="wide"
)

# --------------------------------------------
# Theme / styling (your original, kept)
# --------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffd6eb 0%, #ffeaf3 50%, #fff8fb 100%);
    font-family: 'Poppins', sans-serif;
}
.main-title {
    background: rgba(255,255,255,0.85);
    border: 2px solid rgba(255,255,255,0.8);
    border-radius: 25px;
    padding: 25px 40px;
    text-align: center;
    box-shadow: 0 6px 25px rgba(255, 182, 193, 0.35);
    margin-bottom: 25px;
}
.main-title h1 { color:#2c2c2c; font-weight:700; font-size:40px; }
.main-title p  { color:#4a4a4a; font-size:16px; }

[data-testid="stDataFrameContainer"] {
    background: rgba(255,255,255,0.85);
    border: 1.5px solid rgba(255,255,255,0.8);
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(255, 182, 193, 0.2);
    backdrop-filter: blur(8px);
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
    color: white;
    border-radius: 12px;
    font-weight: 600;
    border: none;
    padding: 0.6em 1.2em;
    box-shadow: 0 5px 12px rgba(255, 105, 180, 0.25);
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

/* 📱 Mobile tweaks */
@media (max-width: 768px){
  .block-container{ padding: 0.8rem 0.6rem !important; }
  .stDataFrame, .stTable{ overflow-x:auto; }
  .stButton>button, .stDownloadButton>button{ width:100% !important; }
  [data-baseweb="select"], .stTextInput, .stNumberInput, .stSlider,
  .stMultiSelect, .stSelectbox{ min-width:100% !important; }

  .card{
    background: rgba(255,255,255,0.85);
    border: 1.5px solid rgba(255,255,255,0.8);
    border-radius: 14px;
    box-shadow: 0 6px 16px rgba(255,105,180,0.18);
    padding: 12px 14px; margin: 10px 0;
  }
  .card h4{ margin:0 0 6px; color:#2c2c2c; }
  .card small{ color:#555; }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------
# Title
# --------------------------------------------
st.markdown(
    '<div class="main-title"><h1>💇🏽‍♀️ Hair Product Recommender</h1><p>Tell us your hair profile → get product picks</p></div>',
    unsafe_allow_html=True
)

# --------------------------------------------
# Load data
# --------------------------------------------
@st.cache_data
def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize common alt column names
    rename_map = {"product_name":"name","curl_patterns":"curl_pattern","price_usd":"price"}
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)
    # Ensure required cols exist
    for col in ["name","category","curl_pattern","porosity","ingredients"]:
        if col not in df.columns:
            df[col] = ""
    # Clean / types
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"].astype(str).str.replace("$","",regex=False), errors="coerce")
    for col in ["category","curl_pattern","porosity"]:
        df[col] = df[col].astype(str).str.strip()
    return df

df = load_df(DATA)
st.caption(f"Products loaded: {len(df):,}")

# --------------------------------------------
# Build corpus for TF-IDF
# --------------------------------------------
def build_corpus_row(row: pd.Series) -> str:
    tokens = []
    def add_dup(prefix, value, times=2):
        v = str(value).strip()
        if v and v.lower() != "nan":
            tokens.extend([f"{prefix}-{v.replace(' ','-').lower()}"] * times)

    add_dup("category", row.get("category",""), times=2)

    for c in str(row.get("curl_pattern","")).split(","):
        c = c.strip()
        if c:
            tokens.extend([f"curl-{c.replace(' ','-').lower()}"] * 2)

    for p in str(row.get("porosity","")).split(","):
        p = p.strip()
        if p:
            tokens.extend([f"porosity-{p.replace(' ','-').lower()}"] * 2)

    for c in str(row.get("concerns","")).split(","):
        c = c.strip()
        if c:
            tokens.append(f"concern-{c.replace(' ','-').lower()}")

    for flag in ["protein_treatment","sulfate_free","silicone_free","glycerin_present"]:
        if flag in row.index:
            val = str(row.get(flag,"0")).strip().lower()
            if val in ("1","true","yes"):
                tokens.append(flag.replace("_","-"))

    ings = str(row.get("ingredients",""))
    if ings:
        tokens += [w.strip().replace(" ","-").lower() for w in ings.split(",") if w.strip()]

    return " ".join(tokens)

@st.cache_data
def build_corpus(df_in: pd.DataFrame) -> pd.Series:
    return df_in.apply(build_corpus_row, axis=1)

corpus = build_corpus(df)

@st.cache_resource
def make_tfidf(corpus_series: pd.Series):
    vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(corpus_series.values)
    return vec, X

vec, X = make_tfidf(corpus)

# --------------------------------------------
# Query builder (NEW)
# --------------------------------------------
def build_query_tokens(sel_cat: str | None,
                       sel_curl: str | None,
                       sel_por: str | None,
                       concerns: list[str] | None = None,
                       flags: dict[str, bool] | None = None) -> str:
    toks = []
    def tokify(prefix, value, times=2):
        v = (value or "").strip()
        if v and v.lower() != "nan":
            toks.extend([f"{prefix}-{v.replace(' ', '-').lower()}"] * times)

    tokify("category", sel_cat, times=2)
    tokify("curl", sel_curl,   times=2)
    tokify("porosity", sel_por, times=2)

    if concerns:
        for c in concerns:
            c = (c or "").strip()
            if c:
                toks.append(f"concern-{c.replace(' ', '-').lower()}")

    if flags:
        for k, v in flags.items():
            if v:
                toks.append(k.replace("_","-"))

    return " ".join(toks)

# --------------------------------------------
# Sidebar — Hair profile inputs (NEW)
# --------------------------------------------
st.sidebar.header("Your hair profile")

cat_opts  = sorted([c for c in df["category"].dropna().unique().tolist() if c])
curl_opts = sorted([c for c in df["curl_pattern"].dropna().unique().tolist() if c])
por_opts  = sorted([p for p in df["porosity"].dropna().unique().tolist() if p])

sel_cat  = st.sidebar.selectbox("Category (e.g., Shampoo, Mask…)", ["(any)"] + cat_opts)
sel_curl = st.sidebar.selectbox("Curl pattern", ["(any)"] + curl_opts)
sel_por  = st.sidebar.selectbox("Porosity", ["(any)"] + por_opts)

# Optional concerns from data
concern_opts = sorted({c.strip()
                       for x in df.get("concerns", pd.Series([""])).fillna("")
                       for c in str(x).split(",") if c.strip()})
sel_concerns = st.sidebar.multiselect("Concerns (optional)", concern_opts) if concern_opts else []

# Optional binary flags
flag_fields = [f for f in ["protein_treatment","sulfate_free","silicone_free","glycerin_present"]
               if f in df.columns]
flag_values = {}
if flag_fields:
    st.sidebar.markdown("**Preferences (optional)**")
    for f in flag_fields:
        flag_values[f] = st.sidebar.checkbox(f.replace("_"," ").title(), value=False)

# Exact filter first (keeps catalog relevant)
filtered = df.copy()
if sel_cat  != "(any)": filtered = filtered[filtered["category"] == sel_cat]
if sel_curl != "(any)": filtered = filtered[filtered["curl_pattern"].str.contains(sel_curl, na=False)]
if sel_por  != "(any)": filtered = filtered[filtered["porosity"].str.contains(sel_por, na=False)]

# Build a query vector from selections
query = build_query_tokens(
    sel_cat  if sel_cat  != "(any)" else None,
    sel_curl if sel_curl != "(any)" else None,
    sel_por  if sel_por  != "(any)" else None,
    sel_concerns,
    flag_values if flag_fields else None
)

# --------------------------------------------
# Recommend from query (NEW)
# --------------------------------------------
def recommend_from_query(query_str: str, candidate_df: pd.DataFrame, top_n=12) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    if not query_str.strip():
        out = candidate_df.copy()
        out["score"] = 0.0  # no preferences → neutral
        return out.head(top_n)
    qvec = vec.transform([query_str])
    sims = cosine_similarity(qvec, X[candidate_df.index, :]).ravel()
    out = candidate_df.copy()
    out["score"] = sims
    return out.sort_values("score", ascending=False).head(top_n)

# --------------------------------------------
# UI actions
# --------------------------------------------
st.markdown("### Tell us your hair type & porosity, and we’ll suggest products ✨")

# Quick utilities
col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("🔄 Reload data"):
        st.cache_data.clear()
        st.rerun()
with col_b:
    mobile_mode = st.toggle("📱 Mobile mode (stacked)", value=False)

# Results
if st.button("Find products"):
    recs = recommend_from_query(query, filtered, top_n=12)
    if recs.empty:
        st.info("No matches. Try loosening your selections.")
    else:
        st.subheader("Recommended for you")
        cols_to_show = ["name", "category", "curl_pattern", "porosity"]
        if "price" in recs.columns: cols_to_show.insert(2, "price")
        if "score" in recs.columns: cols_to_show.append("score")

        if mobile_mode:
            for _, row in recs[cols_to_show].reset_index(drop=True).iterrows():
                st.markdown(f"""
                <div class="card">
                  <h4>{row['name']}</h4>
                  <small>Category: {row.get('category','')} · Curl: {row.get('curl_pattern','')} · Porosity: {row.get('porosity','')}</small><br/>
                  {"<small>Price: $" + f"{row['price']:.2f}" + "</small><br/>" if "price" in recs.columns and pd.notna(row.get("price")) else ""}
                  {"<small>Match score: " + f"{row.get('score',0):.3f}" + "</small>" if "score" in recs.columns else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.dataframe(
                recs[cols_to_show].reset_index(drop=True),
                use_container_width=True
            )

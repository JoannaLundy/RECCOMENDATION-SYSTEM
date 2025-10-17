# streamlit_app.py — Hair Product Recommender (Desktop + Mobile Mode)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("hair_products.csv")

# -----------------------------------------------------------
# Keep your original desktop feel: WIDE layout
# (no forced sidebar state changes)
# -----------------------------------------------------------
st.set_page_config(
    page_title="Hair Product Recommender",
    page_icon="💇🏽‍♀️",
    layout="wide"     # ← same as your original
)

# -----------------------------------------------------------
# Your original pink theme + a few safe mobile tweaks
# (desktop styling remains as-is; mobile-only rules live in @media)
# -----------------------------------------------------------
st.markdown("""
<style>
/* --- Background (your original style) --- */
.stApp {
    background: linear-gradient(135deg, #ffd6eb 0%, #ffeaf3 50%, #fff8fb 100%);
    font-family: 'Poppins', sans-serif;
}
/* --- Title card --- */
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

/* --- Tables (your original style) --- */
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

/* --- Sliders (original) --- */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #ff77b9, #ff4da6) !important;
}

/* --- Buttons (original) --- */
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

/* --- Sidebar (original) --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff5fa 0%, #ffeaf3 100%);
    border-right: 2px solid #ffd6ea;
    box-shadow: 4px 0 8px rgba(0,0,0,0.05);
}

/* ===== 📱 MOBILE-ONLY TWEAKS (desktop unaffected) ===== */
@media (max-width: 768px){
  .block-container{ padding: 0.8rem 0.6rem !important; }
  .stDataFrame, .stTable{ overflow-x:auto; }        /* avoid overflow */
  .stButton>button, .stDownloadButton>button{ width:100% !important; }
  [data-baseweb="select"], .stTextInput, .stNumberInput, .stSlider,
  .stMultiSelect, .stSelectbox{ min-width:100% !important; }

  /* Card style used in 'Mobile mode' stacked layout */
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

# -----------------------------------------------------------
# Title
# -----------------------------------------------------------
st.markdown(
    '<div class="main-title"><h1>💇🏽‍♀️ Hair Product Recommender</h1><p>Find products that love your curls!</p></div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------
# Load Data
# -----------------------------------------------------------
@st.cache_data
def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize names if needed
    rename_map = {"product_name":"name","curl_patterns":"curl_pattern","price_usd":"price"}
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k:v}, inplace=True)
    # ensure required cols
    for col in ["name","category","curl_pattern","porosity","ingredients"]:
        if col not in df.columns:
            df[col] = ""
    # price coercion
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"].astype(str).str.replace("$","",regex=False), errors="coerce")
    for col in ["category","curl_pattern","porosity"]:
        df[col] = df[col].astype(str).str.strip()
    return df

df = load_df(DATA)

# -----------------------------------------------------------
# Build Corpus
# -----------------------------------------------------------
def build_corpus_row(row: pd.Series) -> str:
    tokens = []
    def add_dup(prefix, value, times=2):
        v = str(value).strip()
        if v and v.lower()!="nan":
            tokens.extend([f"{prefix}-{v.replace(' ','-').lower()}"]*times)
    add_dup("category", row.get("category",""), times=2)
    for c in str(row.get("curl_pattern","")).split(","):
        c=c.strip()
        if c: tokens.extend([f"curl-{c.replace(' ','-').lower()}"]*2)
    for p in str(row.get("porosity","")).split(","):
        p=p.strip()
        if p: tokens.extend([f"porosity-{p.replace(' ','-').lower()}"]*2)
    for c in str(row.get("concerns","")).split(","):
        c=c.strip()
        if c: tokens.append(f"concern-{c.replace(' ','-').lower()}")
    for flag in ["protein_treatment","sulfate_free","silicone_free","glycerin_present"]:
        if flag in row.index:
            val=str(row.get(flag,"0")).strip().lower()
            if val in ("1","true","yes"): tokens.append(flag.replace("_","-"))
    ings=str(row.get("ingredients",""))
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

# -----------------------------------------------------------
# Sidebar Filters (unchanged desktop behavior)
# -----------------------------------------------------------
st.sidebar.header("Filter products")
cat_opts = ["(any)"] + sorted([c for c in df["category"].dropna().unique().tolist() if c])
curl_opts = ["(any)"] + sorted([c for c in df["curl_pattern"].dropna().unique().tolist() if c])
por_opts  = ["(any)"] + sorted([p for p in df["porosity"].dropna().unique().tolist() if p])

sel_cat  = st.sidebar.selectbox("Category", cat_opts)
sel_curl = st.sidebar.selectbox("Curl pattern", curl_opts)
sel_por  = st.sidebar.selectbox("Porosity", por_opts)

filtered = df.copy()
if sel_cat!="(any)":  filtered = filtered[filtered["category"]==sel_cat]
if sel_curl!="(any)": filtered = filtered[filtered["curl_pattern"].str.contains(sel_curl,na=False)]
if sel_por!="(any)":  filtered = filtered[filtered["porosity"].str.contains(sel_por,na=False)]

st.write("Rate a few products (1–10). You only need 2–3 ratings to get personalized results.")

# -----------------------------------------------------------
# Desktop grid (original) + Mobile stacked option (new)
# Users can toggle mobile mode ON when on phones; OFF keeps your desktop grid.
# -----------------------------------------------------------
sample = filtered.head(12).reset_index(drop=True)
ratings = {}

mobile_mode = st.toggle("📱 Mobile mode (stacked)", value=False)

if mobile_mode:
    # Stacked cards — great on small screens
    for i, row in sample.iterrows():
        st.markdown(f"""
        <div class="card">
          <h4>{row['name']}</h4>
          <small>Category: {row.get('category','')} · Curl: {row.get('curl_pattern','')} · Porosity: {row.get('porosity','')}</small>
        </div>
        """, unsafe_allow_html=True)
        ratings[row["name"]] = st.slider("Your rating", 0, 10, 0, key=f"r{i}")
        st.divider()
else:
    # Your original wide/desktop layout with columns
    show_cols = ["name", "category"]
    if "price" in df.columns: show_cols.append("price")
    cols = st.columns([4, 2, 2, 2]) if "price" in df.columns else st.columns([5, 2, 2])
    header = ["Name", "Category"] + (["Price"] if "price" in df.columns else []) + ["Your rating"]
    for c, h in zip(cols, header):
        c.markdown(f"**{h}**")

    for i, row in sample.iterrows():
        c0, c1, *rest = cols
        c0.write(row["name"])
        c1.write(row["category"])
        if "price" in df.columns:
            c2, c3 = rest
            c2.write(f"${row['price']:.2f}" if pd.notna(row["price"]) else "—")
            ratings[row["name"]] = c3.slider("", 0, 10, 0, key=f"r{i}")
        else:
            c2 = rest[0]
            ratings[row["name"]] = c2.slider("", 0, 10, 0, key=f"r{i}")

# -----------------------------------------------------------
# Recommendation Logic (unchanged)
# -----------------------------------------------------------
def get_recommendations(rated_dict, top_n=8):
    if not rated_dict:
        return pd.DataFrame()
    rated_df = df[df["name"].isin(rated_dict.keys())]
    if rated_df.empty:
        return pd.DataFrame()

    idx = rated_df.index.to_numpy()
    weights = np.array([rated_dict[n] for n in rated_df["name"]], dtype=float)

    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    user_vec = (weights @ X[idx, :]).reshape(1, -1)  # weighted avg vector
    sims = cosine_similarity(user_vec, X).ravel()

    out = df.copy()
    out["score"] = sims
    out = out[~out["name"].isin(rated_dict.keys())]  # exclude rated
    # keep respecting active filters
    if sel_cat!="(any)":  out = out[out["category"]==sel_cat]
    if sel_curl!="(any)": out = out[out["curl_pattern"].str.contains(sel_curl,na=False)]
    if sel_por!="(any)":  out = out[out["porosity"].str.contains(sel_por,na=False)]

    return out.sort_values("score", ascending=False).head(top_n)

if st.button("Get recommendations"):
    rated = {k: v for k, v in ratings.items() if v > 0}
    if len(rated) < 1:
        st.warning("Please rate at least one product.")
    else:
        recs = get_recommendations(rated, top_n=8)
        if recs.empty:
            st.info("No candidates match your filters. Try loosening filters or rate different items.")
        else:
            st.subheader("Recommended for you")
            cols_to_show = ["name", "category", "curl_pattern", "porosity", "score"]
            if "price" in recs.columns:
                cols_to_show.insert(2, "price")
            st.dataframe(
                recs[cols_to_show].reset_index(drop=True),
                use_container_width=True   # fluid table for both desktop & mobile
            )

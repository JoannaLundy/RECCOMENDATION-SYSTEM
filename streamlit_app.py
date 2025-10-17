# streamlit_app.py  — Hair Product Recommender (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("hair_products.csv")

st.set_page_config(page_title="Hair Product Recommender", page_icon="💇🏽‍♀️", layout="wide")
# Add gradient background and custom CSS
# 💅 Modern Pink Gradient Theme
st.markdown("""
<style>
/* --- Background --- */
.stApp {
    background: linear-gradient(135deg, #ffd6eb 0%, #ffeaf3 50%, #fff8fb 100%);
    font-family: 'Poppins', sans-serif;
}

/* --- Main Title --- */
.main-title {
    background: rgba(255,255,255,0.85);
    border: 2px solid rgba(255,255,255,0.8);
    border-radius: 25px;
    padding: 25px 40px;
    text-align: center;
    box-shadow: 0 6px 25px rgba(255, 182, 193, 0.35);
    margin-bottom: 25px;
}
.main-title h1 {
    color: #2c2c2c;
    font-weight: 700;
    font-size: 40px;
}
.main-title p {
    color: #4a4a4a;
    font-size: 16px;
}

/* --- Table Container --- */
.stDataFrame {
    border: none !important;
}
[data-testid="stDataFrameContainer"] {
    background: rgba(255,255,255,0.85);
    border: 1.5px solid rgba(255,255,255,0.8);
    border-radius: 15px;
    box-shadow: 0 10px 25px rgba(255, 182, 193, 0.2);
    backdrop-filter: blur(8px);
    padding: 10px;
}

/* --- Table Text Cells --- */
.stDataFrame td, .stDataFrame th {
    color: #2c2c2c !important;
    background: rgba(255,255,255,0.65) !important;
    border: 1.5px solid rgba(255,255,255,0.9) !important;
    border-radius: 10px !important;
    text-align: center !important;
    transition: all 0.3s ease;
}

/* --- Hover Effects --- */
.stDataFrame tbody tr:hover td {
    background: rgba(255,255,255,0.95) !important;
    border: 1.5px solid #ffb3d9 !important;
    box-shadow: 0 4px 15px rgba(255,105,180,0.25);
    transform: scale(1.01);
    color: #ff4da6 !important;
    font-weight: 600;
}

/* --- Sliders --- */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #ff77b9, #ff4da6) !important;
}

/* --- Buttons --- */
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

/* --- Sidebar --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff5fa 0%, #ffeaf3 100%);
    border-right: 2px solid #ffd6ea;
    box-shadow: 4px 0 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


st.title("💇🏽‍♀️ Hair Product Recommender")

# ---------- Load data ----------
@st.cache_data
def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize expected column names
    rename_map = {
        "product_name": "name",
        "curl_patterns": "curl_pattern",
        "price_usd": "price",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    # Ensure required columns exist
    for col in ["name", "category", "curl_pattern", "porosity", "ingredients"]:
        if col not in df.columns:
            df[col] = ""

    # Coerce price if present
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"].astype(str).str.replace("$", "", regex=False), errors="coerce")

    # Strip/clean
    for col in ["category", "curl_pattern", "porosity"]:
        df[col] = df[col].astype(str).str.strip()

    return df

df = load_df(DATA)

# ---------- Build text corpus per product ----------
def build_corpus_row(row: pd.Series) -> str:
    tokens = []

    def add_dup(prefix, value, times=2):
        v = str(value).strip()
        if v and v.lower() != "nan" and v != "":
            tokens.extend([f"{prefix}-{v.replace(' ', '-')}".lower()] * times)

    add_dup("category", row.get("category", ""), times=2)

    # split CSV-like lists
    for c in str(row.get("curl_pattern", "")).split(","):
        c = c.strip()
        if c:
            tokens.extend([f"curl-{c.replace(' ', '-')}".lower()] * 2)

    for p in str(row.get("porosity", "")).split(","):
        p = p.strip()
        if p:
            tokens.extend([f"porosity-{p.replace(' ', '-')}".lower()] * 2)

    for c in str(row.get("concerns", "")).split(","):
        c = c.strip()
        if c:
            tokens.append(f"concern-{c.replace(' ', '-')}".lower())

    # boolean flags if present
    for flag in ["protein_treatment", "sulfate_free", "silicone_free", "glycerin_present"]:
        if flag in row.index:
            val = str(row.get(flag, "0")).strip().lower()
            if val in ("1", "true", "yes"):
                tokens.append(flag.replace("_", "-"))

    # ingredients list
    ings = str(row.get("ingredients", ""))
    if ings:
        tokens += [w.strip().replace(" ", "-").lower() for w in ings.split(",") if w.strip()]

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

# ---------- Sidebar filters ----------
st.sidebar.header("Filter products")
cat_opts = ["(any)"] + sorted([c for c in df["category"].dropna().unique().tolist() if c != ""])
curl_opts = ["(any)"] + sorted([c for c in df["curl_pattern"].dropna().unique().tolist() if c != ""])
por_opts = ["(any)"] + sorted([p for p in df["porosity"].dropna().unique().tolist() if p != ""])

sel_cat = st.sidebar.selectbox("Category", cat_opts)
sel_curl = st.sidebar.selectbox("Curl pattern", curl_opts)
sel_por = st.sidebar.selectbox("Porosity", por_opts)

filtered = df.copy()
if sel_cat != "(any)":
    filtered = filtered[filtered["category"] == sel_cat]
if sel_curl != "(any)":
    # allow multi-values like "3B,3C,4A"
    filtered = filtered[filtered["curl_pattern"].str.contains(sel_curl, na=False)]
if sel_por != "(any)":
    filtered = filtered[filtered["porosity"].str.contains(sel_por, na=False)]

st.write("Rate a few products (1–10). You only need 2–3 ratings to get personalized results.")
show_cols = ["name", "category"]
if "price" in df.columns: show_cols.append("price")

# Limit how many sliders we render at once
sample = filtered.head(12).reset_index(drop=True)
ratings = {}
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

# ---------- Recommend ----------
def get_recommendations(rated_dict, top_n=8):
    if not rated_dict:
        return pd.DataFrame()

    # indices of rated items
    rated_df = df[df["name"].isin(rated_dict.keys())]
    if rated_df.empty:
        return pd.DataFrame()

    idx = rated_df.index.to_numpy()
    weights = np.array([rated_dict[n] for n in rated_df["name"]], dtype=float)

    # normalize weights (avoid divide-by-zero)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    user_vec = (weights @ X[idx, :]).reshape(1, -1)  # weighted average vector
    sims = cosine_similarity(user_vec, X).ravel()

    out = df.copy()
    out["score"] = sims
    out = out[~out["name"].isin(rated_dict.keys())]  # exclude rated
    # respect sidebar filters for candidates
    if sel_cat != "(any)":
        out = out[out["category"] == sel_cat]
    if sel_curl != "(any)":
        out = out[out["curl_pattern"].str.contains(sel_curl, na=False)]
    if sel_por != "(any)":
        out = out[out["porosity"].str.contains(sel_por, na=False)]

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
            cols_to_show = ["name", "category", "curl_pattern", "porosity", "score"]
            if "price" in recs.columns:
                cols_to_show.insert(2, "price")
            st.subheader("Recommended for you")
            st.dataframe(recs[cols_to_show].reset_index(drop=True))


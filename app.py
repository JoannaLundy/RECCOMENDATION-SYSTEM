# app.py — Hair Product Recommender (Flask)
from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load data ----------
CSV = Path("data/hair_products.csv")
df_full = pd.read_csv(CSV)

# ---------- Feature engineering ----------
def build_corpus_row(row):
    tokens = []

    # Upweight category (duplicate once)
    cat = str(row.get("category", "")).strip()
    if cat:
        tokens += [f"category_{cat}", f"category_{cat}"]

    # Upweight curl patterns
    for c in str(row.get("curl_patterns", "")).split(","):
        c = c.strip()
        if c:
            tokens += [f"curl_{c}", f"curl_{c}"]

    # Upweight porosity
    for p in str(row.get("porosity", "")).split(","):
        p = p.strip()
        if p:
            tokens += [f"porosity_{p}", f"porosity_{p}"]

    # Concerns (single weight)
    for c in str(row.get("concerns", "")).split(","):
        c = c.strip()
        if c:
            tokens.append(f"concern_{c}")

    # Binary flags
    if str(row.get("protein_treatment", "0")) == "1": tokens.append("protein_treatment")
    if str(row.get("sulfate_free", "0")) == "1":     tokens.append("sulfate_free")
    if str(row.get("silicone_free", "0")) == "1":    tokens.append("silicone_free")
    if str(row.get("glycerin_present", "0")) == "1": tokens.append("glycerin_present")

    # Ingredients
    for w in str(row.get("ingredients", "")).split(","):
        w = w.strip().replace(" ", "_")
        if w:
            tokens.append(w)

    return " ".join(tokens)

def fit_vectorizer(df):
    df = df.copy()
    df["corpus"] = df.apply(build_corpus_row, axis=1)
    # small quality bump
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    X = tfidf.fit_transform(df["corpus"])
    return df, tfidf, X

def user_profile_from_ratings(X, ratings):
    if not ratings:
        return None
    idxs = list(ratings.keys())
    w = np.array([ratings[i] for i in idxs], dtype=float)
    w = w / w.sum()
    prof = w @ X[idxs].toarray()
    return prof / (np.linalg.norm(prof) + 1e-9)

def recommend(X, profile, exclude=None, topn=8):
    if profile is None:
        order = list(range(min(topn, X.shape[0])))
        return order, [0.0] * len(order)
    sims = cosine_similarity(X, profile.reshape(1, -1)).ravel()
    if exclude:
        for i in exclude:
            if 0 <= i < len(sims):
                sims[i] = -1.0
    order = sims.argsort()[::-1][:topn]
    return list(order), list(sims[order])

# ---------- Flask app ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    df = df_full.copy()

    # dropdown options
    cats  = sorted(df["category"].dropna().unique())
    curls = sorted({c.strip() for s in df["curl_patterns"].dropna() for c in str(s).split(",") if c.strip()})
    poros = sorted({p.strip() for s in df["porosity"].dropna()     for p in str(s).split(",") if p.strip()})

    return render_template(
        "index.html",
        products=df.to_dict(orient="records"),
        cats=cats, curls=curls, poros=poros
    )

@app.route("/recommend", methods=["POST"])
def do_reco():
    df = df_full.copy()

    # ----- filters (robust, no "truth value" bug) -----
    cat  = request.form.get("category", "").strip()
    curl = request.form.get("curl", "").strip()
    por  = request.form.get("porosity", "").strip()

    if cat:
        df = df[df["category"].fillna("").str.lower() == cat.lower()].reset_index(drop=True)

    if curl:
        df = df[df["curl_patterns"].fillna("").str.contains(rf"\b{re.escape(curl)}\b", case=False, na=False)]

    if por:
        df = df[df["porosity"].fillna("").str.contains(rf"\b{re.escape(por)}\b", case=False, na=False)]

    # If filters removed everything, fallback to full set
    if df.empty:
        df = df_full.copy()

    # Vectorize after filtering
    df_vec, tfidf, X = fit_vectorizer(df)

    # ratings from form inputs r_0, r_1, ...
    ratings = {}
    for i in range(len(df_vec)):
        val = request.form.get(f"r_{i}", "").strip()
        if val:
            try:
                r = float(val)
                if 1 <= r <= 10:
                    ratings[i] = r
            except:
                pass

    profile = user_profile_from_ratings(X, ratings)
    exclude = list(ratings.keys())
    order, scores = recommend(X, profile, exclude=exclude, topn=8)

    results = []
    for i, s in zip(order, scores):
        row = df_vec.iloc[i].to_dict()
        row["score"] = round(float(s), 3)
        results.append(row)

    return render_template("results.html", results=results)

if __name__ == "__main__":
    # run on localhost
    app.run(host="127.0.0.1", port=7860, debug=True)

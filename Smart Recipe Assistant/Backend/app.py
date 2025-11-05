# app.py
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ---- Copyed/adapted helpers from your notebook ----
SYNONYM_MAP_PHRASES = {
    r"\bcapsicum\b": "bell pepper",
    r"\bmirchi\b": "chili",
    r"\bchilli(es)?\b": "chili",
    r"\bchilies\b": "chili",
    r"\bgreen chili(es)?\b": "chili",
    r"\bred chili(es)?\b": "chili",
    r"\bcoriander leaves\b": "coriander",
    r"\bdhania\b": "coriander",
    r"\bcilantro\b": "coriander",
    r"\bjeera\b": "cumin",
    r"\bzeera\b": "cumin",
    r"\bhari mirch\b": "chili",
    r"\bhing\b": "asafoetida",
    r"\bgur\b": "jaggery",
    r"\bgud\b": "jaggery",
    r"\bmaida\b": "all purpose flour",
    r"\batta\b": "wheat flour",
    r"\bdahi\b": "yogurt",
    r"\bcurd\b": "yogurt",
    r"\bmethi\b": "fenugreek",
    r"\bkasuri methi\b": "fenugreek",
    r"\baloo\b": "potato",
    r"\bbhindi\b": "okra",
    r"\btadka\b": "tempering",
}

NOISE_TOKENS = {
    "fresh", "finely", "chopped", "sliced", "diced", "ground", "powder",
    "optional", "to", "taste", "medium", "large", "small", "cup", "cups",
    "tsp", "tbsp", "tablespoon", "teaspoon", "pinch", "piece", "pieces",
    "handful", "and", "or", "of"
}

def apply_synonyms(text: str) -> str:
    s = text
    for pattern, repl in SYNONYM_MAP_PHRASES.items():
        s = re.sub(pattern, repl, s)
    return s

def normalize_ingredients(value: Union[str, None]) -> str:
    if value is None:
        return ""
    s = str(value).lower()
    s = apply_synonyms(s)
    s = re.sub(r"(\d+\/\d+|\d+\.\d+|\d+)", " ", s)
    s = re.sub(r"[^a-z,;\s\-]", " ", s)
    tokens = [t for t in re.split(r"[\s,;]+", s) if t and t not in NOISE_TOKENS]
    s = " ".join(tokens)
    return re.sub(r"\s+", " ", s).strip()

def _tokenize_norm(text: str):
    if not text:
        return []
    return [t for t in normalize_ingredients(text).split() if t]

class RecipeRecord:
    def __init__(self, idx: int, name: str, ing_text_norm: str, raw_ing: str, raw_ins: str, cuisine: str, source: str):
        self.idx = idx
        self.name = name
        self.ing_text = ing_text_norm
        self.raw_ingredients = raw_ing
        self.raw_instructions = raw_ins
        self.cuisine = cuisine
        self.source = source
        self.token_set = set(ing_text_norm.split())

class IngredientSearch:
    def __init__(self):
        self.vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, strip_accents="unicode", sublinear_tf=True)
        self.vec_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=1, strip_accents="unicode")
        self.records: List[RecipeRecord] = []
        self.M_word = None
        self.M_char = None
        self.idf_lookup = {}
        self.idf_default = 1.0

    def fit(self, df: pd.DataFrame):
        ing_norm = df["ingredients"].apply(normalize_ingredients)
        self.records = [
            RecipeRecord(i, str(df.iloc[i]["name"]), ing_norm.iloc[i], str(df.iloc[i]["ingredients"]), str(df.iloc[i]["instructions"]), str(df.iloc[i]["cuisine"]), str(df.iloc[i]["source"]))
            for i in range(len(df))
        ]
        corpus = [r.ing_text for r in self.records]
        self.M_word = self.vec_word.fit_transform(corpus)
        self.M_char = self.vec_char.fit_transform(corpus)

        vocab = self.vec_word.vocabulary_
        idfs = self.vec_word.idf_
        self.idf_lookup = {term: float(idfs[idx]) for term, idx in vocab.items()}
        self.idf_default = float(np.median(idfs)) if len(idfs) else 1.0
        return self

    def search(self, query: Union[str, List[str]], top_k: int = 10) -> List[Tuple[float, RecipeRecord]]:
        if isinstance(query, list):
            tokens_raw = [str(q) for q in query]
            tokens = []
            for t in tokens_raw:
                tokens.extend(_tokenize_norm(t))
        else:
            tokens = _tokenize_norm(str(query))

        tokens = sorted(set(tokens))
        if not tokens:
            return []

        q_text = " ".join(tokens)
        q_w = self.vec_word.transform([q_text])
        q_c = self.vec_char.transform([q_text])
        s_w = linear_kernel(q_w, self.M_word)[0]
        s_c = linear_kernel(q_c, self.M_char)[0]
        base_sim = 0.65 * s_w + 0.35 * s_c

        denom_idf = sum(self.idf_lookup.get(t, self.idf_default) for t in tokens) + 1e-6
        min_match = max(1, int(np.ceil(0.6 * len(tokens))))

        results = []
        token_set_q = set(tokens)
        for i, rec in enumerate(self.records):
            rec_tokens = rec.token_set
            matched = token_set_q & rec_tokens
            overlap = len(matched)
            coverage_idf = sum(self.idf_lookup.get(t, self.idf_default) for t in matched) / denom_idf
            union_sz = len(rec_tokens | token_set_q) + 1e-6
            jaccard = overlap / union_sz
            length_penalty = 1.0 / (1.0 + 0.02 * max(0, len(rec_tokens) - len(tokens)))

            score = float(base_sim[i])
            score += 0.75 * coverage_idf
            score += 0.10 * jaccard
            score += 0.05 * length_penalty

            if overlap == len(tokens):
                score += 0.50

            results.append((score, rec, overlap, coverage_idf))

        results.sort(key=lambda x: (x[2], x[0]), reverse=True)
        filtered = [r for r in results if r[2] >= min_match]
        ranked = filtered if filtered else results
        top = ranked[:max(top_k, 50)]
        return [(float(score), rec) for score, rec, _, _ in top]

# ---- Flask app ----
app = Flask(__name__)
CORS(app)  # allow cross origin for front-end dev; in production restrict origins

DATA_PATH = Path("final_indian_recipes.csv")  # must exist next to app.py

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH.resolve()}. Place final_indian_recipes.csv next to app.py")

def load_and_build_model(path: Path) -> IngredientSearch:
    df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    # unify columns similar to your notebook
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    name_col = next((c for c in ["name", "title", "recipe_name", "dish", "recipe"] if c in df.columns), None)
    ing_col = next((c for c in ["ingredients", "ingredient", "ingredients_name", "translatedingredients"] if c in df.columns), None)
    ins_col = next((c for c in ["instructions", "steps", "directions", "method", "translatedinstructions"] if c in df.columns), None)
    cuisine_col = next((c for c in ["recipecuisine", "cuisine", "recipe_cuisine"] if c in df.columns), None)
    source_col = next((c for c in ["url", "source", "link"] if c in df.columns), None)

    out = pd.DataFrame({
        "name": df[name_col] if name_col else [f"recipe_{i}" for i in range(len(df))],
        "ingredients": df[ing_col] if ing_col else "",
        "instructions": df[ins_col] if ins_col else "",
        "cuisine": df[cuisine_col] if cuisine_col else "",
        "source": df[source_col] if source_col else "",
    })
    for c in ["name", "ingredients", "instructions", "cuisine", "source"]:
        out[c] = out[c].astype(str).fillna("")

    model = IngredientSearch().fit(out)
    # attach the dataframe so we can access original raw columns if needed
    model.df = out
    return model

MODEL = load_and_build_model(DATA_PATH)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/search", methods=["POST"])
def search():
    """
    Request JSON: { "ingredients": "chicken, garlic", "cuisine": "indian", "top_k": 20 }
    Response JSON: { "results": [ { "score": 0.95, "name": "...", "ingredients": "...", "instructions": "...", "cuisine": "...", "source": "..." }, ... ] }
    """
    payload = request.get_json(force=True)
    ingredients = payload.get("ingredients", "") or ""
    cuisine = (payload.get("cuisine", "") or "").strip().lower()
    top_k = int(payload.get("top_k", 20))

    if not ingredients.strip():
        return jsonify({"results": [], "error": "empty ingredients"}), 400

    # perform search on TF-IDF model
    results = MODEL.search(ingredients, top_k=top_k)
    out = []
    for score, rec in results:
        # optional cuisine filter: if cuisine provided, penalize non-matching items
        score_adj = score
        if cuisine:
            if rec.cuisine and cuisine not in rec.cuisine.lower():
                score_adj *= 0.7
        out.append({
            "score": float(score_adj),
            "name": rec.name,
            "ingredients": rec.raw_ingredients,
            "instructions": rec.raw_instructions,
            "cuisine": rec.cuisine,
            "source": rec.source
        })

    # sort by score and return
    out_sorted = sorted(out, key=lambda r: r["score"], reverse=True)
    return jsonify({"results": out_sorted})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Starting Flask app on port", port)
    app.run(host="0.0.0.0", port=port, debug=True)
